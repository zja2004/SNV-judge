"""
predict.py — SNV-judge v5 Offline Prediction Script
=====================================================
直接调用训练好的模型文件，无需 Evo2 / Genos API Key。

工作流程：
  1. 加载本地 pkl 模型文件（xgb_model_v5.pkl 等）
  2. 通过 Ensembl VEP REST API 获取 SIFT/PolyPhen/AlphaMissense/CADD/phyloP（免费，无需 Key）
  3. 通过 gnomAD GraphQL API 获取人群频率（免费，无需 Key）
  4. Evo2 LLR 和 Genos Score 用训练集中位数填充（无需任何 API Key）
  5. 运行 XGBoost+LightGBM Stacking 集成模型 + Isotonic 校准
  6. 输出校准概率、ACMG 分级、SHAP 解释

新增功能（v5.1）：
  - 批量 VEP 注释（fetch_vep_batch，最多 200 变异/次）
  - VCF 文件支持（parse_vcf + predict_vcf）
  - SHAP 瀑布图可视化（plot_shap_waterfall）
  - 蛋白变异名称解析（resolve_protein_variant，如 "TP53 R175H"）
  - 模型版本特征数量验证（load_model_artifacts 自动检查）
  - 基因组上下文缓存（lru_cache，避免重复 API 调用）
  - ClinVar 实时查询（fetch_clinvar_classification）
  - CSV 批量输出（--output results.csv CLI 参数）

Usage:
    from scripts.predict import predict_variant, load_model_artifacts

    artifacts = load_model_artifacts(model_dir="/path/to/SNV-judge")
    result = predict_variant("17", 7674220, "C", "T", artifacts=artifacts)
    print(result["cal_prob"], result["acmg_class"])

    results_df = predict_vcf("variants.vcf", artifacts=artifacts, output_csv="results.csv")

    coords = resolve_protein_variant("TP53", "R175H")
    result = predict_variant(coords["chrom"], coords["pos"], coords["ref"], coords["alt"], artifacts=artifacts)

CLI:
    python predict.py 17 7674220 C T /path/to/SNV-judge
    python predict.py --vcf variants.vcf /path/to/SNV-judge --output results.csv
    python predict.py --gene TP53 --protein R175H /path/to/SNV-judge --clinvar
"""

import argparse
import csv
import gzip
import os
import pickle
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests

warnings.filterwarnings("ignore")

# ── 免费 API 端点（无需 Key）─────────────────────────────────────────────
VEP_URL      = "https://rest.ensembl.org/vep/homo_sapiens/region"
VEP_HDR      = {"Content-Type": "application/json", "Accept": "application/json"}
GNOMAD_URL   = "https://gnomad.broadinstitute.org/api"
ENSEMBL_BASE = "https://rest.ensembl.org"

# ── ACMG 5 级分类阈值 ─────────────────────────────────────────────────────
ACMG_TIERS = [
    (0.90, "Pathogenic (P)",         "High confidence"),
    (0.70, "Likely Pathogenic (LP)", "Moderate confidence"),
    (0.40, "VUS",                    "Uncertain significance"),
    (0.20, "Likely Benign (LB)",     "Moderate confidence"),
    (0.00, "Benign (B)",             "High confidence"),
]

# ── 特征名称（与模型训练时一致）──────────────────────────────────────────
FEATURE_NAMES_V5 = ["sift_inv", "polyphen", "alphamissense", "cadd",
                    "evo2_llr", "genos_score", "phylop", "gnomad_log_af"]
FEATURE_NAMES_V4 = ["sift_inv", "polyphen", "alphamissense", "cadd",
                    "phylop", "gnomad_log_af"]
FEATURE_LABELS_V5 = ["SIFT (inv)", "PolyPhen-2", "AlphaMissense", "CADD Phred",
                     "Evo2 LLR*", "Genos Score*", "phyloP", "gnomAD log-AF"]
FEATURE_LABELS_V4 = ["SIFT (inv)", "PolyPhen-2", "AlphaMissense", "CADD Phred",
                     "phyloP", "gnomAD log-AF"]
# * 表示该特征在离线模式下使用训练集中位数填充

VERSION_FEATURE_MAP = {
    "v5": (8, FEATURE_NAMES_V5, FEATURE_LABELS_V5),
    "v4": (6, FEATURE_NAMES_V4, FEATURE_LABELS_V4),
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. 模型加载（含版本特征数量验证）
# ═══════════════════════════════════════════════════════════════════════════

def load_model_artifacts(model_dir: str = ".") -> dict:
    """
    加载 v5 模型文件，自动降级到 v4/v3/v2/v1。
    新增：验证模型特征数量与版本一致性。

    Args:
        model_dir: SNV-judge 项目根目录（含 xgb_model_v5.pkl 等文件）

    Returns:
        dict 包含 model, medians, calibrator, version, n_features, feature_names, feature_labels
    """
    base = Path(model_dir)
    for suffix, ver in [("_v5", "v5"), ("_v4", "v4"), ("_v3", "v3"), ("_v2", "v2"), ("", "v1")]:
        xgb_path = base / f"xgb_model{suffix}.pkl"
        med_path  = base / f"train_medians{suffix}.pkl"
        cal_path  = base / f"platt_scaler{suffix}.pkl"
        if not (xgb_path.exists() and med_path.exists() and cal_path.exists()):
            continue
        with open(xgb_path, "rb") as f:
            model = pickle.load(f)
        with open(med_path, "rb") as f:
            medians = pickle.load(f)
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)

        # ── 特征数量验证 ──────────────────────────────────────────────────
        expected_n, feat_names, feat_labels = VERSION_FEATURE_MAP.get(
            ver, (8, FEATURE_NAMES_V5, FEATURE_LABELS_V5))
        actual_n = None
        try:
            xgb_m = model["xgb"] if isinstance(model, dict) else model
            actual_n = xgb_m.n_features_in_
            if actual_n != expected_n:
                print(f"⚠️  模型特征数量不匹配: 期望 {expected_n}，实际 {actual_n}。"
                      f"将使用实际特征数量 {actual_n}。")
                if actual_n == 6:
                    feat_names, feat_labels = FEATURE_NAMES_V4, FEATURE_LABELS_V4
                elif actual_n == 8:
                    feat_names, feat_labels = FEATURE_NAMES_V5, FEATURE_LABELS_V5
        except AttributeError:
            actual_n = expected_n

        print(f"✓ 模型加载成功: SNV-judge {ver} (XGB + LGB + LR meta + Isotonic 校准)")
        print(f"  模型路径: {xgb_path}")
        print(f"  特征数量: {actual_n}  特征: {feat_names}")
        return {
            "model":          model,
            "medians":        medians,
            "calibrator":     calibrator,
            "version":        ver,
            "n_features":     actual_n or expected_n,
            "feature_names":  feat_names,
            "feature_labels": feat_labels,
        }
    raise FileNotFoundError(
        f"未找到模型文件。请确认 model_dir='{model_dir}' 下存在 xgb_model_v*.pkl 等文件。\n"
        "如需重新训练：python train.py --use-cache"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. 免费 API 调用（VEP + gnomAD，无需任何 Key）
# ═══════════════════════════════════════════════════════════════════════════

def fetch_vep_scores(chrom: str, pos: int, ref: str, alt: str) -> dict:
    """
    调用 Ensembl VEP REST API 获取：
    SIFT · PolyPhen-2 · AlphaMissense · CADD · phyloP
    完全免费，无需 API Key。
    """
    variant_str = f"{chrom} {pos} . {ref} {alt} . . ."
    payload = {
        "variants":      [variant_str],
        "AlphaMissense": 1,
        "CADD":          1,
        "Conservation":  1,
        "canonical":     1,
        "mane":          1,
    }
    for attempt in range(3):
        try:
            r = requests.post(VEP_URL, headers=VEP_HDR, json=payload, timeout=30)
            if r.status_code == 200:
                break
            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 5))
                print(f"  VEP 限速，等待 {wait}s...")
                time.sleep(wait)
            else:
                return {"error": f"VEP HTTP {r.status_code}: {r.text[:100]}"}
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                return {"error": f"VEP 请求失败: {e}"}
    else:
        return {"error": "VEP API 不可用（3次重试后失败）"}

    data = r.json()
    if not data:
        return {"error": "VEP 无返回结果"}

    tcs = data[0].get("transcript_consequences", [])
    chosen = (next((tc for tc in tcs if tc.get("mane_select")), None)
              or next((tc for tc in tcs if tc.get("canonical") == 1), None)
              or (tcs[0] if tcs else None))
    if not chosen:
        return {"error": "未找到转录本后果"}

    csq = chosen.get("consequence_terms", [])
    am  = chosen.get("alphamissense", {})
    result = {
        "gene":             chosen.get("gene_symbol", ""),
        "transcript":       chosen.get("transcript_id", ""),
        "hgvsp":            chosen.get("hgvsp", ""),
        "consequence":      ", ".join(csq),
        "sift_score":       chosen.get("sift_score"),
        "sift_pred":        chosen.get("sift_prediction"),
        "polyphen_score":   chosen.get("polyphen_score"),
        "polyphen_pred":    chosen.get("polyphen_prediction"),
        "cadd_phred":       chosen.get("cadd_phred"),
        "am_pathogenicity": am.get("am_pathogenicity") if isinstance(am, dict) else None,
        "am_class":         am.get("am_class")         if isinstance(am, dict) else None,
        "phylop":           chosen.get("conservation"),
    }
    if "missense_variant" not in csq:
        result["warning"] = (f"变异类型为 {', '.join(csq)}，"
                             "非 missense_variant，评分可能不可靠")
    return result


def fetch_vep_batch(variants: list) -> list:
    """
    批量 VEP 注释（最多 200 变异/次）。
    比逐个调用快 10-50 倍。

    Args:
        variants: list of dict，每个含 chrom/pos/ref/alt

    Returns:
        list of VEP result dicts
    """
    if not variants:
        return []
    vep_strings = [
        f"{v['chrom']} {v['pos']} . {v['ref']} {v['alt']} . . ."
        for v in variants
    ]
    payload = {
        "variants":      vep_strings,
        "AlphaMissense": 1,
        "CADD":          1,
        "Conservation":  1,
        "canonical":     1,
        "mane":          1,
    }
    for attempt in range(3):
        try:
            r = requests.post(VEP_URL, headers=VEP_HDR, json=payload, timeout=120)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                print(f"  VEP 批量限速，等待 {wait}s...")
                time.sleep(wait)
            else:
                print(f"  VEP 批量 HTTP {r.status_code}")
                time.sleep(5)
        except Exception as e:
            print(f"  VEP 批量请求失败 (attempt {attempt+1}): {e}")
            time.sleep(5)
    return []


def fetch_gnomad_af(chrom: str, pos: int, ref: str, alt: str) -> float:
    """
    查询 gnomAD v4 等位基因频率（GraphQL API，免费无需 Key）。
    返回 log10(AF + 1e-8)，缺失时返回 np.nan。
    """
    query = """
    query V($vid: String!) {
      variant(variantId: $vid, dataset: gnomad_r4) {
        exome  { af }
        genome { af }
      }
    }
    """
    vid = f"{chrom}-{pos}-{ref}-{alt}"
    for attempt in range(3):
        try:
            r = requests.post(
                GNOMAD_URL,
                json={"query": query, "variables": {"vid": vid}},
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            if r.status_code == 200:
                d = r.json().get("data", {}).get("variant")
                if d:
                    vals = []
                    for src in ("exome", "genome"):
                        if d.get(src) and d[src].get("af") is not None:
                            vals.append(float(d[src]["af"]))
                    af = max(vals) if vals else 0.0
                else:
                    af = 0.0
                return float(np.log10(af + 1e-8))
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return np.nan


@lru_cache(maxsize=512)
def fetch_genomic_context(chrom: str, pos: int, flank: int = 50) -> str:
    """
    获取变异位点周围基因组序列（用于 Evo2 评分）。
    使用 lru_cache 缓存，避免重复 API 调用。

    Args:
        chrom: 染色体（不含 chr 前缀）
        pos:   位置（1-based，GRCh38）
        flank: 两侧各取 flank bp（默认 50）

    Returns:
        (2*flank+1) bp 序列字符串，失败时返回空字符串
    """
    start = pos - flank
    end   = pos + flank
    url   = f"{ENSEMBL_BASE}/sequence/region/human/{chrom}:{start}..{end}:1"
    hdrs  = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        r = requests.get(url, headers=hdrs, timeout=15)
        if r.status_code == 200:
            return r.json().get("seq", "")
    except Exception:
        pass
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# 3. ClinVar 实时查询
# ═══════════════════════════════════════════════════════════════════════════

def fetch_clinvar_classification(chrom: str, pos: int, ref: str, alt: str) -> dict:
    """
    通过 NCBI E-utilities 查询 ClinVar 临床意义分类。
    完全免费，无需 API Key。

    Args:
        chrom: 染色体（不含 chr 前缀，GRCh38）
        pos:   位置（1-based）
        ref:   参考等位基因
        alt:   替代等位基因

    Returns:
        dict 包含 clinvar_id, clinical_significance, review_status, conditions, last_evaluated
    """
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    try:
        # Step 1: 搜索 ClinVar ID（按染色体+位置）
        params = {
            "db":      "clinvar",
            "term":    f"{chrom}[Chromosome] AND {pos}[Base Position for Assembly GRCh38]",
            "retmode": "json",
            "retmax":  20,
        }
        r = requests.get(search_url, params=params, timeout=15)
        if r.status_code != 200:
            return {"error": f"ClinVar 搜索失败: HTTP {r.status_code}"}

        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"clinvar_id": None, "clinical_significance": "Not in ClinVar",
                    "review_status": "", "conditions": [], "last_evaluated": ""}

        # Step 2: 获取详细信息
        fetch_params = {
            "db":      "clinvar",
            "id":      ",".join(ids[:5]),
            "rettype": "vcv",
            "retmode": "json",
        }
        r2 = requests.get(fetch_url, params=fetch_params, timeout=15)
        if r2.status_code != 200:
            return {"error": f"ClinVar 获取失败: HTTP {r2.status_code}"}

        data = r2.json()
        result_set = data.get("ClinVarResult-Set", {})
        variants_list = result_set.get("VariationArchive", [])
        if isinstance(variants_list, dict):
            variants_list = [variants_list]

        for var in variants_list:
            interp = var.get("InterpretedRecord", {})
            allele = interp.get("SimpleAllele", {})
            loc_list = allele.get("Location", {}).get("SequenceLocation", [])
            if isinstance(loc_list, dict):
                loc_list = [loc_list]
            for loc in loc_list:
                if (loc.get("Assembly") == "GRCh38"
                        and str(loc.get("positionVCF")) == str(pos)
                        and loc.get("referenceAlleleVCF", "").upper() == ref.upper()
                        and loc.get("alternateAlleleVCF", "").upper() == alt.upper()):
                    interps = interp.get("Interpretations", {}).get("Interpretation", [])
                    if isinstance(interps, dict):
                        interps = [interps]
                    sig    = interps[0].get("Description", "") if interps else ""
                    status = interps[0].get("ReviewStatus", "") if interps else ""
                    date   = interps[0].get("DateLastEvaluated", "") if interps else ""
                    conds  = []
                    for cond in interp.get("ConditionList", {}).get("TraitSet", []):
                        if isinstance(cond, dict):
                            for trait in cond.get("Trait", []):
                                if isinstance(trait, dict):
                                    for name in trait.get("Name", []):
                                        if (isinstance(name, dict)
                                                and name.get("ElementValue", {}).get("Type") == "Preferred"):
                                            conds.append(name["ElementValue"].get("$", ""))
                    return {
                        "clinvar_id":            var.get("Accession", ""),
                        "clinical_significance": sig,
                        "review_status":         status,
                        "conditions":            conds,
                        "last_evaluated":        date,
                    }

        return {"clinvar_id": None, "clinical_significance": "Not found at this position",
                "review_status": "", "conditions": [], "last_evaluated": ""}

    except Exception as e:
        return {"error": f"ClinVar 查询异常: {e}"}


# ═══════════════════════════════════════════════════════════════════════════
# 4. 蛋白变异名称解析
# ═══════════════════════════════════════════════════════════════════════════

def resolve_protein_variant(gene: str, protein_change: str,
                             assembly: str = "GRCh38") -> dict:
    """
    将蛋白变异名称（如 "TP53 R175H"）解析为基因组坐标。
    使用 Ensembl HGVS REST API，完全免费。

    Args:
        gene:           基因名称（如 "TP53"、"BRCA1"）
        protein_change: 蛋白变化（如 "R175H"、"p.Arg175His"）
        assembly:       基因组版本（"GRCh38" 或 "GRCh37"）

    Returns:
        dict 包含 chrom, pos, ref, alt, hgvs_p, transcript, gene

    Example:
        coords = resolve_protein_variant("TP53", "R175H")
        result = predict_variant(coords["chrom"], coords["pos"],
                                 coords["ref"], coords["alt"], artifacts=artifacts)
    """
    import re

    pc = protein_change.strip()
    if not pc.startswith("p."):
        aa_map = {
            "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
            "E": "Glu", "Q": "Gln", "G": "Gly", "H": "His", "I": "Ile",
            "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
            "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
        }
        m = re.match(r"^([A-Z])(\d+)([A-Z*])$", pc)
        if m:
            ref_aa, pos_aa, alt_aa = m.group(1), m.group(2), m.group(3)
            ref_3 = aa_map.get(ref_aa, ref_aa)
            alt_3 = "Ter" if alt_aa == "*" else aa_map.get(alt_aa, alt_aa)
            pc = f"p.{ref_3}{pos_aa}{alt_3}"

    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}

    # Step 1: 获取基因的 MANE Select 转录本
    gene_url = f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{gene}"
    try:
        r = requests.get(gene_url, headers=hdrs, params={"expand": 1}, timeout=15)
        if r.status_code != 200:
            return {"error": f"基因查询失败: {gene} HTTP {r.status_code}"}
        gene_data = r.json()
        transcripts = gene_data.get("Transcript", [])
        transcript_id = None
        for t in transcripts:
            if t.get("is_mane_select"):
                transcript_id = t["id"]
                break
        if not transcript_id:
            for t in transcripts:
                if t.get("is_canonical"):
                    transcript_id = t["id"]
                    break
        if not transcript_id and transcripts:
            transcript_id = transcripts[0]["id"]
        if not transcript_id:
            return {"error": f"未找到 {gene} 的转录本"}
    except Exception as e:
        return {"error": f"基因查询异常: {e}"}

    # Step 2: HGVS 解析
    hgvs_notation = f"{transcript_id}:{pc}"
    hgvs_url = f"{ENSEMBL_BASE}/variant_recoder/homo_sapiens/{hgvs_notation}"
    try:
        r = requests.get(hgvs_url, headers=hdrs, timeout=15)
        if r.status_code != 200:
            return {"error": f"HGVS 解析失败: {hgvs_notation} HTTP {r.status_code}"}
        recoder_data = r.json()
        if not recoder_data:
            return {"error": f"HGVS 无结果: {hgvs_notation}"}

        entry = recoder_data[0] if isinstance(recoder_data, list) else recoder_data
        vcf_strings = []
        for key, val in entry.items():
            if isinstance(val, dict):
                vcf_list = val.get("vcf_string", [])
                if vcf_list:
                    vcf_strings.extend(vcf_list if isinstance(vcf_list, list) else [vcf_list])

        if not vcf_strings:
            return {"error": f"无法获取 VCF 坐标: {hgvs_notation}"}

        vcf_str = vcf_strings[0]
        parts = vcf_str.split("-")
        if len(parts) < 4:
            return {"error": f"VCF 格式异常: {vcf_str}"}

        chrom = parts[0].replace("chr", "")
        pos   = int(parts[1])
        ref   = parts[2].upper()
        alt   = parts[3].upper()

        print(f"✓ 蛋白变异解析: {gene} {protein_change} → chr{chrom}:{pos} {ref}>{alt}")
        return {
            "chrom":      chrom,
            "pos":        pos,
            "ref":        ref,
            "alt":        alt,
            "hgvs_p":     pc,
            "transcript": transcript_id,
            "gene":       gene,
        }
    except Exception as e:
        return {"error": f"HGVS 解析异常: {e}"}


# ═══════════════════════════════════════════════════════════════════════════
# 5. VCF 文件解析
# ═══════════════════════════════════════════════════════════════════════════

def parse_vcf(vcf_path: str) -> list:
    """
    解析 VCF 文件（支持普通文本和 gzip 压缩），提取 SNV 变异列表。

    Args:
        vcf_path: VCF 文件路径（.vcf 或 .vcf.gz）

    Returns:
        list of dict，每个含 chrom/pos/ref/alt
    """
    path = Path(vcf_path)
    if not path.exists():
        raise FileNotFoundError(f"VCF 文件不存在: {vcf_path}")

    with open(path, "rb") as f:
        raw = f.read()

    if raw[:2] == b"\x1f\x8b":
        text = gzip.decompress(raw).decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")

    variants = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        chrom, pos_str, _, ref, alt_field = parts[0], parts[1], parts[2], parts[3], parts[4]
        chrom = chrom.replace("chr", "")
        alt = alt_field.split(",")[0]
        if len(ref) == 1 and len(alt) == 1 and ref.upper() != alt.upper():
            try:
                variants.append({
                    "chrom": chrom,
                    "pos":   int(pos_str),
                    "ref":   ref.upper(),
                    "alt":   alt.upper(),
                })
            except ValueError:
                continue

    print(f"✓ VCF 解析完成: {len(variants)} 个 SNV（来自 {path.name}）")
    return variants


# ═══════════════════════════════════════════════════════════════════════════
# 6. 核心预测函数
# ═══════════════════════════════════════════════════════════════════════════

def _get_acmg_tier(prob: float) -> dict:
    for thresh, label, confidence in ACMG_TIERS:
        if prob >= thresh:
            return {"label": label, "confidence": confidence, "prob": prob}
    return {"label": "Benign (B)", "confidence": "High confidence", "prob": prob}


def _extract_vep_scores_from_raw(vep_raw: dict) -> dict:
    """从 VEP 批量结果中提取单个变异的评分字段。"""
    tcs = vep_raw.get("transcript_consequences", [])
    chosen = (next((tc for tc in tcs if tc.get("mane_select")), None)
              or next((tc for tc in tcs if tc.get("canonical") == 1), None)
              or (tcs[0] if tcs else None))
    if not chosen:
        return {}
    am  = chosen.get("alphamissense", {})
    csq = chosen.get("consequence_terms", [])
    return {
        "gene":             chosen.get("gene_symbol", ""),
        "transcript":       chosen.get("transcript_id", ""),
        "hgvsp":            chosen.get("hgvsp", ""),
        "consequence":      ", ".join(csq),
        "sift_score":       chosen.get("sift_score"),
        "sift_pred":        chosen.get("sift_prediction"),
        "polyphen_score":   chosen.get("polyphen_score"),
        "polyphen_pred":    chosen.get("polyphen_prediction"),
        "cadd_phred":       chosen.get("cadd_phred"),
        "am_pathogenicity": am.get("am_pathogenicity") if isinstance(am, dict) else None,
        "am_class":         am.get("am_class")         if isinstance(am, dict) else None,
        "phylop":           chosen.get("conservation"),
    }


def predict_from_scores(vep_scores: dict,
                         gnomad_log_af: float,
                         artifacts: dict) -> dict:
    """
    用 VEP 评分 + gnomAD AF 直接预测（Evo2/Genos 用训练中位数填充）。

    Args:
        vep_scores:    fetch_vep_scores() 的返回值
        gnomad_log_af: fetch_gnomad_af() 的返回值
        artifacts:     load_model_artifacts() 的返回值

    Returns:
        dict 包含 cal_prob, acmg_class, shap_values, feature_vec 等
    """
    model       = artifacts["model"]
    medians     = artifacts["medians"]
    calibrator  = artifacts["calibrator"]
    feat_labels = artifacts.get("feature_labels", FEATURE_LABELS_V5)
    n_features  = artifacts.get("n_features", 8)

    def _val(key, med_key=None):
        v = vep_scores.get(key)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
        return float(medians.get(med_key or key, 0.0))

    def _impute(val, med_key):
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return float(val)
        return float(medians.get(med_key, 0.0))

    sift_score = vep_scores.get("sift_score")
    sift_inv   = (1.0 - float(sift_score)) if sift_score is not None else float(medians.get("sift_inv", 1.0))

    if n_features == 8:
        feature_vec = [
            sift_inv,
            _val("polyphen_score", "polyphen"),
            _val("am_pathogenicity", "alphamissense"),
            _val("cadd_phred", "cadd"),
            float(medians.get("evo2_llr", -0.074)),
            float(medians.get("genos_score", 0.676)),
            _val("phylop", "phylop"),
            _impute(gnomad_log_af, "gnomad_log_af"),
        ]
    else:
        feature_vec = [
            sift_inv,
            _val("polyphen_score", "polyphen"),
            _val("am_pathogenicity", "alphamissense"),
            _val("cadd_phred", "cadd"),
            _val("phylop", "phylop"),
            _impute(gnomad_log_af, "gnomad_log_af"),
        ]

    X = np.array([feature_vec])

    if isinstance(model, dict):
        xgb_p    = model["xgb"].predict_proba(X)[0, 1]
        lgb_p    = model["lgb"].predict_proba(X)[0, 1]
        raw_prob = float(model["meta"].predict_proba(np.array([[xgb_p, lgb_p]]))[0, 1])
    else:
        raw_prob = float(model.predict_proba(X)[0, 1])

    from sklearn.isotonic import IsotonicRegression
    if isinstance(calibrator, IsotonicRegression):
        cal_prob = float(np.clip(calibrator.predict([raw_prob])[0], 0.0, 1.0))
    else:
        logit    = np.log(raw_prob / (1 - raw_prob + 1e-9))
        cal_prob = float(calibrator.predict_proba([[logit]])[0, 1])

    acmg = _get_acmg_tier(cal_prob)

    try:
        import shap
        xgb_model = model["xgb"] if isinstance(model, dict) else model
        explainer  = shap.TreeExplainer(xgb_model)
        shap_vals  = explainer.shap_values(X)[0].tolist()
    except Exception:
        shap_vals = [0.0] * len(feature_vec)

    top_idx = int(np.argmax(np.abs(shap_vals)))

    return {
        "cal_prob":         cal_prob,
        "raw_prob":         raw_prob,
        "acmg_class":       acmg["label"],
        "acmg_confidence":  acmg["confidence"],
        "shap_values":      shap_vals,
        "feature_labels":   feat_labels,
        "feature_vec":      feature_vec,
        "top_shap_feature": feat_labels[top_idx] if top_idx < len(feat_labels) else "",
        "offline_mode":     True,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 7. SHAP 瀑布图可视化
# ═══════════════════════════════════════════════════════════════════════════

def plot_shap_waterfall(result: dict,
                         title: str = None,
                         save_path: str = None,
                         figsize: tuple = (9, 5)):
    """
    绘制 SHAP 瀑布图，展示各特征对预测结果的贡献。

    Args:
        result:    predict_variant() 或 predict_from_scores() 的返回值
        title:     图标题（默认自动生成）
        save_path: 保存路径（如 "shap_waterfall.svg"），None 则不保存
        figsize:   图形尺寸

    Returns:
        matplotlib Figure 对象
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    shap_vals = result["shap_values"]
    labels    = result["feature_labels"]
    cal_prob  = result["cal_prob"]
    acmg      = result["acmg_class"]

    order         = np.argsort(np.abs(shap_vals))
    sorted_labels = [labels[i] for i in order]
    sorted_vals   = [shap_vals[i] for i in order]
    colors        = ["#D55E00" if v > 0 else "#0072B2" for v in sorted_vals]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(sorted_labels, sorted_vals, color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, sorted_vals):
        ha = "left" if val >= 0 else "right"
        x  = val + (0.002 if val >= 0 else -0.002)
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", ha=ha, fontsize=9)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (contribution to pathogenicity log-odds)", fontsize=9)

    if title is None:
        chrom = result.get("chrom", "?")
        pos   = result.get("pos", "?")
        ref   = result.get("ref", "?")
        alt   = result.get("alt", "?")
        gene  = result.get("gene", "")
        hgvsp = result.get("hgvsp", "")
        title = (f"SHAP Feature Contributions\n"
                 f"chr{chrom}:{pos} {ref}>{alt}"
                 + (f"  {gene} {hgvsp}" if gene else "")
                 + f"\nPrediction: {cal_prob:.1%}  →  {acmg}")
    ax.set_title(title, fontsize=10, fontweight="bold")

    red_patch  = mpatches.Patch(color="#D55E00", label="→ Pathogenic")
    blue_patch = mpatches.Patch(color="#0072B2", label="→ Benign")
    ax.legend(handles=[red_patch, blue_patch], fontsize=8, loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ SHAP 瀑布图已保存: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 8. 高层接口（推荐使用）
# ═══════════════════════════════════════════════════════════════════════════

def predict_variant(chrom: str, pos: int, ref: str, alt: str,
                     artifacts: dict = None,
                     model_dir: str = ".") -> dict:
    """
    完整预测流程：VEP + gnomAD → 模型预测 → ACMG 分级 + SHAP。
    无需 Evo2 / Genos API Key。

    Args:
        chrom:     染色体（如 "17"，不含 "chr" 前缀，GRCh38）
        pos:       位置（1-based）
        ref:       参考等位基因（单碱基）
        alt:       替代等位基因（单碱基）
        artifacts: load_model_artifacts() 的返回值（可复用，避免重复加载）
        model_dir: 模型文件目录（artifacts 为 None 时使用）

    Returns:
        dict 包含 cal_prob, acmg_class, acmg_confidence, shap_values,
                  top_shap_feature, gene, hgvsp, feature_vec, offline_mode
    """
    chrom = str(chrom).replace("chr", "")
    ref, alt = ref.upper().strip(), alt.upper().strip()

    if artifacts is None:
        artifacts = load_model_artifacts(model_dir)

    print(f"正在获取 {chrom}:{pos} {ref}>{alt} 的注释信息...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_vep    = ex.submit(fetch_vep_scores, chrom, pos, ref, alt)
        fut_gnomad = ex.submit(fetch_gnomad_af,  chrom, pos, ref, alt)
        vep_scores    = fut_vep.result()
        gnomad_log_af = fut_gnomad.result()

    if "error" in vep_scores:
        raise RuntimeError(f"VEP 注释失败: {vep_scores['error']}")

    if vep_scores.get("warning"):
        print(f"⚠️  {vep_scores['warning']}")

    result = predict_from_scores(vep_scores, gnomad_log_af, artifacts)
    result.update({
        "chrom":         chrom,
        "pos":           pos,
        "ref":           ref,
        "alt":           alt,
        "gene":          vep_scores.get("gene", ""),
        "transcript":    vep_scores.get("transcript", ""),
        "hgvsp":         vep_scores.get("hgvsp", ""),
        "consequence":   vep_scores.get("consequence", ""),
        "am_class":      vep_scores.get("am_class", ""),
        "sift_pred":     vep_scores.get("sift_pred", ""),
        "gnomad_log_af": gnomad_log_af,
    })

    print(f"\n{'='*55}")
    print(f"变异: chr{chrom}:{pos} {ref}>{alt}")
    print(f"基因: {result['gene']}  蛋白变化: {result['hgvsp']}")
    print(f"致病概率: {result['cal_prob']:.1%}  →  {result['acmg_class']} ({result['acmg_confidence']})")
    print(f"主要贡献特征: {result['top_shap_feature']}")
    print(f"注: Evo2 LLR 和 Genos Score 使用训练集中位数填充（离线模式）")
    print(f"{'='*55}")

    return result


def predict_vcf(vcf_path: str,
                artifacts: dict = None,
                model_dir: str = ".",
                output_csv: str = None,
                batch_size: int = 50,
                verbose: bool = True) -> list:
    """
    批量预测 VCF 文件中的所有 SNV。
    使用批量 VEP 注释（最多 batch_size 变异/次），效率远高于逐个预测。

    Args:
        vcf_path:   VCF 文件路径（.vcf 或 .vcf.gz）
        artifacts:  load_model_artifacts() 的返回值
        model_dir:  模型文件目录（artifacts 为 None 时使用）
        output_csv: 结果 CSV 保存路径（None 则不保存）
        batch_size: 每批 VEP 注释的变异数量（最大 200）
        verbose:    是否打印进度

    Returns:
        list of dict，每个含完整预测结果
    """
    if artifacts is None:
        artifacts = load_model_artifacts(model_dir)

    variants = parse_vcf(vcf_path)
    if not variants:
        print("⚠️  VCF 文件中未找到有效 SNV")
        return []

    total = len(variants)
    if verbose:
        print(f"开始批量预测 {total} 个变异（批大小: {batch_size}）...")

    # Step 1: 批量 VEP 注释
    vep_map = {}
    for i in range(0, total, batch_size):
        batch = variants[i: i + batch_size]
        if verbose:
            print(f"  VEP 批量注释: {min(i+batch_size, total)}/{total}")
        raw_results = fetch_vep_batch(batch)
        for res in raw_results:
            parts = res.get("input", "").split()
            if len(parts) >= 5:
                key = (parts[0], int(parts[1]), parts[3], parts[4])
                vep_map[key] = res
        if i + batch_size < total:
            time.sleep(0.5)

    # Step 2: 并行获取 gnomAD AF + 预测
    def process_one(v):
        key        = (v["chrom"], v["pos"], v["ref"], v["alt"])
        vep_raw    = vep_map.get(key, {})
        vep_scores = _extract_vep_scores_from_raw(vep_raw) if vep_raw else {}
        gnomad_log_af = fetch_gnomad_af(v["chrom"], v["pos"], v["ref"], v["alt"])
        pred = predict_from_scores(vep_scores, gnomad_log_af, artifacts)
        pred.update({
            "chrom":         v["chrom"],
            "pos":           v["pos"],
            "ref":           v["ref"],
            "alt":           v["alt"],
            "gene":          vep_scores.get("gene", ""),
            "hgvsp":         vep_scores.get("hgvsp", ""),
            "consequence":   vep_scores.get("consequence", ""),
            "gnomad_log_af": gnomad_log_af,
        })
        return pred

    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(process_one, v) for v in variants]
        for i, fut in enumerate(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                v = variants[i]
                results.append({
                    "chrom": v["chrom"], "pos": v["pos"],
                    "ref": v["ref"], "alt": v["alt"],
                    "error": str(e), "cal_prob": None, "acmg_class": "Error",
                })
            if verbose and (i + 1) % 10 == 0:
                print(f"  预测进度: {i+1}/{total}")

    if output_csv:
        save_results_csv(results, output_csv)

    if verbose:
        print(f"\n✓ 批量预测完成: {len(results)} 个变异")
        pathogenic = sum(1 for r in results
                         if r.get("acmg_class", "").startswith(("Pathogenic", "Likely Pathogenic")))
        vus_count  = sum(1 for r in results if r.get("acmg_class") == "VUS")
        print(f"  致病/可能致病: {pathogenic}  VUS: {vus_count}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 9. CSV 输出
# ═══════════════════════════════════════════════════════════════════════════

def save_results_csv(results: list, output_path: str):
    """
    将预测结果列表保存为 CSV 文件。

    Args:
        results:     predict_variant() 或 predict_vcf() 的返回值列表
        output_path: CSV 文件保存路径
    """
    if not results:
        print("⚠️  无结果可保存")
        return

    fieldnames = [
        "chrom", "pos", "ref", "alt",
        "gene", "hgvsp", "consequence",
        "cal_prob", "acmg_class", "acmg_confidence",
        "top_shap_feature", "gnomad_log_af",
        "offline_mode", "error",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            if row.get("cal_prob") is not None:
                try:
                    row["cal_prob"] = f"{float(row['cal_prob']):.4f}"
                except (TypeError, ValueError):
                    pass
            writer.writerow(row)

    print(f"✓ 结果已保存: {output_path}  ({len(results)} 行)")


# ═══════════════════════════════════════════════════════════════════════════
# 10. 辅助打印函数
# ═══════════════════════════════════════════════════════════════════════════

def print_shap_summary(result: dict):
    """打印 SHAP 特征贡献摘要（文本版）。"""
    print("\nSHAP 特征贡献（正值→致病，负值→良性）:")
    pairs = sorted(zip(result["feature_labels"], result["shap_values"]),
                   key=lambda x: abs(x[1]), reverse=True)
    for label, sv in pairs:
        bar       = "█" * int(abs(sv) * 30)
        direction = "→致病" if sv > 0 else "→良性"
        offline   = " [中位数填充]" if "*" in label else ""
        print(f"  {label:<18} {sv:+.4f} {direction}  {bar}{offline}")


# ═══════════════════════════════════════════════════════════════════════════
# 11. CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SNV-judge v5 — 变异致病性预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单变异预测
  python predict.py 17 7674220 C T /path/to/SNV-judge

  # VCF 批量预测
  python predict.py --vcf variants.vcf /path/to/SNV-judge --output results.csv

  # 蛋白变异名称解析后预测
  python predict.py --gene TP53 --protein R175H /path/to/SNV-judge

  # 附带 ClinVar 查询 + SHAP 图
  python predict.py 17 7674220 C T /path/to/SNV-judge --clinvar --shap shap.svg
        """
    )

    parser.add_argument("chrom",     nargs="?", help="染色体（如 17）")
    parser.add_argument("pos",       nargs="?", type=int, help="位置（1-based）")
    parser.add_argument("ref",       nargs="?", help="参考等位基因")
    parser.add_argument("alt",       nargs="?", help="替代等位基因")
    parser.add_argument("model_dir", nargs="?", default=".", help="模型文件目录")

    parser.add_argument("--vcf",        help="VCF 文件路径（批量预测模式）")
    parser.add_argument("--gene",       help="基因名称（蛋白变异解析模式）")
    parser.add_argument("--protein",    help="蛋白变化（如 R175H）")
    parser.add_argument("--output",     help="CSV 输出文件路径")
    parser.add_argument("--shap",       help="SHAP 瀑布图保存路径（如 shap.svg）")
    parser.add_argument("--clinvar",    action="store_true", help="同时查询 ClinVar 分类")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="VCF 批量 VEP 大小（默认 50，最大 200）")

    args = parser.parse_args()

    # ── 模式 1: VCF 批量预测 ──────────────────────────────────────────────
    if args.vcf:
        model_dir = args.model_dir or "."
        artifacts = load_model_artifacts(model_dir)
        results   = predict_vcf(args.vcf, artifacts=artifacts,
                                 output_csv=args.output,
                                 batch_size=args.batch_size)
        if not args.output:
            print("\n前 5 个结果:")
            for r in results[:5]:
                prob_str = f"{r['cal_prob']:.1%}" if r.get("cal_prob") is not None else "N/A"
                print(f"  chr{r['chrom']}:{r['pos']} {r['ref']}>{r['alt']}  "
                      f"{r.get('gene','')}  {prob_str}  {r.get('acmg_class','')}")

    # ── 模式 2: 蛋白变异名称解析 ─────────────────────────────────────────
    elif args.gene and args.protein:
        coords = resolve_protein_variant(args.gene, args.protein)
        if "error" in coords:
            print(f"❌ 解析失败: {coords['error']}")
            import sys; sys.exit(1)
        model_dir = args.model_dir or "."
        artifacts = load_model_artifacts(model_dir)
        result    = predict_variant(coords["chrom"], coords["pos"],
                                     coords["ref"], coords["alt"],
                                     artifacts=artifacts)
        print_shap_summary(result)
        if args.shap:
            plot_shap_waterfall(result, save_path=args.shap)
        if args.clinvar:
            cv = fetch_clinvar_classification(coords["chrom"], coords["pos"],
                                              coords["ref"], coords["alt"])
            print(f"\nClinVar: {cv.get('clinical_significance', 'N/A')}  "
                  f"({cv.get('review_status', '')})")
        if args.output:
            save_results_csv([result], args.output)

    # ── 模式 3: 单变异预测（位置参数）────────────────────────────────────
    elif args.chrom and args.pos and args.ref and args.alt:
        artifacts = load_model_artifacts(args.model_dir)
        result    = predict_variant(args.chrom, args.pos, args.ref, args.alt,
                                     artifacts=artifacts)
        print_shap_summary(result)
        if args.shap:
            plot_shap_waterfall(result, save_path=args.shap)
        if args.clinvar:
            cv = fetch_clinvar_classification(args.chrom, args.pos, args.ref, args.alt)
            print(f"\nClinVar: {cv.get('clinical_significance', 'N/A')}  "
                  f"({cv.get('review_status', '')})")
        if args.output:
            save_results_csv([result], args.output)

    else:
        parser.print_help()
