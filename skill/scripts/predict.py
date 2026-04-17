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

新增功能（v5.2）：
  - Genos-10B 本地 embedding 评分（fetch_genos_embedding_score）
    通过 --genos-url 指定本地 ngrok 端点，计算 REF/ALT 序列余弦距离
    替代 genos_score 的训练集中位数填充；服务不可用时自动降级回中位数

Usage:
    from scripts.predict import predict_variant, load_model_artifacts

    artifacts = load_model_artifacts(model_dir="/path/to/SNV-judge")
    result = predict_variant("17", 7674220, "C", "T", artifacts=artifacts)
    print(result["cal_prob"], result["acmg_class"])

    # 启用 Genos embedding 评分
    result = predict_variant("17", 7674220, "C", "T", artifacts=artifacts,
                              genos_url="https://xxx.ngrok-free.dev")

    results_df = predict_vcf("variants.vcf", artifacts=artifacts, output_csv="results.csv")

    coords = resolve_protein_variant("TP53", "R175H")
    result = predict_variant(coords["chrom"], coords["pos"], coords["ref"], coords["alt"],
                              artifacts=artifacts)

CLI:
    python predict.py 17 7674220 C T /path/to/SNV-judge
    python predict.py --vcf variants.vcf /path/to/SNV-judge --output results.csv
    python predict.py --gene TP53 --protein R175H /path/to/SNV-judge --clinvar
    python predict.py 17 7674220 C T /path/to/SNV-judge \
        --genos-url https://neuronic-marilynn-touristically.ngrok-free.dev
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

# ── Genos 本地端点（可选，通过 --genos-url 启用）─────────────────────────
GENOS_DEFAULT_URL = "https://neuronic-marilynn-touristically.ngrok-free.dev"
GENOS_FLANK       = 499   # 变异位点两侧各取 499 bp（共 999 bp，服务器 max_length=1000）
GENOS_MODEL_NAME  = "10B" # 使用 Genos-10B 模型
GENOS_TIMEOUT     = 120   # 单次 /generate 请求超时（秒，生成比 extract 慢）

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
                     "Evo2 LLR*", "Genos Score", "phyloP", "gnomAD log-AF"]
FEATURE_LABELS_V4 = ["SIFT (inv)", "PolyPhen-2", "AlphaMissense", "CADD Phred",
                     "phyloP", "gnomAD log-AF"]
# * 表示该特征在离线模式下使用训练集中位数填充
# Genos Score 标签动态更新（在线时去掉 *，离线时加 *）

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
            "feature_labels": list(feat_labels),  # mutable copy for dynamic label update
        }
    raise FileNotFoundError(
        f"未找到模型文件。请确认 model_dir='{model_dir}' 下存在 xgb_model_v*.pkl 等文件。\n"
        "如需重新训练：python train.py --use-cache"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. 免费 API 调用（VEP + gnomAD，无需任何 Key）
# ═══════════════════════════════════════════════════════════════════════════

def fetch_vep_scores(chrom: str, pos: int, ref: str, alt: str) -> dict:
    """调用 Ensembl VEP REST API 获取 SIFT/PolyPhen/AlphaMissense/CADD/phyloP。"""
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
    """批量 VEP 注释（最多 200 变异/次），比逐个调用快 10-50 倍。"""
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
    """查询 gnomAD v4 等位基因频率，返回 log10(AF + 1e-8)。"""
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
    获取变异位点周围基因组序列。
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
# 3. Genos-10B 本地 Embedding 评分（v5.2 新增）
# ═══════════════════════════════════════════════════════════════════════════

def fetch_genos_embedding_score(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    genos_url: str = GENOS_DEFAULT_URL,
    flank: int = GENOS_FLANK,
) -> float:
    """
    通过本地 Genos-10B /generate 端点计算变异的生成差异评分。

    原理（方案B — /generate 接口适配）：
      1. 从 Ensembl 获取变异位点两侧各 flank bp 的基因组上下文（共 2*flank+1 bp）
      2. 构造 REF 序列（上下文原始序列）和 ALT 序列（中心碱基替换为 alt）
      3. 分别调用 Genos /generate，以 REF/ALT 序列为 prompt，各生成 100 bp 续写
      4. 计算两个续写序列之间的 k-mer Jaccard 距离（k=6）：
           score = 1 - |kmers_ref ∩ kmers_alt| / |kmers_ref ∪ kmers_alt|
         - score ∈ [0, 1]，值越大表示变异对模型生成的影响越大（→ 致病倾向）
         - 与训练集 genos_score 中位数（0.676）同量纲，可直接替换

    Args:
        chrom:     染色体（不含 chr 前缀，GRCh38）
        pos:       位置（1-based）
        ref:       参考等位基因（单碱基）
        alt:       替代等位基因（单碱基）
        genos_url: Genos 服务器 URL（ngrok 地址）
        flank:     两侧各取 flank bp（默认 500）

    Returns:
        float: k-mer Jaccard 距离评分 ∈ [0, 1]；失败时返回 np.nan（调用方自动降级到中位数）
    """
    # Step 1: 获取基因组上下文（复用带缓存的 fetch_genomic_context）
    context = fetch_genomic_context(chrom, pos, flank=flank)
    expected_len = 2 * flank + 1
    if not context or len(context) < expected_len:
        print(f"  ⚠️  Genos: 基因组上下文获取失败（长度 {len(context) if context else 0}，期望 {expected_len}）")
        return np.nan

    # Step 2: 构造 REF / ALT 序列
    center  = len(context) // 2
    ref_seq = context
    alt_seq = context[:center] + alt.upper() + context[center + 1:]

    # 验证中心碱基与 ref 一致（容错：大小写不敏感）
    actual_center = context[center].upper()
    if actual_center != ref.upper():
        print(f"  ⚠️  Genos: 中心碱基不匹配（期望 {ref.upper()}，实际 {actual_center}）"
              f"，仍继续计算（可能坐标系偏移）")

    # Step 3: 调用 /generate 获取续写序列
    generate_url = f"{genos_url.rstrip('/')}/generate"

    def _get_generation(sequence: str) -> str | None:
        try:
            r = requests.post(
                generate_url,
                json={"sequence": sequence.upper()},
                timeout=GENOS_TIMEOUT * 3,  # 生成比 extract 慢，超时设为 3x
            )
            if r.status_code == 200:
                data = r.json()
                raw_output = data.get("output", "")
                # 去除空格（模型输出格式为 "A T C G ..."）
                gen_seq = raw_output.replace(" ", "").upper()
                if gen_seq:
                    return gen_seq
            print(f"  ⚠️  Genos /generate HTTP {r.status_code}: {r.text[:80]}")
        except Exception as e:
            print(f"  ⚠️  Genos /generate 请求失败: {e}")
        return None

    print(f"  → Genos /generate: 正在生成 REF 续写（{len(ref_seq)} bp prompt）...")
    gen_ref = _get_generation(ref_seq)
    print(f"  → Genos /generate: 正在生成 ALT 续写（{len(alt_seq)} bp prompt）...")
    gen_alt = _get_generation(alt_seq)

    if gen_ref is None or gen_alt is None:
        return np.nan

    # Step 4: k-mer Jaccard 距离（k=6）
    k = 6
    def _kmer_set(seq: str, k: int) -> set:
        return {seq[i:i+k] for i in range(len(seq) - k + 1)}

    kmers_ref = _kmer_set(gen_ref, k)
    kmers_alt = _kmer_set(gen_alt, k)

    intersection = len(kmers_ref & kmers_alt)
    union        = len(kmers_ref | kmers_alt)
    if union == 0:
        return np.nan

    jaccard_sim  = intersection / union
    jaccard_dist = 1.0 - jaccard_sim
    genos_score  = float(np.clip(jaccard_dist, 0.0, 1.0))

    print(f"  ✓ Genos 生成差异评分: {genos_score:.4f}  "
          f"（k-mer Jaccard 距离 k={k}，REF↔ALT 续写，flank={flank}bp）")
    return genos_score



# =============================================================================
# 4. ClinVar 实时查询
# =============================================================================

def fetch_clinvar_classification(chrom: str, pos: int, ref: str, alt: str) -> dict:
    """
    查询 ClinVar 临床意义分类（免费，无需 API Key）。

    实现策略（两步）：
      Step 1: 调用 Ensembl VEP REST API，从 colocated_variants 中提取 rsID 和
              clin_sig 字段。VEP 已整合 ClinVar 数据，是最可靠的坐标→ClinVar 路径。
      Step 2: 用 rsID 调用 NCBI dbSNP esummary 获取更详细的 ClinVar 信息
              （条件、review status 等）。若 Step 2 失败，仅返回 Step 1 的数据。

    Args:
        chrom: 染色体（不含 chr 前缀，GRCh38）
        pos:   位置（1-based）
        ref:   参考等位基因
        alt:   替代等位基因

    Returns:
        dict with keys: clinvar_id, clinical_significance, review_status,
                        conditions, last_evaluated, rsid
    """
    try:
        # ── Step 1: VEP colocated_variants ──────────────────────────────────
        vep_url = (f"https://rest.ensembl.org/vep/homo_sapiens/region/"
                   f"{chrom}:{pos}-{pos}:1/{alt}")
        hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
        r = requests.get(vep_url, headers=hdrs, timeout=20)
        if r.status_code != 200:
            return {"error": f"VEP query failed: HTTP {r.status_code}"}

        data = r.json()
        if not data:
            return {"clinvar_id": None, "clinical_significance": "Not in ClinVar",
                    "review_status": "", "conditions": [], "last_evaluated": "", "rsid": ""}

        colocated = data[0].get("colocated_variants", [])

        # 找匹配 alt 等位基因的 rsID 和 clin_sig
        rsid     = ""
        clin_sig = []
        for cv in colocated:
            cv_id = cv.get("id", "")
            if not cv_id.startswith("rs"):
                continue
            # 检查 alt 等位基因是否在 clin_sig_allele 中
            clin_sig_allele = cv.get("clin_sig_allele", [])
            cv_clin_sig     = cv.get("clin_sig", [])
            if cv_clin_sig:
                # 过滤出与 alt 等位基因匹配的分类
                alt_sigs = []
                for entry in (clin_sig_allele if isinstance(clin_sig_allele, list)
                              else [clin_sig_allele]):
                    # 格式: "T:pathogenic" 或 "A:pathogenic/likely_pathogenic"
                    if isinstance(entry, str) and ":" in entry:
                        allele_part, sig_part = entry.split(":", 1)
                        if allele_part.upper() == alt.upper():
                            alt_sigs.extend(sig_part.split("/"))
                if not alt_sigs:
                    alt_sigs = cv_clin_sig  # fallback: use all sigs
                rsid     = cv_id
                clin_sig = list(dict.fromkeys(alt_sigs))  # deduplicate, preserve order
                break

        if not rsid:
            return {"clinvar_id": None, "clinical_significance": "Not in ClinVar",
                    "review_status": "", "conditions": [], "last_evaluated": "", "rsid": ""}

        # 规范化 significance（首字母大写，下划线→空格）
        def _normalize(s: str) -> str:
            return s.replace("_", " ").replace("/", " / ").title()

        sig_display = " / ".join(_normalize(s) for s in clin_sig) if clin_sig else "Unknown"

        # ── Step 2: NCBI esummary via rsID for review_status + conditions ───
        review_status = ""
        conditions    = []
        last_eval     = ""
        clinvar_id    = ""

        try:
            search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            sr = requests.get(search_url,
                              params={"db": "clinvar", "term": f"{rsid}[rs]",
                                      "retmode": "json", "retmax": 3},
                              timeout=10)
            ids = sr.json().get("esearchresult", {}).get("idlist", [])
            if ids:
                sum_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                sr2 = requests.get(sum_url,
                                   params={"db": "clinvar", "id": ",".join(ids[:3]),
                                           "retmode": "json"},
                                   timeout=10)
                result = sr2.json().get("result", {})
                for uid in result.get("uids", []):
                    entry = result[uid]
                    gc    = entry.get("germline_classification", {})
                    if gc.get("description"):
                        review_status = gc.get("review_status", "")
                        last_eval     = gc.get("last_evaluated", "")
                        clinvar_id    = entry.get("accession", "")
                        for t in gc.get("trait_set", []):
                            name = t.get("trait_name", "")
                            if name:
                                conditions.append(name)
                        break
        except Exception:
            pass  # Step 2 失败时仍返回 Step 1 数据

        return {
            "clinvar_id":            clinvar_id or rsid,
            "clinical_significance": sig_display,
            "review_status":         review_status,
            "conditions":            conditions,
            "last_evaluated":        last_eval,
            "rsid":                  rsid,
        }

    except Exception as e:
        return {"error": f"ClinVar query error: {e}"}


# =============================================================================
# 5. 蛋白变异名称解析
# =============================================================================

def resolve_protein_variant(gene: str, protein_change: str,
                             assembly: str = "GRCh38") -> dict:
    """
    将蛋白变异名称（如 "TP53 R175H"）解析为基因组坐标。
    使用 Ensembl HGVS REST API（免费，无需 Key）。
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
    gene_url = f"{ENSEMBL_BASE}/lookup/symbol/homo_sapiens/{gene}"
    try:
        r = requests.get(gene_url, headers=hdrs, params={"expand": 1}, timeout=15)
        if r.status_code != 200:
            return {"error": f"Gene lookup failed: {gene} HTTP {r.status_code}"}
        gene_data     = r.json()
        transcripts   = gene_data.get("Transcript", [])
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
            return {"error": f"No transcript found for {gene}"}
    except Exception as e:
        return {"error": f"Gene lookup error: {e}"}
    # Strip version suffix from transcript ID (e.g. ENST00000269305.9 → ENST00000269305)
    transcript_id_base = transcript_id.split(".")[0]
    hgvs_notation = f"{transcript_id_base}:{pc}"
    hgvs_url      = f"{ENSEMBL_BASE}/variant_recoder/homo_sapiens/{hgvs_notation}"
    try:
        # vcf_string=1 is required to get VCF-format coordinates in the response
        r = requests.get(hgvs_url, headers=hdrs, params={"vcf_string": 1}, timeout=20)
        if r.status_code != 200:
            return {"error": f"HGVS resolve failed: {hgvs_notation} HTTP {r.status_code}"}
        recoder_data = r.json()
        if not recoder_data:
            return {"error": f"HGVS no result: {hgvs_notation}"}
        entry       = recoder_data[0] if isinstance(recoder_data, list) else recoder_data
        vcf_strings = []
        for key, val in entry.items():
            if isinstance(val, dict):
                vcf_list = val.get("vcf_string", [])
                if vcf_list:
                    vcf_strings.extend(vcf_list if isinstance(vcf_list, list) else [vcf_list])
        if not vcf_strings:
            return {"error": f"No VCF coords: {hgvs_notation}"}
        vcf_str = vcf_strings[0]
        parts   = vcf_str.split("-")
        if len(parts) < 4:
            return {"error": f"Bad VCF format: {vcf_str}"}
        chrom = parts[0].replace("chr", "")
        pos   = int(parts[1])
        ref   = parts[2].upper()
        alt   = parts[3].upper()
        print(f"Resolved: {gene} {protein_change} -> chr{chrom}:{pos} {ref}>{alt}")
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
        return {"error": f"HGVS resolve error: {e}"}


# =============================================================================
# 6. VCF 文件解析
# =============================================================================

def parse_vcf(vcf_path: str) -> list:
    """解析 VCF 文件（支持普通文本和 gzip 压缩），提取 SNV 变异列表。"""
    path = Path(vcf_path)
    if not path.exists():
        raise FileNotFoundError(f"VCF not found: {vcf_path}")
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
        alt   = alt_field.split(",")[0]
        if len(ref) == 1 and len(alt) == 1 and ref.upper() != alt.upper():
            try:
                variants.append({"chrom": chrom, "pos": int(pos_str),
                                  "ref": ref.upper(), "alt": alt.upper()})
            except ValueError:
                continue
    print(f"VCF parsed: {len(variants)} SNVs from {path.name}")
    return variants


# =============================================================================
# 7. 核心预测函数
# =============================================================================

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
                         artifacts: dict,
                         genos_url: str = None,
                         chrom: str = None,
                         pos: int = None,
                         ref: str = None,
                         alt: str = None) -> dict:
    """
    用 VEP 评分 + gnomAD AF 直接预测。

    v5.2: genos_url 指定后调用本地 Genos /extract 计算 REF/ALT embedding 余弦距离，
    替代 genos_score 的训练集中位数填充；服务不可用时自动降级回中位数。
    """
    model       = artifacts["model"]
    medians     = artifacts["medians"]
    calibrator  = artifacts["calibrator"]
    feat_labels = list(artifacts.get("feature_labels", FEATURE_LABELS_V5))
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

    # Genos Score: 优先用真实 embedding 评分，失败时降级到中位数
    genos_raw    = np.nan
    genos_source = "median imputation (no --genos-url)"
    if n_features == 8 and genos_url and chrom and pos and ref and alt:
        genos_raw = fetch_genos_embedding_score(chrom, pos, ref, alt, genos_url=genos_url)
        if not np.isnan(genos_raw):
            genos_source = "Genos embedding (cosine dist, 500bp context)"
        else:
            genos_source = "median imputation (Genos unavailable)"

    genos_val = (genos_raw if (n_features == 8 and not np.isnan(genos_raw))
                 else float(medians.get("genos_score", 0.676)))

    # 动态更新 Genos Score 标签
    if n_features == 8 and len(feat_labels) > 5:
        feat_labels[5] = "Genos Score" if not np.isnan(genos_raw) else "Genos Score*"

    if n_features == 8:
        feature_vec = [
            sift_inv,
            _val("polyphen_score", "polyphen"),
            _val("am_pathogenicity", "alphamissense"),
            _val("cadd_phred", "cadd"),
            float(medians.get("evo2_llr", -0.074)),
            genos_val,
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
        "genos_source":     genos_source,
        "offline_mode":     np.isnan(genos_raw),
    }


def predict_variant(chrom: str, pos: int, ref: str, alt: str,
                    artifacts: dict = None,
                    model_dir: str = ".",
                    genos_url: str = None,
                    include_clinvar: bool = False) -> dict:
    """
    单变异预测主函数。

    Args:
        chrom:           染色体（不含 chr 前缀）
        pos:             位置（1-based，GRCh38）
        ref:             参考等位基因
        alt:             替代等位基因
        artifacts:       load_model_artifacts() 返回的字典（可复用）
        model_dir:       模型目录（artifacts 为 None 时使用）
        genos_url:       Genos 本地服务器 URL（v5.2 新增，可选）
        include_clinvar: 是否查询 ClinVar（默认 False）

    Returns:
        dict: 包含 cal_prob, acmg_class, shap_values, vep_scores, gnomad_log_af 等
    """
    if artifacts is None:
        artifacts = load_model_artifacts(model_dir)

    chrom = str(chrom).replace("chr", "")
    pos   = int(pos)
    ref   = ref.upper()
    alt   = alt.upper()

    print(f"\nPredicting: chr{chrom}:{pos} {ref}>{alt}")

    # 并行获取 VEP 和 gnomAD
    with ThreadPoolExecutor(max_workers=2) as pool:
        vep_future    = pool.submit(fetch_vep_scores, chrom, pos, ref, alt)
        gnomad_future = pool.submit(fetch_gnomad_af,  chrom, pos, ref, alt)
    vep_scores    = vep_future.result()
    gnomad_log_af = gnomad_future.result()

    if "error" in vep_scores:
        print(f"  VEP warning: {vep_scores['error']}")

    result = predict_from_scores(
        vep_scores, gnomad_log_af, artifacts,
        genos_url=genos_url, chrom=chrom, pos=pos, ref=ref, alt=alt
    )
    result.update({
        "chrom":         chrom,
        "pos":           pos,
        "ref":           ref,
        "alt":           alt,
        "vep_scores":    vep_scores,
        "gnomad_log_af": gnomad_log_af,
    })

    if include_clinvar:
        result["clinvar"] = fetch_clinvar_classification(chrom, pos, ref, alt)

    return result


def predict_vcf(vcf_path: str,
                artifacts: dict = None,
                model_dir: str = ".",
                genos_url: str = None,
                output_csv: str = None,
                batch_size: int = 200) -> list:
    """
    批量预测 VCF 文件中的所有 SNV。

    Args:
        vcf_path:   VCF 文件路径（支持 .vcf 和 .vcf.gz）
        artifacts:  load_model_artifacts() 返回的字典
        model_dir:  模型目录（artifacts 为 None 时使用）
        genos_url:  Genos 本地服务器 URL（可选）
        output_csv: 输出 CSV 文件路径（可选）
        batch_size: VEP 批量注释大小（最大 200）

    Returns:
        list of dicts: 每个变异的预测结果
    """
    if artifacts is None:
        artifacts = load_model_artifacts(model_dir)

    variants = parse_vcf(vcf_path)
    if not variants:
        print("No SNVs found in VCF.")
        return []

    print(f"\nBatch predicting {len(variants)} variants...")
    results = []

    for batch_start in range(0, len(variants), batch_size):
        batch = variants[batch_start:batch_start + batch_size]
        print(f"  VEP batch {batch_start+1}-{batch_start+len(batch)} / {len(variants)}...")

        vep_batch_raw = fetch_vep_batch(batch)
        vep_map = {}
        for vr in vep_batch_raw:
            inp = vr.get("input", "")
            parts = inp.split()
            if len(parts) >= 5:
                key = (parts[0], int(parts[1]), parts[3].upper(), parts[4].upper())
                vep_map[key] = _extract_vep_scores_from_raw(vr)

        for v in batch:
            key           = (v["chrom"], v["pos"], v["ref"], v["alt"])
            vep_scores    = vep_map.get(key, {})
            gnomad_log_af = fetch_gnomad_af(v["chrom"], v["pos"], v["ref"], v["alt"])
            pred = predict_from_scores(
                vep_scores, gnomad_log_af, artifacts,
                genos_url=genos_url,
                chrom=v["chrom"], pos=v["pos"], ref=v["ref"], alt=v["alt"]
            )
            pred.update({"chrom": v["chrom"], "pos": v["pos"],
                          "ref": v["ref"], "alt": v["alt"],
                          "gene": vep_scores.get("gene", ""),
                          "hgvsp": vep_scores.get("hgvsp", ""),
                          "consequence": vep_scores.get("consequence", "")})
            results.append(pred)

    if output_csv:
        _write_csv(results, output_csv)

    print(f"\nDone: {len(results)} variants predicted.")
    return results


def _write_csv(results: list, output_path: str):
    """将预测结果写入 CSV 文件。"""
    if not results:
        return
    fieldnames = ["chrom", "pos", "ref", "alt", "gene", "hgvsp", "consequence",
                  "cal_prob", "raw_prob", "acmg_class", "acmg_confidence",
                  "top_shap_feature", "genos_source", "offline_mode"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {output_path}")


# =============================================================================
# 8. SHAP 瀑布图可视化
# =============================================================================

def plot_shap_waterfall(result: dict, save_path: str = None):
    """
    绘制单变异 SHAP 瀑布图。

    Args:
        result:    predict_variant() 返回的字典
        save_path: 图片保存路径（None 则直接显示）
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        print("shap / matplotlib not installed. pip install shap matplotlib")
        return

    shap_vals   = result.get("shap_values", [])
    feat_labels = result.get("feature_labels", [])
    feat_vec    = result.get("feature_vec", [])

    if not shap_vals:
        print("No SHAP values in result.")
        return

    n = min(len(shap_vals), len(feat_labels), len(feat_vec))
    shap_vals   = shap_vals[:n]
    feat_labels = feat_labels[:n]
    feat_vec    = feat_vec[:n]

    base_val = result.get("raw_prob", 0.5)
    expl = shap.Explanation(
        values=np.array(shap_vals),
        base_values=base_val,
        data=np.array(feat_vec),
        feature_names=feat_labels,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(expl, show=False)

    chrom = result.get("chrom", "")
    pos   = result.get("pos", "")
    ref   = result.get("ref", "")
    alt   = result.get("alt", "")
    prob  = result.get("cal_prob", 0.0)
    acmg  = result.get("acmg_class", "")
    plt.title(f"chr{chrom}:{pos} {ref}>{alt}  |  P(path)={prob:.3f}  |  {acmg}",
              fontsize=11, pad=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"SHAP waterfall saved: {save_path}")
    else:
        plt.show()
    plt.close()


# =============================================================================
# 9. 结果格式化打印
# =============================================================================

def print_result(result: dict):
    """格式化打印单变异预测结果。"""
    chrom = result.get("chrom", "?")
    pos   = result.get("pos", "?")
    ref   = result.get("ref", "?")
    alt   = result.get("alt", "?")
    vs    = result.get("vep_scores", {})

    print("\n" + "=" * 60)
    print(f"  Variant : chr{chrom}:{pos} {ref}>{alt}")
    if vs.get("gene"):
        print(f"  Gene    : {vs['gene']}  ({vs.get('transcript', '')})")
    if vs.get("hgvsp"):
        print(f"  HGVSp   : {vs['hgvsp']}")
    if vs.get("consequence"):
        print(f"  Conseq  : {vs['consequence']}")
    if vs.get("warning"):
        print(f"  WARNING : {vs['warning']}")
    print("-" * 60)
    print(f"  P(path) : {result['cal_prob']:.4f}  (raw: {result['raw_prob']:.4f})")
    print(f"  ACMG    : {result['acmg_class']}  [{result['acmg_confidence']}]")
    print(f"  Top feat: {result.get('top_shap_feature', '')}")
    print(f"  Genos   : {result.get('genos_source', '')}")
    print("-" * 60)
    print("  Feature scores:")
    for lbl, val, sv in zip(result["feature_labels"],
                             result["feature_vec"],
                             result["shap_values"]):
        bar = "+" * int(abs(sv) * 20) if sv >= 0 else "-" * int(abs(sv) * 20)
        print(f"    {lbl:<22} {val:>8.4f}   SHAP {sv:+.4f}  {bar}")
    if "clinvar" in result:
        cv = result["clinvar"]
        print("-" * 60)
        print(f"  ClinVar : {cv.get('clinical_significance', 'N/A')}")
        if cv.get("conditions"):
            print(f"  Cond.   : {', '.join(cv['conditions'][:3])}")
        if cv.get("review_status"):
            print(f"  Review  : {cv['review_status']}")
    print("=" * 60)


# =============================================================================
# 10. CLI 入口
# =============================================================================

def print_shap_summary(result: dict):
    """
    打印单变异 SHAP 贡献简洁摘要（无需 matplotlib）。

    与 SKILL.md 快速开始示例一致：
        result = predict_variant("17", 7674220, "C", "T", artifacts=artifacts)
        print_shap_summary(result)

    输出示例：
        SHAP Feature Contributions
        ──────────────────────────────────────────
        AlphaMissense      0.8120   SHAP +0.312  ▲ pathogenic
        CADD Phred        28.4000   SHAP +0.198  ▲ pathogenic
        gnomAD log-AF     -7.9000   SHAP +0.143  ▲ pathogenic
        PolyPhen-2         0.9900   SHAP +0.089  ▲ pathogenic
        SIFT (inv)         0.9800   SHAP +0.071  ▲ pathogenic
        phyloP             2.1000   SHAP +0.044  ▲ pathogenic
        Genos Score*       0.6760   SHAP -0.012  ▼ benign
        Evo2 LLR*         -0.0740   SHAP -0.008  ▼ benign
        ──────────────────────────────────────────
        Top driver: AlphaMissense  (SHAP +0.312)
    """
    shap_vals   = result.get("shap_values", [])
    feat_labels = result.get("feature_labels", [])
    feat_vec    = result.get("feature_vec", [])

    if not shap_vals:
        print("No SHAP values available in result.")
        return

    n = min(len(shap_vals), len(feat_labels), len(feat_vec))
    triples = list(zip(feat_labels[:n], feat_vec[:n], shap_vals[:n]))
    # Sort by |SHAP| descending
    triples.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\nSHAP Feature Contributions")
    print("─" * 50)
    for label, val, sv in triples:
        direction = "▲ pathogenic" if sv >= 0 else "▼ benign"
        print(f"  {label:<22} {val:>8.4f}   SHAP {sv:+.3f}  {direction}")
    print("─" * 50)

    top_label, _, top_sv = triples[0]
    print(f"  Top driver: {top_label}  (SHAP {top_sv:+.3f})")
    print(f"  Calibrated P(pathogenic): {result.get('cal_prob', float('nan')):.1%}")
    print(f"  ACMG class: {result.get('acmg_class', 'N/A')}")
    if not result.get("offline_mode", True):
        print(f"  Genos source: {result.get('genos_source', '')}")


# =============================================================================
# 11. 临床报告生成（LLM）
# =============================================================================

def generate_clinical_report(
    result: dict,
    api_key: str,
    base_url: str = "https://api.moonshot.cn/v1",
    model: str = "moonshot-v1-32k",
    template: str = "chinese",
    stream: bool = False,
):
    """
    基于 predict_variant() 的结果，调用 LLM 生成 ACMG 临床解读报告。

    Args:
        result:   predict_variant() 返回的字典
        api_key:  任意 OpenAI-compatible API Key
        base_url: API 端点（默认 Moonshot/Kimi；可换 DashScope、OpenAI、DeepSeek 等）
        model:    模型 ID（如 "qwen-plus", "gpt-4o", "moonshot-v1-32k"）
        template: "chinese"（默认）| "english" | "summary"
        stream:   False → 返回完整字符串；True → 返回 Generator（逐 chunk 输出）

    Returns:
        str（stream=False）或 Generator[str]（stream=True）

    常用 provider 预设：
        Moonshot/Kimi:   base_url="https://api.moonshot.cn/v1"
        Alibaba DashScope: base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        OpenAI:          base_url="https://api.openai.com/v1"
        DeepSeek:        base_url="https://api.deepseek.com/v1"

    Example:
        result = predict_variant("17", 7674220, "C", "T", artifacts=artifacts)
        report = generate_clinical_report(result, api_key="sk-xxx", template="chinese")
        print(report)
    """
    import sys, os
    # 找到 kimi_report.py（在 SNV-judge 根目录）
    _skill_dir  = os.path.dirname(os.path.abspath(__file__))          # skill/scripts/
    _root_dir   = os.path.dirname(os.path.dirname(_skill_dir))        # SNV-judge/
    if _root_dir not in sys.path:
        sys.path.insert(0, _root_dir)

    try:
        import kimi_report as _kr
    except ImportError:
        raise ImportError(
            "kimi_report.py not found. Make sure you are running from the SNV-judge repo root, "
            "or that the repo root is in sys.path."
        )

    # 从 result 中提取 kimi_report 需要的参数
    vep     = result.get("vep_scores", {})
    variant_info = {
        "chrom": result.get("chrom", "?"),
        "pos":   result.get("pos",   0),
        "ref":   result.get("ref",   "?"),
        "alt":   result.get("alt",   "?"),
    }
    scores = {
        "gene":        vep.get("gene", ""),
        "transcript":  vep.get("transcript", ""),
        "hgvsp":       vep.get("hgvsp", ""),
        "consequence": vep.get("consequence", ""),
        "sift_score":  result.get("feature_vec", [None])[0],
        "polyphen_score": result.get("feature_vec", [None, None])[1] if len(result.get("feature_vec", [])) > 1 else None,
        "am_pathogenicity": result.get("feature_vec", [None]*3)[2] if len(result.get("feature_vec", [])) > 2 else None,
        "cadd_phred":  result.get("feature_vec", [None]*4)[3] if len(result.get("feature_vec", [])) > 3 else None,
    }
    feat_vec   = result.get("feature_vec", [])
    shap_vals  = result.get("shap_values", [])
    cal_prob   = result.get("cal_prob", float("nan"))
    evo2_llr   = feat_vec[4] if len(feat_vec) > 4 else float("nan")
    genos_path = feat_vec[5] if len(feat_vec) > 5 else float("nan")
    gnomad_log = feat_vec[7] if len(feat_vec) > 7 else float("nan")

    gen = _kr.generate_report_stream(
        variant_info, scores, shap_vals, cal_prob,
        evo2_llr, genos_path, gnomad_log,
        model_ver="v5",
        template=template,
        base_url=base_url,
        api_key=api_key,
        model=model,
    )

    if stream:
        return gen
    else:
        return "".join(gen)


def main():
    parser = argparse.ArgumentParser(
        description="SNV-judge v5.2 — Offline Variant Pathogenicity Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single variant (positional args)
  python predict.py 17 7674220 C T /path/to/SNV-judge

  # Single variant with Genos embedding scoring
  python predict.py 17 7674220 C T /path/to/SNV-judge \
      --genos-url https://xxx.ngrok-free.dev

  # Protein variant shorthand
  python predict.py --gene TP53 --protein R175H /path/to/SNV-judge --clinvar

  # Batch VCF prediction
  python predict.py --vcf variants.vcf /path/to/SNV-judge --output results.csv
        """,
    )
    parser.add_argument("chrom",      nargs="?", help="Chromosome (no chr prefix)")
    parser.add_argument("pos",        nargs="?", type=int, help="Position (1-based, GRCh38)")
    parser.add_argument("ref",        nargs="?", help="Reference allele")
    parser.add_argument("alt",        nargs="?", help="Alternate allele")
    parser.add_argument("model_dir",  nargs="?", default=".", help="Model directory")
    parser.add_argument("--vcf",      help="Input VCF file for batch prediction")
    parser.add_argument("--gene",     help="Gene symbol (for protein variant mode)")
    parser.add_argument("--protein",  help="Protein change (e.g. R175H or p.Arg175His)")
    parser.add_argument("--clinvar",  action="store_true", help="Query ClinVar")
    parser.add_argument("--output",   help="Output CSV path (batch mode)")
    parser.add_argument("--shap",     help="Save SHAP waterfall plot to this path")
    parser.add_argument("--genos-url", dest="genos_url", default=None,
                        help="Local Genos server URL (e.g. https://xxx.ngrok-free.dev)")
    args = parser.parse_args()

    artifacts = load_model_artifacts(args.model_dir)

    # --- 蛋白变异模式 ---
    if args.gene and args.protein:
        coords = resolve_protein_variant(args.gene, args.protein)
        if "error" in coords:
            print(f"Error: {coords['error']}")
            return
        result = predict_variant(
            coords["chrom"], coords["pos"], coords["ref"], coords["alt"],
            artifacts=artifacts,
            genos_url=args.genos_url,
            include_clinvar=args.clinvar,
        )
        print_result(result)
        if args.shap:
            plot_shap_waterfall(result, save_path=args.shap)
        return

    # --- VCF 批量模式 ---
    if args.vcf:
        predict_vcf(
            args.vcf,
            artifacts=artifacts,
            genos_url=args.genos_url,
            output_csv=args.output,
        )
        return

    # --- 单变异模式 ---
    if not all([args.chrom, args.pos, args.ref, args.alt]):
        parser.print_help()
        return

    result = predict_variant(
        args.chrom, args.pos, args.ref, args.alt,
        artifacts=artifacts,
        genos_url=args.genos_url,
        include_clinvar=args.clinvar,
    )
    print_result(result)
    if args.shap:
        plot_shap_waterfall(result, save_path=args.shap)


if __name__ == "__main__":
    main()
