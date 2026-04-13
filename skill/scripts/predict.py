"""
predict.py — SNV-judge v5 完整预测脚本
=======================================
直接调用训练好的模型文件进行预测。

工作流程：
  1. 加载本地 pkl 模型文件（xgb_model_v5.pkl 等）
  2. 通过 Ensembl VEP REST API 获取 SIFT/PolyPhen/AlphaMissense/CADD/phyloP（免费）
  3. 通过 gnomAD GraphQL API 获取人群频率（免费）
  4. 通过 NVIDIA Evo2-40B NIM API 计算 Log-Likelihood Ratio（需要 NVIDIA API Key）
  5. 通过 Genos-10B Stomics API 获取致病性评分（需要 Stomics API Key）
     → 任一 API Key 未提供时，对应特征自动降级为训练集中位数填充
  6. 运行 XGBoost+LightGBM Stacking 集成模型 + Isotonic 校准
  7. 输出校准概率、ACMG 分级、SHAP 解释

Usage:
    from scripts.predict import predict_variant, load_model_artifacts

    artifacts = load_model_artifacts(model_dir="/path/to/SNV-judge")

    # 完整模式（Evo2 + Genos 均提供）
    result = predict_variant("17", 7674220, "C", "T",
                             artifacts=artifacts,
                             evo2_api_key="nvapi-...",
                             genos_api_key="sk-...")

    # 仅 Evo2 模式
    result = predict_variant("17", 7674220, "C", "T",
                             artifacts=artifacts,
                             evo2_api_key="nvapi-...")

    # 离线模式（无需任何 API Key，AUROC 仅下降 0.0093）
    result = predict_variant("17", 7674220, "C", "T", artifacts=artifacts)

    print(result['cal_prob'], result['acmg_class'])

CLI:
    python predict.py 17 7674220 C T /path/to/SNV-judge [nvapi-key] [sk-genos-key]
"""

import os
import pickle
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

warnings.filterwarnings("ignore")

# ── API 端点 ──────────────────────────────────────────────────────────────
VEP_URL    = "https://rest.ensembl.org/vep/homo_sapiens/region"
VEP_HDR    = {"Content-Type": "application/json", "Accept": "application/json"}
GNOMAD_URL = "https://gnomad.broadinstitute.org/api"
# Source: https://docs.api.nvidia.com/nim/reference/arc-evo2-40b-infer
EVO2_URL   = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate"
# Source: https://cloud.stomics.tech (Zhejiang Lab Genos API)
GENOS_URL  = "https://cloud.stomics.tech/api/aigateway/genos/variant_predict"

# Evo2 DNA base → logit index（ASCII 值，来自官方文档）
EVO2_BASE_IDX = {'A': 65, 'C': 67, 'T': 84, 'G': 71}

# ── ACMG 5 级分类阈值 ─────────────────────────────────────────────────────
ACMG_TIERS = [
    (0.90, "Pathogenic (P)",         "High confidence"),
    (0.70, "Likely Pathogenic (LP)", "Moderate confidence"),
    (0.40, "VUS",                    "Uncertain significance"),
    (0.20, "Likely Benign (LB)",     "Moderate confidence"),
    (0.00, "Benign (B)",             "High confidence"),
]

# ── 特征名称（与模型训练时一致）──────────────────────────────────────────
FEATURE_NAMES  = ["sift_inv", "polyphen", "alphamissense", "cadd",
                  "evo2_llr", "genos_score", "phylop", "gnomad_log_af"]
FEATURE_LABELS = ["SIFT (inv)", "PolyPhen-2", "AlphaMissense", "CADD Phred",
                  "Evo2-40B LLR", "Genos-10B Score", "phyloP", "gnomAD log-AF"]


# ═══════════════════════════════════════════════════════════════════════════
# 1. 模型加载
# ═══════════════════════════════════════════════════════════════════════════

def load_model_artifacts(model_dir: str = ".") -> dict:
    """
    加载 v5 模型文件，自动降级到 v4/v3/v2/v1。

    Args:
        model_dir: SNV-judge 项目根目录（含 xgb_model_v5.pkl 等文件）

    Returns:
        dict 包含 model, medians, calibrator, version
    """
    base = Path(model_dir)
    for suffix, ver in [("_v5","v5"), ("_v4","v4"), ("_v3","v3"), ("_v2","v2"), ("","v1")]:
        xgb_path = base / f"xgb_model{suffix}.pkl"
        med_path  = base / f"train_medians{suffix}.pkl"
        cal_path  = base / f"platt_scaler{suffix}.pkl"
        if not (xgb_path.exists() and med_path.exists() and cal_path.exists()):
            continue
        with open(xgb_path, "rb") as f: model      = pickle.load(f)
        with open(med_path,  "rb") as f: medians    = pickle.load(f)
        with open(cal_path,  "rb") as f: calibrator = pickle.load(f)
        print(f"✓ 模型加载成功: SNV-judge {ver} (XGB + LGB + LR meta + Isotonic 校准)")
        return {"model": model, "medians": medians,
                "calibrator": calibrator, "version": ver}
    raise FileNotFoundError(
        f"未找到模型文件。请确认 model_dir='{model_dir}' 下存在 xgb_model_v*.pkl 等文件。"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Evo2-40B LLR 计算（NVIDIA NIM API）
# ═══════════════════════════════════════════════════════════════════════════

def compute_evo2_llr(context_prefix: str, ref: str, alt: str,
                     api_key: str) -> float:
    """
    调用 Evo2-40B NIM API 计算 Log-Likelihood Ratio。
    LLR = log P(alt|context) - log P(ref|context)
    负值 → alt 比 ref 更不自然（致病信号）

    Args:
        context_prefix: 变异位点上游的 DNA 序列（建议 ≥ 50bp，GRCh38）
        ref:            参考等位基因（单碱基）
        alt:            替代等位基因（单碱基）
        api_key:        NVIDIA NIM API Key（nvapi-...）
    """
    for attempt in range(3):
        try:
            r = requests.post(
                EVO2_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "sequence":      context_prefix,
                    "num_tokens":    1,
                    "enable_logits": True,
                    "top_k":         1,
                    "top_p":         0.0,
                },
                timeout=120,
            )
            if r.status_code == 200:
                logits  = np.array(r.json()["logits"][0])  # (512,)
                shifted = logits - logits.max()
                log_sum = np.log(np.sum(np.exp(shifted)))
                lp_ref  = float(shifted[EVO2_BASE_IDX[ref.upper()]] - log_sum)
                lp_alt  = float(shifted[EVO2_BASE_IDX[alt.upper()]] - log_sum)
                return lp_alt - lp_ref
            elif r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 10)))
            elif r.status_code == 401:
                raise ValueError("Evo2 API Key 无效，请检查 nvapi-... 格式")
            else:
                raise RuntimeError(f"Evo2 API 错误 {r.status_code}: {r.text[:200]}")
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            if attempt < 2: time.sleep(3 * (attempt + 1))
            else: raise RuntimeError(f"Evo2 API 请求失败（3次重试）: {e}")
    raise RuntimeError("Evo2 API 不可用")


def fetch_genomic_context(chrom: str, pos: int, window: int = 60) -> str:
    """从 Ensembl REST API 获取变异位点上游参考序列（GRCh38），用作 Evo2 上下文。"""
    start = max(1, pos - window)
    end   = pos - 1
    url   = (f"https://rest.ensembl.org/sequence/region/human/"
             f"{chrom}:{start}..{end}?content-type=text/plain")
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            seq = r.text.strip().upper()
            return "".join(b if b in "ACGTN" else "N" for b in seq)
    except Exception:
        pass
    return "N" * window


# ═══════════════════════════════════════════════════════════════════════════
# 3. Genos-10B 致病性评分（Stomics Cloud API）
# ═══════════════════════════════════════════════════════════════════════════

def compute_genos_score(chrom: str, pos: int, ref: str, alt: str,
                        api_key: str,
                        api_url: str = GENOS_URL) -> float:
    """
    调用 Genos-10B Stomics API 获取致病性评分。
    返回 score_Pathogenic（0~1，越高越致病）。

    Args:
        chrom:   染色体（不含 chr 前缀）
        pos:     位置（1-based，GRCh38）
        ref:     参考等位基因
        alt:     替代等位基因
        api_key: Stomics API Key（sk-...）
        api_url: Genos API 端点（默认 cloud.stomics.tech，可自定义）
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "assembly": "hg38",
        "chrom":    f"chr{chrom}",
        "pos":      int(pos),
        "ref":      ref.upper(),
        "alt":      alt.upper(),
    }
    for attempt in range(3):
        try:
            r = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if r.status_code == 200:
                result = r.json().get("result", {})
                score  = result.get("score_Pathogenic")
                if score is not None:
                    return float(score)
                raise RuntimeError(f"Genos 返回结果中无 score_Pathogenic 字段: {r.text[:200]}")
            elif r.status_code == 429:
                time.sleep(5 * (attempt + 1))
            elif r.status_code == 401:
                raise ValueError("Genos API Key 无效，请检查 sk-... 格式")
            elif r.status_code == 404:
                raise ValueError(f"Genos API 端点不存在，请确认 URL: {api_url}")
            else:
                raise RuntimeError(f"Genos API 错误 {r.status_code}: {r.text[:200]}")
        except (ValueError, RuntimeError):
            raise
        except Exception as e:
            if attempt < 2: time.sleep(2 * (attempt + 1))
            else: raise RuntimeError(f"Genos API 请求失败（3次重试）: {e}")
    raise RuntimeError("Genos API 不可用")


# ═══════════════════════════════════════════════════════════════════════════
# 4. VEP + gnomAD（免费 API）
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
                time.sleep(int(r.headers.get("Retry-After", 5)))
            else:
                return {"error": f"VEP HTTP {r.status_code}"}
        except Exception as e:
            if attempt < 2: time.sleep(3)
            else: return {"error": f"VEP 请求失败: {e}"}
    else:
        return {"error": "VEP API 不可用"}

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
        result["warning"] = f"变异类型为 {', '.join(csq)}，非 missense_variant，评分可能不可靠"
    return result


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
            r = requests.post(GNOMAD_URL,
                              json={"query": query, "variables": {"vid": vid}},
                              headers={"Content-Type": "application/json"},
                              timeout=15)
            if r.status_code == 200:
                d = r.json().get("data", {}).get("variant")
                if d:
                    vals = [float(d[s]["af"]) for s in ("exome","genome")
                            if d.get(s) and d[s].get("af") is not None]
                    af = max(vals) if vals else 0.0
                else:
                    af = 0.0
                return float(np.log10(af + 1e-8))
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return np.nan


# ═══════════════════════════════════════════════════════════════════════════
# 5. 核心预测
# ═══════════════════════════════════════════════════════════════════════════

def _get_acmg_tier(prob: float) -> dict:
    for thresh, label, confidence in ACMG_TIERS:
        if prob >= thresh:
            return {"label": label, "confidence": confidence}
    return {"label": "Benign (B)", "confidence": "High confidence"}


def predict_from_scores(vep_scores: dict, gnomad_log_af: float,
                         evo2_llr: float, genos_score: float,
                         artifacts: dict) -> dict:
    """
    组装特征向量并运行 stacking 模型预测。

    Args:
        vep_scores:    fetch_vep_scores() 返回值
        gnomad_log_af: fetch_gnomad_af() 返回值
        evo2_llr:      compute_evo2_llr() 返回值，或 np.nan（触发中位数填充）
        genos_score:   compute_genos_score() 返回值，或 np.nan（触发中位数填充）
        artifacts:     load_model_artifacts() 返回值
    """
    model      = artifacts["model"]
    medians    = artifacts["medians"]
    calibrator = artifacts["calibrator"]

    def _val(key, med_key=None):
        v = vep_scores.get(key)
        if v is not None:
            try: return float(v)
            except: pass
        return float(medians.get(med_key or key, 0.0))

    def _impute(val, med_key):
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return float(val)
        return float(medians.get(med_key, 0.0))

    sift_score   = vep_scores.get("sift_score")
    sift_inv     = (1.0 - float(sift_score)) if sift_score is not None \
                   else float(medians.get("sift_inv", 1.0))

    evo2_used    = _impute(evo2_llr,    "evo2_llr")
    genos_used   = _impute(genos_score, "genos_score")
    evo2_real    = not (evo2_llr   is None or (isinstance(evo2_llr,   float) and np.isnan(evo2_llr)))
    genos_real   = not (genos_score is None or (isinstance(genos_score, float) and np.isnan(genos_score)))

    feature_vec = [
        sift_inv,                                   # SIFT (inv)
        _val("polyphen_score", "polyphen"),          # PolyPhen-2
        _val("am_pathogenicity", "alphamissense"),   # AlphaMissense
        _val("cadd_phred", "cadd"),                  # CADD Phred
        evo2_used,                                   # Evo2-40B LLR
        genos_used,                                  # Genos-10B Score
        _val("phylop", "phylop"),                    # phyloP
        _impute(gnomad_log_af, "gnomad_log_af"),     # gnomAD log-AF
    ]

    X = np.array([feature_vec])

    # Stacking 预测
    if isinstance(model, dict):
        xgb_p    = model["xgb"].predict_proba(X)[0, 1]
        lgb_p    = model["lgb"].predict_proba(X)[0, 1]
        raw_prob = float(model["meta"].predict_proba(
                         np.array([[xgb_p, lgb_p]]))[0, 1])
    else:
        raw_prob = float(model.predict_proba(X)[0, 1])

    # Isotonic Regression 校准
    from sklearn.isotonic import IsotonicRegression
    if isinstance(calibrator, IsotonicRegression):
        cal_prob = float(np.clip(calibrator.predict([raw_prob])[0], 0.0, 1.0))
    else:
        logit    = np.log(raw_prob / (1 - raw_prob + 1e-9))
        cal_prob = float(calibrator.predict_proba([[logit]])[0, 1])

    acmg = _get_acmg_tier(cal_prob)

    # SHAP 解释
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
        "feature_labels":   FEATURE_LABELS,
        "feature_vec":      feature_vec,
        "top_shap_feature": FEATURE_LABELS[top_idx],
        "evo2_llr":         evo2_used,
        "genos_score":      genos_used,
        "evo2_real":        evo2_real,
        "genos_real":       genos_real,
        "offline_mode":     not (evo2_real and genos_real),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. 高层接口（推荐使用）
# ═══════════════════════════════════════════════════════════════════════════

def predict_variant(chrom: str, pos: int, ref: str, alt: str,
                     artifacts: dict = None,
                     model_dir: str = ".",
                     evo2_api_key: str = None,
                     genos_api_key: str = None,
                     genos_api_url: str = GENOS_URL) -> dict:
    """
    完整预测流程：VEP + gnomAD + Evo2（可选）+ Genos（可选）→ 模型预测 → ACMG 分级 + SHAP。

    Args:
        chrom:         染色体（如 "17"，不含 "chr" 前缀，GRCh38）
        pos:           位置（1-based）
        ref:           参考等位基因（单碱基）
        alt:           替代等位基因（单碱基）
        artifacts:     load_model_artifacts() 的返回值（可复用）
        model_dir:     模型文件目录（artifacts 为 None 时使用）
        evo2_api_key:  NVIDIA NIM API Key（nvapi-...）
                       不提供时 Evo2 LLR 用训练集中位数填充
        genos_api_key: Stomics API Key（sk-...）
                       不提供时 Genos Score 用训练集中位数填充
        genos_api_url: Genos API 端点（默认 cloud.stomics.tech，可自定义私有部署地址）

    Returns:
        dict 包含 cal_prob, acmg_class, shap_values, evo2_real, genos_real 等
    """
    chrom = str(chrom).replace("chr", "")
    ref, alt = ref.upper().strip(), alt.upper().strip()

    if artifacts is None:
        artifacts = load_model_artifacts(model_dir)

    # 模式说明
    mode_parts = []
    if evo2_api_key:  mode_parts.append("Evo2 API")
    if genos_api_key: mode_parts.append("Genos API")
    mode_str = " + ".join(mode_parts) if mode_parts else "离线模式（中位数填充）"
    print(f"预测 chr{chrom}:{pos} {ref}>{alt}  [{mode_str}]")

    # 并行获取所有特征
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures["vep"]    = ex.submit(fetch_vep_scores, chrom, pos, ref, alt)
        futures["gnomad"] = ex.submit(fetch_gnomad_af,  chrom, pos, ref, alt)
        if evo2_api_key:
            futures["ctx"] = ex.submit(fetch_genomic_context, chrom, pos)

        vep_scores    = futures["vep"].result()
        gnomad_log_af = futures["gnomad"].result()
        context_seq   = futures["ctx"].result() if "ctx" in futures else None

    if "error" in vep_scores:
        raise RuntimeError(f"VEP 注释失败: {vep_scores['error']}")
    if vep_scores.get("warning"):
        print(f"  ⚠ {vep_scores['warning']}")

    # Evo2 LLR（并行 Genos 调用）
    evo2_llr    = np.nan
    genos_score = np.nan

    def _get_evo2():
        if evo2_api_key and context_seq:
            try:
                return compute_evo2_llr(context_seq, ref, alt, evo2_api_key)
            except Exception as e:
                print(f"  ⚠ Evo2 API 失败，降级为中位数填充: {e}")
        return np.nan

    def _get_genos():
        if genos_api_key:
            try:
                return compute_genos_score(chrom, pos, ref, alt,
                                           genos_api_key, genos_api_url)
            except Exception as e:
                print(f"  ⚠ Genos API 失败，降级为中位数填充: {e}")
        return np.nan

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_evo2  = ex.submit(_get_evo2)
        fut_genos = ex.submit(_get_genos)
        evo2_llr    = fut_evo2.result()
        genos_score = fut_genos.result()

    if not np.isnan(evo2_llr):
        print(f"  Evo2 LLR    = {evo2_llr:.4f}")
    if not np.isnan(genos_score):
        print(f"  Genos Score = {genos_score:.4f}")

    # 模型预测
    result = predict_from_scores(vep_scores, gnomad_log_af,
                                  evo2_llr, genos_score, artifacts)

    result.update({
        "chrom":       chrom,
        "pos":         pos,
        "ref":         ref,
        "alt":         alt,
        "gene":        vep_scores.get("gene", ""),
        "transcript":  vep_scores.get("transcript", ""),
        "hgvsp":       vep_scores.get("hgvsp", ""),
        "consequence": vep_scores.get("consequence", ""),
        "am_class":    vep_scores.get("am_class", ""),
        "sift_pred":   vep_scores.get("sift_pred", ""),
        "gnomad_log_af": gnomad_log_af,
    })

    # 打印摘要
    evo2_note  = f"Evo2 LLR = {result['evo2_llr']:.4f} (真实值)" \
                 if result["evo2_real"] else "Evo2 LLR = 中位数填充"
    genos_note = f"Genos Score = {result['genos_score']:.4f} (真实值)" \
                 if result["genos_real"] else "Genos Score = 中位数填充"

    print(f"\n{'='*54}")
    print(f"  变异: chr{chrom}:{pos} {ref}>{alt}")
    print(f"  基因: {result['gene']}  {result['hgvsp']}")
    print(f"  致病概率: {result['cal_prob']:.1%}  →  {result['acmg_class']}")
    print(f"  主要贡献: {result['top_shap_feature']}")
    print(f"  {evo2_note}")
    print(f"  {genos_note}")
    print(f"{'='*54}")

    return result


def print_shap_summary(result: dict):
    """打印 SHAP 特征贡献摘要。"""
    print("\nSHAP 特征贡献（正值→致病，负值→良性）:")
    pairs = sorted(zip(result["feature_labels"], result["shap_values"]),
                   key=lambda x: abs(x[1]), reverse=True)
    for label, sv in pairs:
        bar  = "█" * int(abs(sv) * 30)
        direction = "→致病" if sv > 0 else "→良性"
        is_evo2  = (label == "Evo2-40B LLR"    and not result.get("evo2_real"))
        is_genos = (label == "Genos-10B Score"  and not result.get("genos_real"))
        note = " [中位数填充]" if (is_evo2 or is_genos) else ""
        print(f"  {label:<20} {sv:+.4f} {direction}  {bar}{note}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("用法: python predict.py <chrom> <pos> <ref> <alt> [model_dir] [nvapi-key] [sk-genos-key]")
        print()
        print("示例（完整模式）:")
        print("  python predict.py 17 7674220 C T /path/to/SNV-judge nvapi-... sk-...")
        print()
        print("示例（仅 Evo2）:")
        print("  python predict.py 17 7674220 C T /path/to/SNV-judge nvapi-...")
        print()
        print("示例（离线模式）:")
        print("  python predict.py 17 7674220 C T /path/to/SNV-judge")
        sys.exit(0)

    chrom         = sys.argv[1]
    pos           = int(sys.argv[2])
    ref           = sys.argv[3]
    alt           = sys.argv[4]
    model_dir     = sys.argv[5] if len(sys.argv) > 5 else "."
    evo2_api_key  = sys.argv[6] if len(sys.argv) > 6 else None
    genos_api_key = sys.argv[7] if len(sys.argv) > 7 else None

    artifacts = load_model_artifacts(model_dir)
    result    = predict_variant(chrom, pos, ref, alt,
                                artifacts=artifacts,
                                evo2_api_key=evo2_api_key,
                                genos_api_key=genos_api_key)
    print_shap_summary(result)
