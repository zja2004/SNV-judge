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

Usage:
    from scripts.predict import predict_variant, load_model_artifacts

    # 加载模型（只需一次）
    artifacts = load_model_artifacts(model_dir="/path/to/SNV-judge")

    # 预测单个变异
    result = predict_variant("17", 7674220, "C", "T", artifacts=artifacts)
    print(result['cal_prob'], result['acmg_class'])

CLI:
    python predict.py 17 7674220 C T /path/to/SNV-judge
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

# ── 免费 API 端点（无需 Key）─────────────────────────────────────────────
VEP_URL    = "https://rest.ensembl.org/vep/homo_sapiens/region"
VEP_HDR    = {"Content-Type": "application/json", "Accept": "application/json"}
GNOMAD_URL = "https://gnomad.broadinstitute.org/api"

# ── ACMG 5 级分类阈值 ─────────────────────────────────────────────────────
ACMG_TIERS = [
    (0.90, "Pathogenic (P)",        "High confidence"),
    (0.70, "Likely Pathogenic (LP)","Moderate confidence"),
    (0.40, "VUS",                   "Uncertain significance"),
    (0.20, "Likely Benign (LB)",    "Moderate confidence"),
    (0.00, "Benign (B)",            "High confidence"),
]

# ── 特征名称（与模型训练时一致）──────────────────────────────────────────
FEATURE_NAMES  = ["sift_inv", "polyphen", "alphamissense", "cadd",
                  "evo2_llr", "genos_score", "phylop", "gnomad_log_af"]
FEATURE_LABELS = ["SIFT (inv)", "PolyPhen-2", "AlphaMissense", "CADD Phred",
                  "Evo2 LLR*", "Genos Score*", "phyloP", "gnomAD log-AF"]
# * 表示该特征在离线模式下使用训练集中位数填充


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
        print(f"  模型路径: {xgb_path}")
        return {"model": model, "medians": medians,
                "calibrator": calibrator, "version": ver}
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
        "gene":           chosen.get("gene_symbol", ""),
        "transcript":     chosen.get("transcript_id", ""),
        "hgvsp":          chosen.get("hgvsp", ""),
        "consequence":    ", ".join(csq),
        "sift_score":     chosen.get("sift_score"),
        "sift_pred":      chosen.get("sift_prediction"),
        "polyphen_score": chosen.get("polyphen_score"),
        "polyphen_pred":  chosen.get("polyphen_prediction"),
        "cadd_phred":     chosen.get("cadd_phred"),
        "am_pathogenicity": am.get("am_pathogenicity") if isinstance(am, dict) else None,
        "am_class":         am.get("am_class")         if isinstance(am, dict) else None,
        "phylop":           chosen.get("conservation"),
    }
    if "missense_variant" not in csq:
        result["warning"] = f"变异类型为 {', '.join(csq)}，非 missense_variant，评分可能不可靠"
    return result


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
                    af = 0.0   # 变异不在 gnomAD → PM2 信号
                return float(np.log10(af + 1e-8))
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return np.nan   # API 不可用时返回 nan，后续用中位数填充


# ═══════════════════════════════════════════════════════════════════════════
# 3. 核心预测函数
# ═══════════════════════════════════════════════════════════════════════════

def _get_acmg_tier(prob: float) -> dict:
    for thresh, label, confidence in ACMG_TIERS:
        if prob >= thresh:
            return {"label": label, "confidence": confidence, "prob": prob}
    return {"label": "Benign (B)", "confidence": "High confidence", "prob": prob}


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

    sift_score = vep_scores.get("sift_score")
    sift_inv   = (1.0 - float(sift_score)) if sift_score is not None else float(medians.get("sift_inv", 1.0))

    feature_vec = [
        sift_inv,                                          # SIFT (inv)
        _val("polyphen_score", "polyphen"),                # PolyPhen-2
        _val("am_pathogenicity", "alphamissense"),         # AlphaMissense
        _val("cadd_phred", "cadd"),                        # CADD Phred
        float(medians.get("evo2_llr", -0.074)),            # Evo2 LLR  ← 中位数填充
        float(medians.get("genos_score", 0.676)),          # Genos     ← 中位数填充
        _val("phylop", "phylop"),                          # phyloP
        _impute(gnomad_log_af, "gnomad_log_af"),           # gnomAD log-AF
    ]

    X = np.array([feature_vec])

    # Stacking 预测
    if isinstance(model, dict):
        xgb_p = model["xgb"].predict_proba(X)[0, 1]
        lgb_p = model["lgb"].predict_proba(X)[0, 1]
        raw_prob = float(model["meta"].predict_proba(np.array([[xgb_p, lgb_p]]))[0, 1])
    else:
        raw_prob = float(model.predict_proba(X)[0, 1])

    # Isotonic Regression 校准
    from sklearn.isotonic import IsotonicRegression
    if isinstance(calibrator, IsotonicRegression):
        cal_prob = float(np.clip(calibrator.predict([raw_prob])[0], 0.0, 1.0))
    else:
        logit    = np.log(raw_prob / (1 - raw_prob + 1e-9))
        cal_prob = float(calibrator.predict_proba([[logit]])[0, 1])

    # ACMG 分级
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
        "offline_mode":     True,   # Evo2/Genos 使用中位数填充
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. 高层接口（推荐使用）
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
        dict 包含：
          cal_prob       - 校准后致病概率 [0, 1]
          acmg_class     - ACMG 分级（P/LP/VUS/LB/B）
          acmg_confidence- 置信度描述
          shap_values    - 各特征 SHAP 贡献值列表
          top_shap_feature - 贡献最大的特征名
          gene, hgvsp    - 基因和蛋白变化（来自 VEP）
          feature_vec    - 实际输入模型的特征向量
          offline_mode   - True（Evo2/Genos 使用中位数填充）
    """
    chrom = str(chrom).replace("chr", "")
    ref, alt = ref.upper().strip(), alt.upper().strip()

    if artifacts is None:
        artifacts = load_model_artifacts(model_dir)

    # 并行获取 VEP 评分 + gnomAD AF（两者均免费）
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

    # 模型预测
    result = predict_from_scores(vep_scores, gnomad_log_af, artifacts)

    # 补充变异注释信息
    result.update({
        "chrom":      chrom,
        "pos":        pos,
        "ref":        ref,
        "alt":        alt,
        "gene":       vep_scores.get("gene", ""),
        "transcript": vep_scores.get("transcript", ""),
        "hgvsp":      vep_scores.get("hgvsp", ""),
        "consequence":vep_scores.get("consequence", ""),
        "am_class":   vep_scores.get("am_class", ""),
        "sift_pred":  vep_scores.get("sift_pred", ""),
        "gnomad_log_af": gnomad_log_af,
    })

    # 打印摘要
    print(f"\n{'='*50}")
    print(f"变异: chr{chrom}:{pos} {ref}>{alt}")
    print(f"基因: {result['gene']}  蛋白变化: {result['hgvsp']}")
    print(f"致病概率: {result['cal_prob']:.1%}  →  {result['acmg_class']} ({result['acmg_confidence']})")
    print(f"主要贡献特征: {result['top_shap_feature']}")
    print(f"注: Evo2 LLR 和 Genos Score 使用训练集中位数填充（离线模式）")
    print(f"{'='*50}")

    return result


def print_shap_summary(result: dict):
    """打印 SHAP 特征贡献摘要。"""
    print("\nSHAP 特征贡献（正值→致病，负值→良性）:")
    pairs = sorted(zip(result["feature_labels"], result["shap_values"]),
                   key=lambda x: abs(x[1]), reverse=True)
    for label, sv in pairs:
        bar = "█" * int(abs(sv) * 30)
        direction = "→致病" if sv > 0 else "→良性"
        offline = " [中位数填充]" if "*" in label else ""
        print(f"  {label:<18} {sv:+.4f} {direction}  {bar}{offline}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("用法: python predict.py <chrom> <pos> <ref> <alt> [model_dir]")
        print("示例: python predict.py 17 7674220 C T /path/to/SNV-judge")
        print()
        print("无需 Evo2 / Genos API Key，直接使用训练好的模型文件预测。")
        sys.exit(0)

    chrom     = sys.argv[1]
    pos       = int(sys.argv[2])
    ref       = sys.argv[3]
    alt       = sys.argv[4]
    model_dir = sys.argv[5] if len(sys.argv) > 5 else "."

    artifacts = load_model_artifacts(model_dir)
    result    = predict_variant(chrom, pos, ref, alt, artifacts=artifacts)
    print_shap_summary(result)
