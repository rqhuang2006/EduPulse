from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
OUT_NEXT_STUDY_DIR = ROOT / "outputs_next" / "study"
AAA_DM_DIR = ROOT / "AAA" / "data" / "dm"
A14_DIR = OUT_DIR / "a14"
PLAYGROUND_DIR = ROOT.parents[1]
DEFAULT_SHAPMAP_DIR = PLAYGROUND_DIR / "fuchuang_shapmapl" / "fuchuang_final"

DOMAIN_CN = {"study": "学习", "life": "生活", "sport": "运动", "unknown": "未知"}
MAP_LABELS = {"M": "动机", "A": "能力", "P": "提示"}
INTERVENTION_LIBRARY = {
    "M": ["目标激励", "正反馈机制", "成就感强化"],
    "A": ["学习节奏支持", "任务拆解", "学习资源补足"],
    "P": ["行为提醒", "外部约束", "环境触发优化"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate A14 deliverables from existing outputs.")
    parser.add_argument("--shapmap-root", type=Path, default=DEFAULT_SHAPMAP_DIR, help="Path to SHAP/MAP project root.")
    parser.add_argument("--out-dir", type=Path, default=A14_DIR, help="Directory for generated A14 deliverables.")
    parser.add_argument(
        "--study-prediction-file",
        type=Path,
        default=None,
        help="Optional explicit study prediction file. If omitted, use outputs_next/study, then outputs, then AAA dm fallback.",
    )
    return parser.parse_args()


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def normalize_sid(value: object) -> str:
    return str(value or "").strip().lower()


def to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        if value in [None, ""]:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def round_or_none(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def quantile(values: Iterable[Optional[float]], q: float) -> Optional[float]:
    valid = sorted(float(v) for v in values if v is not None)
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    pos = (len(valid) - 1) * q
    low = int(pos)
    high = min(low + 1, len(valid) - 1)
    frac = pos - low
    return valid[low] * (1 - frac) + valid[high] * frac


def value_counts_ratio(values: Iterable[object]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total = 0
    for value in values:
        key = str(value if value not in [None, ""] else "未知")
        counts[key] = counts.get(key, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {key: round(count / total, 4) for key, count in counts.items()}


def minmax_inverted(values: List[Optional[float]]) -> List[Optional[float]]:
    valid = [v for v in values if v is not None]
    if not valid:
        return [None for _ in values]
    low = min(valid)
    high = max(valid)
    if high == low:
        return [0.5 if v is not None else None for v in values]
    output: List[Optional[float]] = []
    for value in values:
        if value is None:
            output.append(None)
        else:
            normalized = (value - low) / (high - low)
            output.append(max(0.0, min(1.0, 1 - normalized)))
    return output


def aggregate_mean_by_sid(rows: List[Dict[str, str]], value_key: str, extra_keys: Optional[List[str]] = None) -> Dict[str, Dict[str, object]]:
    extra_keys = extra_keys or []
    buckets: Dict[str, Dict[str, object]] = {}
    for row in rows:
        sid = normalize_sid(row.get("sid"))
        if not sid:
            continue
        value = to_float(row.get(value_key))
        if value is None:
            continue
        bucket = buckets.setdefault(sid, {"_values": [], "_terms": set()})
        bucket["_values"].append(value)
        if row.get("term_id"):
            bucket["_terms"].add(str(row["term_id"]))
        for extra_key in extra_keys:
            if extra_key not in bucket and row.get(extra_key) not in [None, ""]:
                bucket[extra_key] = row.get(extra_key)
    output: Dict[str, Dict[str, object]] = {}
    for sid, bucket in buckets.items():
        output[sid] = {"mean": mean(bucket["_values"]), "term_count": len(bucket["_terms"])}
        for extra_key in extra_keys:
            if extra_key in bucket:
                output[sid][extra_key] = bucket[extra_key]
    return output


def resolve_study_prediction_file(explicit: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            OUT_NEXT_STUDY_DIR / "predictions_full.csv",
            OUT_NEXT_STUDY_DIR / "predictions_test.csv",
            OUT_DIR / "predictions_full.csv",
            OUT_DIR / "predictions_test.csv",
            AAA_DM_DIR / "study_prediction_output.csv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def normalize_study_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for row in rows:
        sid = row.get("sid") or row.get("SID") or row.get("xh") or row.get("XH") or row.get("student_id") or row.get("STUDENT_ID")
        term_id = row.get("term_id") or row.get("TERM_ID") or row.get("term")
        pred = row.get("pred_fail_proba")
        if pred in [None, ""]:
            pred = row.get("DOMAIN_SCORE")
        model_name = row.get("best_cls_model") or row.get("PRIMARY_MODEL") or row.get("MODEL_VERSION") or "StudyAgent"
        normalized.append(
            {
                "sid": "" if sid is None else str(sid),
                "term_id": "" if term_id is None else str(term_id),
                "pred_fail_proba": "" if pred is None else str(pred),
                "best_cls_model": str(model_name),
            }
        )
    return normalized


def build_domain_risks(study_prediction_file: Optional[Path] = None) -> Dict[str, Dict[str, Dict[str, object]]]:
    study_path = resolve_study_prediction_file(study_prediction_file)
    study_rows = normalize_study_rows(read_csv_dicts(study_path)) if study_path is not None else []
    study = aggregate_mean_by_sid(study_rows, "pred_fail_proba", ["best_cls_model"])

    life_path = OUT_DIR / "life" / "predictions_full.csv"
    if not life_path.exists():
        life_path = OUT_DIR / "life" / "predictions_test.csv"
    life_rows = read_csv_dicts(life_path)
    life = aggregate_mean_by_sid(life_rows, "pred_prob")

    sport_path = OUT_DIR / "sport" / "predictions_full.csv"
    if not sport_path.exists():
        sport_path = OUT_DIR / "sport" / "predictions_test.csv"
    sport_rows = read_csv_dicts(sport_path)
    direct_risks = [to_float(row.get("pred_fail_proba")) for row in sport_rows]
    if any(value is not None for value in direct_risks):
        risk_scores = direct_risks
    else:
        raw_scores = [to_float(row.get("pred_zf_score")) for row in sport_rows]
        risk_scores = minmax_inverted(raw_scores)
    sport_augmented: List[Dict[str, str]] = []
    for row, risk in zip(sport_rows, risk_scores):
        new_row = dict(row)
        new_row["sport_risk"] = "" if risk is None else str(risk)
        sport_augmented.append(new_row)
    sport = aggregate_mean_by_sid(sport_augmented, "sport_risk")

    return {"study": study, "life": life, "sport": sport}


def top_shap_features(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    by_sid: Dict[str, List[Tuple[float, str]]] = {}
    for row in rows:
        sid = normalize_sid(row.get("sid"))
        feature = str(row.get("feature", "")).strip()
        shap_value = to_float(row.get("shap_value"))
        if not sid or not feature or shap_value is None:
            continue
        by_sid.setdefault(sid, []).append((abs(shap_value), feature))
    output: Dict[str, Dict[str, str]] = {}
    for sid, pairs in by_sid.items():
        pairs.sort(key=lambda item: item[0], reverse=True)
        features = [feature for _, feature in pairs[:3]]
        output[sid] = {
            "top1": features[0] if len(features) > 0 else "",
            "top2": features[1] if len(features) > 1 else "",
            "top3": features[2] if len(features) > 2 else "",
            "top_features": "、".join(features),
        }
    return output


def build_shap_tables(shapmap_root: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    base = shapmap_root / "output" / "shap" / "individual"
    files = {
        "study": base / "shap_individual_study_risk_cls_real_feature.csv",
        "life": base / "shap_life_individual_simple_1775823941.csv",
        "sport": base / "shap_individual_sport_risk.csv",
    }
    return {domain: top_shap_features(read_csv_dicts(path)) for domain, path in files.items()}


def build_global_shap_summary(shapmap_root: Path) -> List[Dict[str, object]]:
    base = shapmap_root / "output" / "shap" / "global"
    files = {
        "study": base / "shap_global_study_risk_cls.csv",
        "life": base / "shap_life_simple_1775823941.csv",
        "sport": base / "shap_global_sport_risk.csv",
    }
    rows: List[Dict[str, object]] = []
    for domain, path in files.items():
        extracted: List[Tuple[str, float]] = []
        for row in read_csv_dicts(path):
            feature = str(row.get("feature", "")).strip()
            importance = None
            for key in ["importance", "mean_abs_shap", "abs_mean", "shap_value"]:
                importance = to_float(row.get(key))
                if importance is not None:
                    break
            if feature and importance is not None:
                extracted.append((feature, importance))
        extracted.sort(key=lambda item: item[1], reverse=True)
        for rank, (feature, importance) in enumerate(extracted[:10], start=1):
            rows.append({"domain": domain, "rank": rank, "feature": feature, "importance": round_or_none(importance, 6)})
    return rows


def build_map_scores(shapmap_root: Path) -> Dict[str, Dict[str, object]]:
    rows = read_csv_dicts(shapmap_root / "output" / "map" / "individual" / "student_map_scores_new.csv")
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        sid = normalize_sid(row.get("student_id") or row.get("sid"))
        if not sid:
            continue
        bucket = buckets.setdefault(sid, {"M": [], "A": [], "P": []})
        for key in ["M", "A", "P"]:
            value = to_float(row.get(key))
            if value is not None:
                bucket[key].append(value)
    output: Dict[str, Dict[str, object]] = {}
    for sid, bucket in buckets.items():
        m_score = mean(bucket["M"]) or 0.0
        a_score = mean(bucket["A"]) or 0.0
        p_score = mean(bucket["P"]) or 0.0
        dominant = max([("M", m_score), ("A", a_score), ("P", p_score)], key=lambda item: item[1])[0]
        output[sid] = {"M_score": m_score, "A_score": a_score, "P_score": p_score, "dominant_MAP": dominant}
    return output


def detect_irregular_life_behavior(row: Dict[str, object]) -> bool:
    text = " ".join(str(row.get(key, "") or "") for key in ["life_shap_top1", "life_shap_top2", "life_shap_top3", "life_shap_top_features"]).lower()
    keywords = ["night", "late", "door", "net", "sleep", "library", "consume", "夜", "网", "作息", "消费"]
    return any(keyword in text for keyword in keywords)


def compute_risk_level(total_risks: List[Optional[float]], value: Optional[float]) -> str:
    if value is None:
        return "未知"
    low_cut = quantile(total_risks, 0.2)
    high_cut = quantile(total_risks, 0.8)
    if high_cut is not None and value >= high_cut:
        return "高风险"
    if low_cut is not None and value <= low_cut:
        return "低风险"
    return "中风险"


def choose_dominant_dimension(row: Dict[str, object]) -> str:
    valid = [(name, row.get(f"{name}_risk")) for name in ["life", "study", "sport"] if row.get(f"{name}_risk") is not None]
    if not valid:
        return "unknown"
    valid.sort(key=lambda item: item[1], reverse=True)
    return valid[0][0]


def assign_pattern(row: Dict[str, object], thresholds: Dict[str, Optional[float]]) -> Tuple[str, str]:
    study_risk = row.get("study_risk") or 0.0
    life_risk = row.get("life_risk") or 0.0
    sport_risk = row.get("sport_risk") or 0.0
    total_risk = row.get("total_risk") or 0.0
    dominant_dimension = row.get("dominant_dimension")
    dominant_map = row.get("dominant_MAP")

    high_dimensions = sum(
        [
            int(thresholds["study"] is not None and study_risk >= thresholds["study"]),
            int(thresholds["life"] is not None and life_risk >= thresholds["life"]),
            int(thresholds["sport"] is not None and sport_risk >= thresholds["sport"]),
        ]
    )

    if thresholds["total_high"] is not None and total_risk >= thresholds["total_high"] and high_dimensions >= 2:
        return "多维叠加高风险型", "至少两个维度风险同时偏高，且综合风险进入高位区间。"
    if dominant_dimension == "study" and thresholds["total_mid"] is not None and total_risk >= thresholds["total_mid"]:
        return "学习主导风险型", f"学习风险高于生活与运动，关键行为特征集中在 {row.get('study_shap_top1') or '学习行为'}。"
    if dominant_dimension == "life" and thresholds["total_mid"] is not None and total_risk >= thresholds["total_mid"]:
        if detect_irregular_life_behavior(row):
            return "生活失衡型", "生活维度风险最高，且 SHAP 特征表现出作息、上网或消费等生活失衡迹象。"
        return "生活失衡型", f"生活维度风险最高，生活行为特征集中在 {row.get('life_shap_top1') or '生活行为'}。"
    if dominant_dimension == "sport" and thresholds["sport"] is not None and sport_risk >= thresholds["sport"]:
        return "运动薄弱型", f"运动维度风险最高，核心特征集中在 {row.get('sport_shap_top1') or '运动行为'}。"
    if dominant_map == "A":
        return "能力不足型", "MAP 机制以能力维度为主，说明核心问题偏向能力、节奏和资源不足。"
    if dominant_map == "P":
        return "提示缺失型", "MAP 机制以提示维度为主，说明外部提醒、约束和环境触发较弱。"
    return "动机波动型", "未落入前述高风险模式，当前更接近动机驱动的波动状态。"


def intervention_text(pattern_label: str, dominant_map: str, dominant_dimension: str) -> str:
    dimension_text = DOMAIN_CN.get(dominant_dimension, dominant_dimension)
    items = INTERVENTION_LIBRARY.get(dominant_map or "P", INTERVENTION_LIBRARY["P"])
    if pattern_label == "学习主导风险型":
        return f"建议优先围绕{dimension_text}维度开展干预，结合{items[0]}、{items[1]}与{items[2]}，提升学习执行与反馈闭环。"
    if pattern_label == "生活失衡型":
        return f"建议先修正{dimension_text}维度的作息与环境行为，采用{items[0]}、{items[1]}与{items[2]}建立稳定习惯。"
    if pattern_label == "运动薄弱型":
        return f"建议围绕{dimension_text}维度建立低门槛运动计划，用{items[0]}、{items[1]}与{items[2]}增强持续性。"
    if pattern_label == "多维叠加高风险型":
        return f"建议采用分阶段干预方案，先解决{dimension_text}主导问题，再以{items[0]}、{items[1]}与{items[2]}联动其他维度。"
    return f"建议从{MAP_LABELS.get(dominant_map, dominant_map)}机制入手，优先使用{items[0]}、{items[1]}与{items[2]}推动改进。"


def priority_from_risk(risk_level: str, total_risk: Optional[float]) -> str:
    value = total_risk or 0.0
    if risk_level == "高风险" or value >= 0.8:
        return "P1"
    if risk_level == "中风险" or value >= 0.5:
        return "P2"
    return "P3"


def build_master_records(shapmap_root: Path, study_prediction_file: Optional[Path] = None) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    domain_risks = build_domain_risks(study_prediction_file)
    shap_tables = build_shap_tables(shapmap_root)
    map_scores = build_map_scores(shapmap_root)
    shap_global = build_global_shap_summary(shapmap_root)

    all_sids = set()
    for domain in domain_risks.values():
        all_sids.update(domain.keys())
    for domain in shap_tables.values():
        all_sids.update(domain.keys())
    all_sids.update(map_scores.keys())

    records: List[Dict[str, object]] = []
    for sid in sorted(all_sids):
        study_info = domain_risks["study"].get(sid, {})
        life_info = domain_risks["life"].get(sid, {})
        sport_info = domain_risks["sport"].get(sid, {})
        map_info = map_scores.get(sid, {"M_score": 0.0, "A_score": 0.0, "P_score": 0.0, "dominant_MAP": "P"})
        study_shap = shap_tables["study"].get(sid, {})
        life_shap = shap_tables["life"].get(sid, {})
        sport_shap = shap_tables["sport"].get(sid, {})

        row: Dict[str, object] = {
            "student_id": sid,
            "life_risk": life_info.get("mean"),
            "study_risk": study_info.get("mean"),
            "sport_risk": sport_info.get("mean"),
            "study_best_cls_model": study_info.get("best_cls_model", "StudyAgent"),
            "M_score": map_info.get("M_score", 0.0),
            "A_score": map_info.get("A_score", 0.0),
            "P_score": map_info.get("P_score", 0.0),
            "dominant_MAP": map_info.get("dominant_MAP", "P"),
            "life_shap_top1": life_shap.get("top1", ""),
            "life_shap_top2": life_shap.get("top2", ""),
            "life_shap_top3": life_shap.get("top3", ""),
            "life_shap_top_features": life_shap.get("top_features", ""),
            "study_shap_top1": study_shap.get("top1", ""),
            "study_shap_top2": study_shap.get("top2", ""),
            "study_shap_top3": study_shap.get("top3", ""),
            "study_shap_top_features": study_shap.get("top_features", ""),
            "sport_shap_top1": sport_shap.get("top1", ""),
            "sport_shap_top2": sport_shap.get("top2", ""),
            "sport_shap_top3": sport_shap.get("top3", ""),
            "sport_shap_top_features": sport_shap.get("top_features", ""),
        }
        row["total_risk"] = mean([row.get("life_risk"), row.get("study_risk"), row.get("sport_risk")])
        row["dominant_dimension"] = choose_dominant_dimension(row)
        records.append(row)

    total_risks = [row.get("total_risk") for row in records]
    thresholds = {
        "study": quantile([row.get("study_risk") for row in records], 0.7),
        "life": quantile([row.get("life_risk") for row in records], 0.7),
        "sport": quantile([row.get("sport_risk") for row in records], 0.7),
        "total_high": quantile(total_risks, 0.8),
        "total_mid": quantile(total_risks, 0.5),
    }

    for row in records:
        row["risk_level"] = compute_risk_level(total_risks, row.get("total_risk"))
        pattern_label, pattern_reason = assign_pattern(row, thresholds)
        row["pattern_label"] = pattern_label
        row["pattern_reason"] = pattern_reason
        row["intervention_type"] = MAP_LABELS.get(str(row["dominant_MAP"]), "提示")
        row["intervention_text"] = intervention_text(str(row["pattern_label"]), str(row["dominant_MAP"]), str(row["dominant_dimension"]))
        row["priority"] = priority_from_risk(str(row["risk_level"]), row.get("total_risk"))
        top_behaviors = "、".join(
            [value for value in [row.get("study_shap_top1"), row.get("life_shap_top1"), row.get("sport_shap_top1")] if value]
        ) or "暂无显著行为特征"
        row["profile_text"] = (
            f"该学生综合风险为{row['risk_level']}，主要由{DOMAIN_CN.get(str(row['dominant_dimension']), '未知')}维度驱动。"
            f"行为上表现为：{top_behaviors}。"
            f"从机制上看，问题主要集中在{MAP_LABELS.get(str(row['dominant_MAP']), str(row['dominant_MAP']))}。"
            f"在群体中，该学生属于{row['pattern_label']}。"
            f"建议优先从{row['intervention_type']}干预入手。"
        )

    records.sort(key=lambda item: item.get("total_risk") or -1, reverse=True)
    return records, shap_global


def mode_value(values: List[str], default: str) -> str:
    counts: Dict[str, int] = {}
    for value in values:
        key = value or default
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return default
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_group_profile(master: List[Dict[str, object]], shap_global: List[Dict[str, object]]) -> Dict[str, object]:
    avg_risk = {
        "life_risk": round_or_none(mean(row.get("life_risk") for row in master)),
        "study_risk": round_or_none(mean(row.get("study_risk") for row in master)),
        "sport_risk": round_or_none(mean(row.get("sport_risk") for row in master)),
        "total_risk": round_or_none(mean(row.get("total_risk") for row in master)),
    }
    high_frequency_shap = {}
    for domain in ["study", "life", "sport"]:
        high_frequency_shap[domain] = [row["feature"] for row in shap_global if row.get("domain") == domain][:5]
    return {
        "sample_count": len(master),
        "risk_level_ratio": value_counts_ratio(row.get("risk_level") for row in master),
        "dominant_dimension_ratio": value_counts_ratio(DOMAIN_CN.get(str(row.get("dominant_dimension")), "未知") for row in master),
        "pattern_ratio": value_counts_ratio(row.get("pattern_label") for row in master),
        "dominant_map_ratio": value_counts_ratio(MAP_LABELS.get(str(row.get("dominant_MAP")), "未知") for row in master),
        "avg_risk": avg_risk,
        "high_frequency_shap_features": high_frequency_shap,
    }


def build_demo_case(master: List[Dict[str, object]]) -> Dict[str, object]:
    if not master:
        return {}
    row = master[0]
    return {
        "student_id": row["student_id"],
        "chain": {
            "submodel_risk": {
                "life_risk": round_or_none(row.get("life_risk")),
                "study_risk": round_or_none(row.get("study_risk")),
                "sport_risk": round_or_none(row.get("sport_risk")),
            },
            "fusion": {
                "total_risk": round_or_none(row.get("total_risk")),
                "risk_level": row.get("risk_level"),
                "dominant_dimension": DOMAIN_CN.get(str(row.get("dominant_dimension")), "未知"),
            },
            "shap": {
                "life": [row.get("life_shap_top1", ""), row.get("life_shap_top2", ""), row.get("life_shap_top3", "")],
                "study": [row.get("study_shap_top1", ""), row.get("study_shap_top2", ""), row.get("study_shap_top3", "")],
                "sport": [row.get("sport_shap_top1", ""), row.get("sport_shap_top2", ""), row.get("sport_shap_top3", "")],
            },
            "map": {
                "M_score": round_or_none(row.get("M_score")),
                "A_score": round_or_none(row.get("A_score")),
                "P_score": round_or_none(row.get("P_score")),
                "dominant_MAP": MAP_LABELS.get(str(row.get("dominant_MAP")), "未知"),
            },
            "pattern": row.get("pattern_label"),
            "profile_text": row.get("profile_text"),
            "intervention_text": row.get("intervention_text"),
        },
    }


def build_multi_agent_reports(master: List[Dict[str, object]]) -> List[Dict[str, object]]:
    reports: List[Dict[str, object]] = []
    for row in master:
        risk_agent = (
            f"该学生当前综合风险为{row['risk_level']}，综合风险分约为{round_or_none(row.get('total_risk'), 3)}，"
            f"主导风险维度为{DOMAIN_CN.get(str(row.get('dominant_dimension')), '未知')}。"
        )
        behavior_agent = (
            f"关键行为证据主要集中在学习、生活、运动三个维度的代表特征上，"
            f"其中较突出的特征包括：{row.get('study_shap_top1') or '无'}、{row.get('life_shap_top1') or '无'}、{row.get('sport_shap_top1') or '无'}。"
        )
        mechanism_agent = (
            f"从 MAP 机制看，当前主导机制为{MAP_LABELS.get(str(row.get('dominant_MAP')), '未知')}，"
            f"M/A/P 分值分别约为 {round_or_none(row.get('M_score'), 3)} / {round_or_none(row.get('A_score'), 3)} / {round_or_none(row.get('P_score'), 3)}。"
        )
        intervention_agent = str(row.get("intervention_text", "暂无个性化干预建议。"))
        summary = f"{risk_agent}{behavior_agent}{mechanism_agent}建议：{intervention_agent}"

        reports.append(
            {
                "student_id": row["student_id"],
                "summary": summary,
                "risk_agent": risk_agent,
                "behavior_agent": behavior_agent,
                "mechanism_agent": mechanism_agent,
                "intervention_agent": intervention_agent,
                "report_agent": row.get("profile_text", ""),
                "intervention_text": intervention_agent,
                "priority": row.get("priority", ""),
                "pattern_label": row.get("pattern_label", ""),
            }
        )
    return reports


def build_single_student_rule_report(master: List[Dict[str, object]]) -> Dict[str, object]:
    if not master:
        return {}
    row = master[0]
    return {
        "student_id": row["student_id"],
        "risk_level": row["risk_level"],
        "total_risk": round_or_none(row.get("total_risk")),
        "dominant_dimension": DOMAIN_CN.get(str(row.get("dominant_dimension")), "未知"),
        "dominant_MAP": MAP_LABELS.get(str(row.get("dominant_MAP")), "未知"),
        "pattern_label": row.get("pattern_label"),
        "pattern_reason": row.get("pattern_reason"),
        "profile_text": row.get("profile_text"),
        "intervention_text": row.get("intervention_text"),
    }


def write_catalog(out_dir: Path) -> None:
    content = """# A14 分析成果清单

1. `01_risk_distribution.csv`
   个体综合风险分布结果，用于展示整体风险谱系。
2. `02_dimension_comparison.csv`
   `life / study / sport` 三维风险对比结果，用于展示维度差异。
3. `03_risk_level_stats.csv`
   高中低风险分层结果，用于分层汇报与筛查。
4. `04_dominant_dimension_stats.csv`
   学生主导风险维度分布结果，用于识别主要矛盾维度。
5. `05_shap_global_summary.csv`
   SHAP 全局关键行为特征排序结果，用于解释行为驱动因素。
6. `06_map_summary.csv`
   MAP 机制占比结果，用于说明动机 / 能力 / 提示主导结构。
7. `07_pattern_summary.csv`
   四类及以上学生行为模式结果，用于答辩中的模式发现部分。
8. `08_group_profile.json`
   群体画像结果，用于报告中的群体视角总结。
9. `09_student_profile.csv`
   个体画像结果，用于个性化评价报告。
10. `10_student_intervention.csv`
    个性化干预建议结果，用于闭环干预输出。
"""
    (out_dir / "analysis_outputs_catalog.md").write_text(content, encoding="utf-8")


def write_fusion_rule(out_dir: Path) -> None:
    content = """# 融合规则说明

- 综合风险：`total_risk = mean(life_risk, study_risk, sport_risk)`
- 学习风险：直接使用学习域模型输出的 `pred_fail_proba`
- 生活风险：直接使用生活域模型输出的 `pred_prob`
- 运动风险：对 `pred_zf_score` 做反向 Min-Max 归一化，分数越低风险越高
- 风险等级：
  - 高风险：综合风险位于前 20%
  - 中风险：综合风险位于中间 60%
  - 低风险：综合风险位于后 20%
- 主导维度：`life_risk / study_risk / sport_risk` 中最大者
- MAP 汇总逻辑：对同一学生在不同子模型中的 `M/A/P` 取均值，再以最大值作为 `dominant_MAP`
- 行为模式：按综合风险、主导维度与 `dominant_MAP` 联合打标，保证输出不少于 4 类模式
"""
    (out_dir / "fusion_rule.md").write_text(content, encoding="utf-8")


def write_map_rule_table(out_dir: Path) -> None:
    rows = []
    for code, label in MAP_LABELS.items():
        for strategy in INTERVENTION_LIBRARY[code]:
            rows.append({"MAP_code": code, "MAP_name": label, "intervention_strategy": strategy})
    write_csv_dicts(out_dir / "map_to_intervention_rule.csv", rows, ["MAP_code", "MAP_name", "intervention_strategy"])


def export_outputs(master: List[Dict[str, object]], shap_global: List[Dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    master_fields = [
        "student_id", "life_risk", "study_risk", "sport_risk", "total_risk", "risk_level", "dominant_dimension",
        "study_best_cls_model",
        "life_shap_top1", "life_shap_top2", "life_shap_top3", "study_shap_top1", "study_shap_top2", "study_shap_top3",
        "sport_shap_top1", "sport_shap_top2", "sport_shap_top3", "M_score", "A_score", "P_score", "dominant_MAP",
        "pattern_label", "pattern_reason", "profile_text", "intervention_type", "intervention_text", "priority",
    ]
    write_csv_dicts(out_dir / "fusion_student_master_table.csv", master, master_fields)

    risk_distribution = [{"student_id": row["student_id"], "total_risk": round_or_none(row.get("total_risk")), "risk_rank": idx} for idx, row in enumerate(master, start=1)]
    write_csv_dicts(out_dir / "01_risk_distribution.csv", risk_distribution, ["student_id", "total_risk", "risk_rank"])

    dimension_comparison = [
        {
            "student_id": row["student_id"],
            "life_risk": round_or_none(row.get("life_risk")),
            "study_risk": round_or_none(row.get("study_risk")),
            "sport_risk": round_or_none(row.get("sport_risk")),
            "total_risk": round_or_none(row.get("total_risk")),
        }
        for row in master
    ]
    write_csv_dicts(out_dir / "02_dimension_comparison.csv", dimension_comparison, ["student_id", "life_risk", "study_risk", "sport_risk", "total_risk"])

    risk_groups: Dict[str, List[Dict[str, object]]] = {}
    for row in master:
        risk_groups.setdefault(str(row["risk_level"]), []).append(row)
    risk_stats = [
        {
            "risk_level": key,
            "student_count": len(rows),
            "avg_total_risk": round_or_none(mean(row.get("total_risk") for row in rows)),
            "ratio": round(len(rows) / max(len(master), 1), 4),
        }
        for key, rows in sorted(risk_groups.items(), key=lambda item: len(item[1]), reverse=True)
    ]
    write_csv_dicts(out_dir / "03_risk_level_stats.csv", risk_stats, ["risk_level", "student_count", "avg_total_risk", "ratio"])

    dimension_groups: Dict[str, List[Dict[str, object]]] = {}
    for row in master:
        dimension_groups.setdefault(str(row["dominant_dimension"]), []).append(row)
    dimension_stats = [
        {
            "dominant_dimension": key,
            "dimension_name": DOMAIN_CN.get(key, key),
            "student_count": len(rows),
            "avg_total_risk": round_or_none(mean(row.get("total_risk") for row in rows)),
            "ratio": round(len(rows) / max(len(master), 1), 4),
        }
        for key, rows in sorted(dimension_groups.items(), key=lambda item: len(item[1]), reverse=True)
    ]
    write_csv_dicts(out_dir / "04_dominant_dimension_stats.csv", dimension_stats, ["dominant_dimension", "dimension_name", "student_count", "avg_total_risk", "ratio"])

    write_csv_dicts(out_dir / "05_shap_global_summary.csv", shap_global, ["domain", "rank", "feature", "importance"])

    map_groups: Dict[str, List[Dict[str, object]]] = {}
    for row in master:
        map_groups.setdefault(str(row["dominant_MAP"]), []).append(row)
    map_summary = [
        {
            "dominant_MAP": key,
            "map_name": MAP_LABELS.get(key, key),
            "student_count": len(rows),
            "avg_M_score": round_or_none(mean(row.get("M_score") for row in rows)),
            "avg_A_score": round_or_none(mean(row.get("A_score") for row in rows)),
            "avg_P_score": round_or_none(mean(row.get("P_score") for row in rows)),
            "ratio": round(len(rows) / max(len(master), 1), 4),
        }
        for key, rows in sorted(map_groups.items(), key=lambda item: len(item[1]), reverse=True)
    ]
    write_csv_dicts(out_dir / "06_map_summary.csv", map_summary, ["dominant_MAP", "map_name", "student_count", "avg_M_score", "avg_A_score", "avg_P_score", "ratio"])

    pattern_groups: Dict[str, List[Dict[str, object]]] = {}
    for row in master:
        pattern_groups.setdefault(str(row["pattern_label"]), []).append(row)
    pattern_summary = []
    for key, rows in sorted(pattern_groups.items(), key=lambda item: len(item[1]), reverse=True):
        main_dimension = mode_value([str(row.get("dominant_dimension") or "") for row in rows], "unknown")
        main_map = mode_value([str(row.get("dominant_MAP") or "") for row in rows], "P")
        pattern_summary.append(
            {
                "pattern_label": key,
                "student_count": len(rows),
                "ratio": round(len(rows) / max(len(master), 1), 4),
                "avg_total_risk": round_or_none(mean(row.get("total_risk") for row in rows)),
                "main_dimension": DOMAIN_CN.get(main_dimension, main_dimension),
                "main_MAP": MAP_LABELS.get(main_map, main_map),
            }
        )
    write_csv_dicts(out_dir / "07_pattern_summary.csv", pattern_summary, ["pattern_label", "student_count", "ratio", "avg_total_risk", "main_dimension", "main_MAP"])
    write_csv_dicts(out_dir / "pattern_summary.csv", pattern_summary, ["pattern_label", "student_count", "ratio", "avg_total_risk", "main_dimension", "main_MAP"])
    write_csv_dicts(out_dir / "student_pattern_label.csv", master, ["student_id", "pattern_label", "pattern_reason"])

    group_profile = build_group_profile(master, shap_global)
    (out_dir / "08_group_profile.json").write_text(json.dumps(group_profile, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "group_profile.json").write_text(json.dumps(group_profile, ensure_ascii=False, indent=2), encoding="utf-8")

    student_profile = [
        {
            "student_id": row["student_id"],
            "risk_level": row["risk_level"],
            "dominant_dimension": DOMAIN_CN.get(str(row["dominant_dimension"]), "未知"),
            "pattern_label": row["pattern_label"],
            "profile_text": row["profile_text"],
        }
        for row in master
    ]
    write_csv_dicts(out_dir / "09_student_profile.csv", student_profile, ["student_id", "risk_level", "dominant_dimension", "pattern_label", "profile_text"])
    write_csv_dicts(out_dir / "student_profile.csv", student_profile, ["student_id", "risk_level", "dominant_dimension", "pattern_label", "profile_text"])

    student_intervention = [
        {
            "student_id": row["student_id"],
            "intervention_type": row["intervention_type"],
            "intervention_text": row["intervention_text"],
            "priority": row["priority"],
        }
        for row in master
    ]
    write_csv_dicts(out_dir / "10_student_intervention.csv", student_intervention, ["student_id", "intervention_type", "intervention_text", "priority"])
    write_csv_dicts(out_dir / "student_intervention.csv", student_intervention, ["student_id", "intervention_type", "intervention_text", "priority"])

    (out_dir / "demo_case_student.json").write_text(json.dumps(build_demo_case(master), ensure_ascii=False, indent=2), encoding="utf-8")

    report_rows = [
        {
            "student_id": row["student_id"],
            "risk_level": row["risk_level"],
            "life_risk": round_or_none(row.get("life_risk")),
            "study_risk": round_or_none(row.get("study_risk")),
            "sport_risk": round_or_none(row.get("sport_risk")),
            "total_risk": round_or_none(row.get("total_risk")),
            "dominant_dimension": DOMAIN_CN.get(str(row["dominant_dimension"]), "未知"),
            "dominant_MAP": MAP_LABELS.get(str(row["dominant_MAP"]), str(row["dominant_MAP"])),
            "pattern_label": row["pattern_label"],
            "profile_text": row["profile_text"],
            "intervention_text": row["intervention_text"],
            "priority": row["priority"],
        }
        for row in master
    ]
    (out_dir / "student_full_report.json").write_text(json.dumps(report_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    multi_agent_reports = build_multi_agent_reports(master)
    (out_dir / "student_full_report_multi_agent.json").write_text(json.dumps(multi_agent_reports, ensure_ascii=False, indent=2), encoding="utf-8")

    single_student_rule_report = build_single_student_rule_report(master)
    (out_dir / "single_student_rule_report.json").write_text(json.dumps(single_student_rule_report, ensure_ascii=False, indent=2), encoding="utf-8")

    write_map_rule_table(out_dir)
    write_catalog(out_dir)
    write_fusion_rule(out_dir)


def main() -> None:
    args = parse_args()
    master, shap_global = build_master_records(args.shapmap_root, args.study_prediction_file)
    if not master:
        raise RuntimeError("No fusion-ready student records were found.")
    export_outputs(master, shap_global, args.out_dir)
    print(f"A14 deliverables generated in: {args.out_dir}")


if __name__ == "__main__":
    main()