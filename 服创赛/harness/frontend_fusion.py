from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.getenv("HARNESS_RUNS_DIR", str(ROOT / "data" / "harness" / "runs")))
FRONTEND_ROOT = Path(os.getenv("FRONTEND_ROOT", str(ROOT.parent / "1")))
FRONTEND_A14_DIR = Path(os.getenv("FRONTEND_A14_DIR", str(FRONTEND_ROOT / "outputs" / "a14")))
BUNDLE_DIR = Path(os.getenv("FRONTEND_BUNDLE_DIR", str(ROOT / "data" / "harness" / "frontend_bundle")))
BUNDLE_PATH = Path(os.getenv("FRONTEND_BUNDLE_PATH", str(BUNDLE_DIR / "latest_frontend_bundle.json")))

STUDY_PREDICTION_PATH = Path(
    os.getenv("STUDY_PREDICTION_PATH", str(ROOT / "study" / "data" / "deliverables" / "study" / "data" / "study_prediction_output.csv"))
)
STUDY_EXPLANATION_PATH = Path(
    os.getenv("STUDY_EXPLANATION_PATH", str(ROOT / "study" / "data" / "deliverables" / "study" / "data" / "study_explanation_output.csv"))
)
LIFE_PREDICTION_PATH = Path(
    os.getenv("LIFE_PREDICTION_PATH", str(ROOT / "life" / "data" / "deliverables" / "life" / "data" / "prediction_output.csv"))
)
SPORT_PREDICTION_PATH = Path(
    os.getenv("SPORT_PREDICTION_PATH", str(ROOT / "data" / "deliverables" / "sport" / "data" / "sport_prediction_output.csv"))
)

CORE_ARTIFACT_STATUS_KEYS = [
    "master_table",
    "interventions",
    "reports",
    "group_profile",
    "pattern_summary",
    "life_shap",
    "sport_shap",
    "map_scores",
]

DOMAIN_LABELS = {"study": "学习", "life": "生活", "sport": "运动", "unknown": "未知"}
MAP_LABELS = {"M": "动机", "A": "能力", "P": "提示"}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def read_csv_records(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    except Exception:
        return []


def write_csv_records(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def artifact_dir_for_run(run_file: Path) -> Path:
    run_id = run_file.stem if run_file.name else "unknown_run"
    return BUNDLE_DIR / "student_artifacts" / run_id


def configured_source_paths() -> dict[str, Path]:
    return {
        "study_prediction": STUDY_PREDICTION_PATH,
        "study_explanation": STUDY_EXPLANATION_PATH,
        "life_prediction": LIFE_PREDICTION_PATH,
        "sport_prediction": SPORT_PREDICTION_PATH,
    }


def snapshot_source_paths(run_file: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    artifact_dir = artifact_dir_for_run(run_file)
    snapshot_dir = artifact_dir / "domain_sources"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_paths: dict[str, Path] = {}
    manifest: dict[str, Any] = {
        "run_id": run_file.stem if run_file.name else "",
        "run_file": str(run_file),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {},
    }
    for key, source_path in configured_source_paths().items():
        suffix = source_path.suffix or ".csv"
        target_path = snapshot_dir / f"{key}{suffix}"
        exists = source_path.exists()
        source_info: dict[str, Any] = {
            "original_path": str(source_path),
            "snapshot_path": str(target_path),
            "status": "fresh" if exists else "missing",
            "source_mtime": source_path.stat().st_mtime if exists else None,
            "source_size": source_path.stat().st_size if exists else None,
        }
        if exists:
            shutil.copy2(source_path, target_path)
        snapshot_paths[key] = target_path
        manifest["sources"][key] = source_info
    (artifact_dir / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return snapshot_paths, manifest


def existing_snapshot_paths(run_file: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    artifact_dir = artifact_dir_for_run(run_file)
    manifest_path = artifact_dir / "source_manifest.json"
    manifest = read_json(manifest_path, {})
    paths: dict[str, Path] = {}
    for key, source_path in configured_source_paths().items():
        manifest_source = manifest.get("sources", {}).get(key, {}) if isinstance(manifest, dict) else {}
        snapshot_path = Path(str(manifest_source.get("snapshot_path") or artifact_dir / "domain_sources" / f"{key}{source_path.suffix or '.csv'}"))
        paths[key] = snapshot_path
    return paths, manifest if isinstance(manifest, dict) else {}


def normalize_sid(value: Any) -> str:
    return str(value or "").strip().lower()


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in [None, ""]:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float | None]) -> float | None:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def round_value(value: Any, digits: int = 4) -> float | str:
    number = to_float(value)
    if number is None:
        return ""
    return round(number, digits)


def quantile(values: list[float | None], q: float) -> float | None:
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


def value_ratios(values: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "未知"
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values()) or 1
    return {key: round(count / total, 4) for key, count in counts.items()}


def mode(values: list[str], default: str = "未知") -> str:
    counts: dict[str, int] = {}
    for value in values:
        key = value or default
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return default
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def aggregate_prediction_rows(rows: list[dict[str, str]], id_keys: list[str], score_keys: list[str], extra_keys: list[str] | None = None) -> dict[str, dict[str, Any]]:
    extra_keys = extra_keys or []
    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = ""
        for key in id_keys:
            sid = normalize_sid(row.get(key))
            if sid:
                break
        if not sid:
            continue
        score = None
        for key in score_keys:
            score = to_float(row.get(key))
            if score is not None:
                break
        if score is None:
            continue
        bucket = buckets.setdefault(sid, {"scores": [], "extras": {}})
        bucket["scores"].append(score)
        for key in extra_keys:
            if key not in bucket["extras"] and row.get(key) not in [None, ""]:
                bucket["extras"][key] = row.get(key)

    output: dict[str, dict[str, Any]] = {}
    for sid, bucket in buckets.items():
        output[sid] = {"risk": mean(bucket["scores"]), **bucket["extras"]}
    return output


def build_study_explanations(path: Path = STUDY_EXPLANATION_PATH) -> dict[str, dict[str, str]]:
    rows = read_csv_records(path)
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        sid = normalize_sid(row.get("XH") or row.get("student_id") or row.get("sid"))
        if not sid:
            continue
        output[sid] = {
            "study_shap_top1": str(row.get("TOP_FEATURE_1", "") or ""),
            "study_shap_top2": str(row.get("TOP_FEATURE_2", "") or ""),
            "study_shap_top3": str(row.get("TOP_FEATURE_3", "") or ""),
        }
    return output


def classify_feature_to_map(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["score", "grade", "course", "credit", "exam", "成绩", "课程", "考试"]):
        return "A"
    if any(token in lowered for token in ["active", "punch", "reply", "club", "参与", "打卡", "互动"]):
        return "M"
    return "P"


def choose_risk_level(total_risks: list[float | None], value: float | None) -> str:
    if value is None:
        return "未知"
    low = quantile(total_risks, 0.2)
    high = quantile(total_risks, 0.8)
    if high is not None and value >= high:
        return "高风险"
    if low is not None and value <= low:
        return "低风险"
    return "中风险"


def intervention_text(pattern: str, dominant_dimension: str, dominant_map: str) -> str:
    dimension = DOMAIN_LABELS.get(dominant_dimension, "未知")
    mechanism = MAP_LABELS.get(dominant_map, "提示")
    if pattern == "多维叠加高风险型":
        return f"建议先处理{dimension}主导问题，再联动学习、生活、运动三域进行阶段性跟踪。"
    if pattern == "学习主导风险型":
        return "建议围绕学习任务拆解、课程资源补足和阶段性反馈建立帮扶闭环。"
    if pattern == "生活失衡型":
        return "建议围绕作息、校园活动轨迹和外部提醒建立稳定生活节律。"
    if pattern == "运动薄弱型":
        return "建议建立低门槛运动计划，并用打卡提醒和过程反馈提高持续性。"
    return f"建议优先从{mechanism}机制入手，配合提醒、反馈和资源支持开展干预。"


def build_student_artifacts(source_paths: dict[str, Path] | None = None) -> dict[str, Any]:
    paths = source_paths or configured_source_paths()
    study = aggregate_prediction_rows(
        read_csv_records(paths["study_prediction"]),
        ["XH", "student_id", "sid"],
        ["FINAL_SCORE", "DOMAIN_SCORE", "BASE_SCORE"],
        ["MODEL_VERSION", "MODEL_NAME"],
    )
    life = aggregate_prediction_rows(
        read_csv_records(paths["life_prediction"]),
        ["student_id", "sid", "XH"],
        ["risk_score", "pred_prob", "DOMAIN_SCORE"],
        ["risk_level"],
    )
    sport = aggregate_prediction_rows(
        read_csv_records(paths["sport_prediction"]),
        ["sid", "student_id", "XH"],
        ["pred_fail_proba", "risk_score", "DOMAIN_SCORE"],
        ["best_cls_model", "pred_zf_grade"],
    )
    study_explanations = build_study_explanations(paths["study_explanation"])

    all_sids = sorted(set(study) | set(life) | set(sport))
    records: list[dict[str, Any]] = []
    for sid in all_sids:
        study_risk = study.get(sid, {}).get("risk")
        life_risk = life.get(sid, {}).get("risk")
        sport_risk = sport.get(sid, {}).get("risk")
        total_risk = mean([study_risk, life_risk, sport_risk])
        dimension_scores = {
            "study": study_risk,
            "life": life_risk,
            "sport": sport_risk,
        }
        available_dimensions = [(name, value) for name, value in dimension_scores.items() if value is not None]
        dominant_dimension = max(available_dimensions, key=lambda item: item[1])[0] if available_dimensions else "unknown"
        explanation = study_explanations.get(sid, {})
        feature_text = " ".join(str(v) for v in explanation.values())
        dominant_map = classify_feature_to_map(feature_text)
        m_score = 1.0 if dominant_map == "M" else 0.0
        a_score = 1.0 if dominant_map == "A" else 0.0
        p_score = 1.0 if dominant_map == "P" else 0.0

        records.append(
            {
                "student_id": sid,
                "life_risk": life_risk,
                "study_risk": study_risk,
                "sport_risk": sport_risk,
                "total_risk": total_risk,
                "dominant_dimension": dominant_dimension,
                "study_best_cls_model": study.get(sid, {}).get("MODEL_VERSION") or study.get(sid, {}).get("MODEL_NAME") or "study_v1",
                "life_shap_top1": "",
                "life_shap_top2": "",
                "life_shap_top3": "",
                "study_shap_top1": explanation.get("study_shap_top1", ""),
                "study_shap_top2": explanation.get("study_shap_top2", ""),
                "study_shap_top3": explanation.get("study_shap_top3", ""),
                "sport_shap_top1": "",
                "sport_shap_top2": "",
                "sport_shap_top3": "",
                "M_score": m_score,
                "A_score": a_score,
                "P_score": p_score,
                "dominant_MAP": dominant_map,
            }
        )

    total_risks = [to_float(row.get("total_risk")) for row in records]
    dim_high = {
        "study": quantile([to_float(row.get("study_risk")) for row in records], 0.7),
        "life": quantile([to_float(row.get("life_risk")) for row in records], 0.7),
        "sport": quantile([to_float(row.get("sport_risk")) for row in records], 0.7),
    }
    high_total = quantile(total_risks, 0.8)
    mid_total = quantile(total_risks, 0.5)

    for row in records:
        row["risk_level"] = choose_risk_level(total_risks, to_float(row.get("total_risk")))
        high_dims = sum(
            1
            for domain in ["study", "life", "sport"]
            if dim_high[domain] is not None and to_float(row.get(f"{domain}_risk")) is not None and to_float(row.get(f"{domain}_risk")) >= dim_high[domain]
        )
        dominant = str(row.get("dominant_dimension", "unknown"))
        if high_total is not None and to_float(row.get("total_risk")) is not None and to_float(row.get("total_risk")) >= high_total and high_dims >= 2:
            pattern = "多维叠加高风险型"
            reason = "至少两个维度风险同时偏高，且综合风险进入高位区间。"
        elif dominant == "study" and mid_total is not None and (to_float(row.get("total_risk")) or 0) >= mid_total:
            pattern = "学习主导风险型"
            reason = "学习风险为当前最突出的主导维度。"
        elif dominant == "life" and mid_total is not None and (to_float(row.get("total_risk")) or 0) >= mid_total:
            pattern = "生活失衡型"
            reason = "生活风险为当前最突出的主导维度。"
        elif dominant == "sport":
            pattern = "运动薄弱型"
            reason = "运动风险为当前最突出的主导维度。"
        else:
            pattern = "提示缺失型" if row.get("dominant_MAP") == "P" else "能力不足型"
            reason = "根据 MAP 机制归因为主进行规则分型。"
        row["pattern_label"] = pattern
        row["pattern_reason"] = reason
        row["intervention_type"] = MAP_LABELS.get(str(row.get("dominant_MAP")), "提示")
        row["intervention_text"] = intervention_text(pattern, dominant, str(row.get("dominant_MAP", "P")))
        row["priority"] = "P1" if row["risk_level"] == "高风险" else ("P2" if row["risk_level"] == "中风险" else "P3")
        top_features = "、".join(v for v in [row.get("study_shap_top1"), row.get("life_shap_top1"), row.get("sport_shap_top1")] if v) or "暂无显著行为特征"
        row["profile_text"] = (
            f"该学生综合风险为{row['risk_level']}，主要由{DOMAIN_LABELS.get(dominant, '未知')}维度驱动。"
            f"行为上表现为：{top_features}。"
            f"从机制上看，问题主要集中在{row['intervention_type']}。"
            f"在群体中，该学生属于{pattern}。建议优先开展{row['intervention_type']}干预。"
        )

    records.sort(key=lambda row: to_float(row.get("total_risk"), -1) or -1, reverse=True)
    return {
        "records": records,
        "domain_row_counts": {"study": len(study), "life": len(life), "sport": len(sport)},
        "source_paths": {
            "study_prediction": str(paths["study_prediction"]),
            "study_explanation": str(paths["study_explanation"]),
            "life_prediction": str(paths["life_prediction"]),
            "sport_prediction": str(paths["sport_prediction"]),
        },
        "source_rows": {
            "study_explanation": len(study_explanations),
        },
    }


def build_pattern_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row.get("pattern_label", "未知")), []).append(row)
    output = []
    total = len(records) or 1
    for pattern, rows in sorted(grouped.items(), key=lambda item: len(item[1]), reverse=True):
        main_dimension = mode([DOMAIN_LABELS.get(str(row.get("dominant_dimension")), "未知") for row in rows])
        main_map = mode([MAP_LABELS.get(str(row.get("dominant_MAP")), "提示") for row in rows], "提示")
        output.append(
            {
                "pattern_label": pattern,
                "student_count": len(rows),
                "ratio": round(len(rows) / total, 4),
                "avg_total_risk": round_value(mean([to_float(row.get("total_risk")) for row in rows])),
                "main_dimension": main_dimension,
                "main_MAP": main_map,
            }
        )
    return output


def build_group_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(records),
        "risk_level_ratio": value_ratios([str(row.get("risk_level", "未知")) for row in records]),
        "dominant_dimension_ratio": value_ratios([DOMAIN_LABELS.get(str(row.get("dominant_dimension")), "未知") for row in records]),
        "pattern_ratio": value_ratios([str(row.get("pattern_label", "未知")) for row in records]),
        "dominant_map_ratio": value_ratios([MAP_LABELS.get(str(row.get("dominant_MAP")), "提示") for row in records]),
        "avg_risk": {
            "life_risk": round_value(mean([to_float(row.get("life_risk")) for row in records])),
            "study_risk": round_value(mean([to_float(row.get("study_risk")) for row in records])),
            "sport_risk": round_value(mean([to_float(row.get("sport_risk")) for row in records])),
            "total_risk": round_value(mean([to_float(row.get("total_risk")) for row in records])),
        },
        "high_frequency_shap_features": {"study": [], "life": [], "sport": []},
    }


def build_reports(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for row in records:
        total = round_value(row.get("total_risk"), 3)
        dimension = DOMAIN_LABELS.get(str(row.get("dominant_dimension")), "未知")
        risk_agent = f"该学生当前综合风险为{row.get('risk_level')}，综合风险分约为{total}，主导风险维度为{dimension}。"
        behavior_agent = f"关键行为证据包括：{row.get('study_shap_top1') or '暂无显著学习特征'}。"
        mechanism_agent = f"从 MAP 机制看，当前主导机制为{row.get('intervention_type')}。"
        intervention_agent = str(row.get("intervention_text", "暂无个性化干预建议。"))
        reports.append(
            {
                "student_id": row.get("student_id"),
                "summary": f"{risk_agent}{behavior_agent}{mechanism_agent}建议：{intervention_agent}",
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


def build_demo_case(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    row = records[0]
    return {
        "student_id": row.get("student_id"),
        "chain": {
            "submodel_risk": {
                "life_risk": round_value(row.get("life_risk")),
                "study_risk": round_value(row.get("study_risk")),
                "sport_risk": round_value(row.get("sport_risk")),
            },
            "fusion": {
                "total_risk": round_value(row.get("total_risk")),
                "risk_level": row.get("risk_level"),
                "dominant_dimension": DOMAIN_LABELS.get(str(row.get("dominant_dimension")), "未知"),
            },
            "shap": {"study": [row.get("study_shap_top1", ""), row.get("study_shap_top2", ""), row.get("study_shap_top3", "")]},
            "map": {
                "M_score": round_value(row.get("M_score")),
                "A_score": round_value(row.get("A_score")),
                "P_score": round_value(row.get("P_score")),
                "dominant_MAP": row.get("intervention_type"),
            },
            "pattern": row.get("pattern_label"),
            "profile_text": row.get("profile_text"),
            "intervention_text": row.get("intervention_text"),
        },
    }


def extract_built_from_domains(result: dict[str, Any], run_file: Path) -> list[str]:
    final_decision = result.get("final_decision", {}) if isinstance(result, dict) else {}
    domains = final_decision.get("domains", {}) if isinstance(final_decision, dict) else {}
    if isinstance(domains, dict) and domains:
        return sorted(str(key) for key in domains.keys())
    parts = run_file.stem.split("_")
    return [part for part in parts if part in {"study", "life", "sport"}]


def is_preferred_multi_domain_run(payload: Any, path: Path) -> bool:
    if not isinstance(payload, dict):
        return False
    domains = extract_built_from_domains(payload, path)
    final_decision = payload.get("final_decision", {})
    status = (
        payload.get("system_status")
        or payload.get("status")
        or (final_decision.get("system_status") if isinstance(final_decision, dict) else "")
        or (final_decision.get("decision") if isinstance(final_decision, dict) else "")
    )
    return (
        path.name.startswith("multi_domain")
        and len(set(domains)) >= 2
        and str(status or "").strip().lower() not in {"", "running"}
    )


def write_student_artifacts(run_file: Path, records: list[dict[str, Any]], pattern_summary: list[dict[str, Any]], group_profile: dict[str, Any], reports: list[dict[str, Any]], demo_case: dict[str, Any]) -> dict[str, str]:
    artifact_dir = artifact_dir_for_run(run_file)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    master_fields = [
        "student_id", "life_risk", "study_risk", "sport_risk", "total_risk", "risk_level", "dominant_dimension",
        "study_best_cls_model", "life_shap_top1", "life_shap_top2", "life_shap_top3", "study_shap_top1",
        "study_shap_top2", "study_shap_top3", "sport_shap_top1", "sport_shap_top2", "sport_shap_top3",
        "M_score", "A_score", "P_score", "dominant_MAP", "pattern_label", "pattern_reason", "profile_text",
        "intervention_type", "intervention_text", "priority",
    ]
    intervention_records = [
        {
            "student_id": row.get("student_id"),
            "intervention_type": row.get("intervention_type"),
            "intervention_text": row.get("intervention_text"),
            "priority": row.get("priority"),
        }
        for row in records
    ]
    write_csv_records(artifact_dir / "fusion_student_master_table.csv", records, master_fields)
    write_csv_records(artifact_dir / "student_intervention.csv", intervention_records, ["student_id", "intervention_type", "intervention_text", "priority"])
    write_csv_records(artifact_dir / "pattern_summary.csv", pattern_summary, ["pattern_label", "student_count", "ratio", "avg_total_risk", "main_dimension", "main_MAP"])
    (artifact_dir / "group_profile.json").write_text(json.dumps(group_profile, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "student_full_report_multi_agent.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_dir / "demo_case_student.json").write_text(json.dumps(demo_case, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "artifact_dir": str(artifact_dir),
        "fusion_student_master_table": str(artifact_dir / "fusion_student_master_table.csv"),
        "student_intervention": str(artifact_dir / "student_intervention.csv"),
        "pattern_summary": str(artifact_dir / "pattern_summary.csv"),
        "group_profile": str(artifact_dir / "group_profile.json"),
        "student_full_report_multi_agent": str(artifact_dir / "student_full_report_multi_agent.json"),
        "demo_case_student": str(artifact_dir / "demo_case_student.json"),
        "source_manifest": str(artifact_dir / "source_manifest.json"),
    }


def latest_run() -> dict[str, Any]:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(str(RUNS_DIR))
    candidates = sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    fallback: tuple[Path, dict[str, Any]] | None = None
    for path in candidates:
        payload = read_json(path, {})
        if is_preferred_multi_domain_run(payload, path):
            return {"run_record_path": str(path), "result": payload}
        if fallback is None and isinstance(payload, dict) and path.name.startswith("multi_domain"):
            fallback = (path, payload)
    if fallback is not None:
        path, payload = fallback
        return {"run_record_path": str(path), "result": payload}
    if not candidates:
        raise FileNotFoundError(str(RUNS_DIR))
    path = candidates[0]
    return {"run_record_path": str(path), "result": read_json(path, {})}


def build_artifact_freshness(records: list[dict[str, Any]], generated: dict[str, Any], source_manifest: dict[str, Any]) -> dict[str, str]:
    has_records = bool(records)
    source_rows = generated.get("source_rows", {}) if isinstance(generated.get("source_rows", {}), dict) else {}
    life_shap_present = any(row.get("life_shap_top1") or row.get("life_shap_top2") or row.get("life_shap_top3") for row in records)
    sport_shap_present = any(row.get("sport_shap_top1") or row.get("sport_shap_top2") or row.get("sport_shap_top3") for row in records)
    study_explained = int(source_rows.get("study_explanation") or 0) > 0
    freshness = {
        "master_table": "fresh" if has_records else "missing",
        "fusion_student_master_table": "fresh" if has_records else "missing",
        "interventions": "fresh" if has_records else "missing",
        "student_intervention": "fresh" if has_records else "missing",
        "reports": "fresh" if has_records else "missing",
        "student_full_report_multi_agent": "fresh" if has_records else "missing",
        "group_profile": "fresh" if has_records else "missing",
        "pattern_summary": "fresh" if has_records else "missing",
        "study_shap": "fresh" if study_explained else "missing",
        "life_shap": "fresh" if life_shap_present else "missing",
        "sport_shap": "fresh" if sport_shap_present else "missing",
        "map_scores": "rule_derived" if has_records else "missing",
    }
    sources = source_manifest.get("sources", {}) if isinstance(source_manifest, dict) else {}
    for key in ["study_prediction", "life_prediction", "sport_prediction"]:
        if sources.get(key, {}).get("status") == "missing":
            domain = key.split("_")[0]
            freshness[f"{domain}_prediction"] = "missing"
    return freshness


def verify_bundle_consistency(bundle: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    run_file = Path(str(bundle.get("built_from_run_file", "")))
    run_id = str(bundle.get("built_from_run_id", ""))
    if not run_id:
        errors.append("missing built_from_run_id")
    if run_file.stem != run_id:
        errors.append("built_from_run_id does not match built_from_run_file")
    if not run_file.exists():
        errors.append("built_from_run_file does not exist")
    domains = bundle.get("built_from_domains", [])
    if not isinstance(domains, list) or len(set(str(domain) for domain in domains)) < 2:
        errors.append("bundle is not bound to a multi-domain run")
    artifact_paths = bundle.get("artifact_paths", {}) if isinstance(bundle.get("artifact_paths", {}), dict) else {}
    artifact_dir = Path(str(artifact_paths.get("artifact_dir", "")))
    if run_id and artifact_dir.name != run_id:
        errors.append("artifact directory is not run-scoped")
    for key in [
        "fusion_student_master_table",
        "student_intervention",
        "student_full_report_multi_agent",
        "pattern_summary",
        "group_profile",
        "demo_case_student",
        "source_manifest",
    ]:
        path = Path(str(artifact_paths.get(key, "")))
        if not path.exists():
            errors.append(f"missing artifact: {key}")
        elif run_id and run_id not in path.parts:
            errors.append(f"artifact is not under run_id directory: {key}")
    source_manifest = bundle.get("source_manifest", {}) if isinstance(bundle.get("source_manifest", {}), dict) else {}
    if source_manifest.get("run_id") != run_id:
        errors.append("source manifest run_id does not match bundle")
    for key, source in (source_manifest.get("sources", {}) or {}).items():
        if source.get("status") == "fresh" and not Path(str(source.get("snapshot_path", ""))).exists():
            errors.append(f"missing source snapshot: {key}")
    counts = bundle.get("counts", {}) if isinstance(bundle.get("counts", {}), dict) else {}
    if int(counts.get("students") or 0) <= 0:
        errors.append("bundle has no students")
    return not errors, errors


def load_existing_bundle_for_run(run_file: Path) -> dict[str, Any] | None:
    bundle = read_json(BUNDLE_PATH, {})
    if not isinstance(bundle, dict):
        return None
    if bundle.get("built_from_run_id") != run_file.stem:
        return None
    ok, errors = verify_bundle_consistency(bundle)
    if not ok:
        return None
    bundle["artifact_consistency_ok"] = True
    bundle["artifact_consistency_errors"] = errors
    return bundle


def build_frontend_bundle(run_record_path: str | Path | None = None) -> dict[str, Any]:
    if run_record_path:
        run_file = Path(run_record_path)
        harness = {"run_record_path": str(run_file), "result": read_json(run_file, {})}
    else:
        harness = latest_run()
        run_file = Path(str(harness.get("run_record_path", "")))
        existing = load_existing_bundle_for_run(run_file)
        if existing is not None:
            return existing

    snapshot_paths, source_manifest = snapshot_source_paths(run_file)
    generated = build_student_artifacts(snapshot_paths)
    records = generated["records"]
    pattern_summary = build_pattern_summary(records)
    group_profile = build_group_profile(records)
    reports = build_reports(records)
    demo_case = build_demo_case(records)
    intervention_records = [
        {
            "student_id": row.get("student_id"),
            "intervention_type": row.get("intervention_type"),
            "intervention_text": row.get("intervention_text"),
            "priority": row.get("priority"),
        }
        for row in records
    ]
    artifact_paths = write_student_artifacts(run_file, records, pattern_summary, group_profile, reports, demo_case)
    student_artifacts_fresh = bool(records)
    built_from_domains = extract_built_from_domains(harness.get("result", {}), run_file)
    artifact_freshness = build_artifact_freshness(records, generated, source_manifest)

    bundle = {
        "schema_version": "frontend_bundle_v1",
        "bundle_source": "backend_recomputed",
        "source": "backend_http_bundle",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "built_from_run_file": str(run_file),
        "built_from_run_id": run_file.stem if run_file.name else "",
        "built_from_domains": built_from_domains,
        "run_record_path": str(run_file),
        "run_record_mtime": run_file.stat().st_mtime if run_file.exists() else None,
        "student_artifacts_fresh": student_artifacts_fresh,
        "artifact_freshness": artifact_freshness,
        "artifact_paths": artifact_paths,
        "source_manifest": source_manifest,
        "harness": harness,
        "student_source_paths": generated["source_paths"],
        "domain_row_counts": generated["domain_row_counts"],
        "a14_source_dir": str(artifact_paths.get("artifact_dir", "")),
        "master_records": records,
        "pattern_records": pattern_summary,
        "intervention_records": intervention_records,
        "reports": reports,
        "group_profile": group_profile,
        "demo_case": demo_case,
        "counts": {
            "students": len(records),
            "patterns": len(pattern_summary),
            "interventions": len(intervention_records),
            "reports": len(reports),
        },
    }
    ok, errors = verify_bundle_consistency(bundle)
    bundle["artifact_consistency_ok"] = ok
    bundle["artifact_consistency_errors"] = errors
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    BUNDLE_PATH.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle
