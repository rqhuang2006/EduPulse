from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from study_agent import StudyAgent
from study_common import DM_DIR, term_sort_key

ROOT = Path(__file__).resolve().parents[1]
DELIVERABLE_MODEL_DIR = ROOT / "data" / "deliverables" / "study" / "model"
REPORT_JSON_PATH = DM_DIR / "study_group_router_experiments.json"
PACKAGE_A_PATH = DM_DIR / "study_router_stoploss_report.csv"
PACKAGE_B_PATH = DM_DIR / "study_core_plus_behavior_subgroup_report.csv"
PACKAGE_C_PATH = DM_DIR / "study_targeted_feature_catalog.csv"

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


def safe_auc(y_true: pd.Series, scores: np.ndarray) -> float | None:
    y = pd.Series(y_true)
    s = pd.Series(scores)
    mask = y.notna() & s.notna()
    y = y[mask]
    s = s[mask]
    if len(y) == 0 or y.nunique() < 2:
        return None
    return float(roc_auc_score(y, s))


def fixed_precision_recall(y_true: pd.Series, scores: np.ndarray, target_precision: float = 0.70) -> float | None:
    y = pd.Series(y_true)
    s = pd.Series(scores)
    mask = y.notna() & s.notna()
    y = y[mask]
    s = s[mask]
    if len(y) == 0 or y.nunique() < 2:
        return None
    precision, recall, _ = precision_recall_curve(y, s)
    feasible = recall[:-1][precision[:-1] >= target_precision] if len(precision) > 1 else np.array([])
    if len(feasible) == 0:
        return None
    return float(np.max(feasible))


def make_model() -> Pipeline:
    if LGBMClassifier is not None:
        estimator: Any = LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.05)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    estimator = HistGradientBoostingClassifier(random_state=42)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])


def make_linear_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ]
    )


def build_targeted_features(agent: StudyAgent, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = agent._augment_feature_table(data.copy())

    def num(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(np.nan, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[name], errors="coerce")

    def z_by_student(series: pd.Series) -> pd.Series:
        grouped_mean = series.groupby(frame["XH"]).transform("mean")
        grouped_std = series.groupby(frame["XH"]).transform("std").replace(0, np.nan)
        return (series - grouped_mean) / (grouped_std + 1e-6)

    targeted_cols: list[str] = []
    base_triplets = [
        ("FEATURE_GRADE_AVG_SCORE", "prev_grade_avg_score", "hist_grade_avg_score", "grade_avg"),
        ("FEATURE_EXAM_SCORE_AVG", "prev_exam_score_avg", "hist_exam_score_avg", "exam_score"),
        ("FEATURE_LIBRARY_VISIT_COUNT", "prev_library_visit_count", "hist_library_visit_count", "library_visit"),
    ]
    for current_col, prev_col, hist_col, alias in base_triplets:
        if current_col in frame.columns:
            current = num(current_col)
            prev = num(prev_col)
            hist = num(hist_col)
            peak = current.groupby(frame["XH"]).cummax().shift(1)
            frame[f"trend_decline_count_{alias}"] = (current.lt(prev).fillna(False).astype(int) + prev.lt(hist).fillna(False).astype(int))
            frame[f"trend_below_hist_{alias}"] = current.lt(hist).fillna(False).astype(int)
            frame[f"trend_drawdown_{alias}"] = (peak - current) / (peak.abs() + 1e-6)
            frame[f"personal_gap_{alias}"] = current - hist
            frame[f"personal_ratio_{alias}"] = current / (hist.abs() + 1e-6)
            frame[f"personal_z_{alias}"] = z_by_student(current)
            targeted_cols.extend(
                [
                    f"trend_decline_count_{alias}",
                    f"trend_below_hist_{alias}",
                    f"trend_drawdown_{alias}",
                    f"personal_gap_{alias}",
                    f"personal_ratio_{alias}",
                    f"personal_z_{alias}",
                ]
            )

    behavior_cols = [c for c in frame.columns if c.startswith("FEATURE_") and any(k in c for k in ["ATTENDANCE", "CLASS_", "ASSIGNMENT", "EXAM", "LIBRARY"])]
    temporal_cols = [c for c in frame.columns if c.startswith(("prev_", "hist_", "delta_", "ratio_"))]
    frame["feature_behavior_coverage_count"] = frame[behavior_cols].notna().sum(axis=1) if behavior_cols else 0
    frame["feature_temporal_coverage_count"] = frame[temporal_cols].notna().sum(axis=1) if temporal_cols else 0
    frame["feature_behavior_missing_ratio"] = 1.0 - frame[behavior_cols].notna().mean(axis=1) if behavior_cols else 1.0
    frame["feature_has_complete_recent_windows"] = (
        frame[[c for c in ["prev_exam_score_avg", "hist_exam_score_avg", "prev_library_visit_count", "hist_library_visit_count"] if c in frame.columns]]
        .notna()
        .all(axis=1)
        .astype(int)
        if any(c in frame.columns for c in ["prev_exam_score_avg", "hist_exam_score_avg", "prev_library_visit_count", "hist_library_visit_count"])
        else 0
    )
    targeted_cols.extend(
        [
            "feature_behavior_coverage_count",
            "feature_temporal_coverage_count",
            "feature_behavior_missing_ratio",
            "feature_has_complete_recent_windows",
        ]
    )

    if "FEATURE_GRADE_AVG_SCORE" in frame.columns and "FEATURE_LIBRARY_VISIT_COUNT" in frame.columns:
        frame["imbalance_grade_vs_library_z"] = z_by_student(num("FEATURE_GRADE_AVG_SCORE")) - z_by_student(num("FEATURE_LIBRARY_VISIT_COUNT"))
        targeted_cols.append("imbalance_grade_vs_library_z")
    if "FEATURE_ASSIGNMENT_SCORE_AVG" in frame.columns and "FEATURE_EXAM_SCORE_AVG" in frame.columns:
        frame["imbalance_assignment_vs_exam_level"] = num("FEATURE_ASSIGNMENT_SCORE_AVG") - num("FEATURE_EXAM_SCORE_AVG")
        targeted_cols.append("imbalance_assignment_vs_exam_level")
    if "delta_grade_avg_score" in frame.columns and "delta_library_visit_count" in frame.columns:
        frame["imbalance_grade_vs_library_delta"] = num("delta_grade_avg_score") - num("delta_library_visit_count")
        targeted_cols.append("imbalance_grade_vs_library_delta")
    if "FEATURE_ATTENDANCE_EVENT_COUNT" in frame.columns and "delta_grade_avg_score" in frame.columns:
        frame["imbalance_attendance_stable_study_drop"] = (
            (num("FEATURE_ATTENDANCE_EVENT_COUNT").fillna(0) > 0).astype(int) * (num("delta_grade_avg_score").fillna(0) < 0).astype(int)
        )
        targeted_cols.append("imbalance_attendance_stable_study_drop")

    catalog = pd.DataFrame(
        {
            "feature_name": targeted_cols,
            "feature_category": [
                "availability"
                if name.startswith("feature_")
                else "imbalance"
                if name.startswith("imbalance_")
                else "continuity_or_personal"
                for name in targeted_cols
            ],
        }
    )
    catalog.to_csv(PACKAGE_C_PATH, index=False, encoding="utf-8-sig")
    return frame, targeted_cols


def main() -> None:
    config = json.loads((DELIVERABLE_MODEL_DIR / "study_model_config.json").read_text(encoding="utf-8"))
    bundle = joblib.load(DELIVERABLE_MODEL_DIR / "study_model.pkl")
    raw_train = pd.read_csv(DM_DIR / "study_train_table.csv")

    agent = StudyAgent(
        request={
            "request_id": "study_group_router_experiments",
            "domain": "study",
            "term_id": "all",
            "run_mode": "infer",
            "input_paths": {},
            "feature_version": config.get("feature_version"),
            "model_version": config.get("model_version"),
            "enable_fallback": True,
            "enable_explanation": False,
        }
    )
    train = agent._augment_feature_table(raw_train)
    train = train.dropna(subset=["LABEL"]).copy()
    train["LABEL"] = pd.to_numeric(train["LABEL"], errors="coerce").fillna(0).astype(int)

    order = train["TERM_ID"].map(term_sort_key) if "TERM_ID" in train.columns else pd.Series(range(len(train)))
    sorted_idx = order.sort_values().index
    split_at = max(1, int(len(sorted_idx) * 0.8))
    train_idx, valid_idx = sorted_idx[:split_at], sorted_idx[split_at:]
    valid_df = train.loc[valid_idx].copy()
    y_valid = valid_df["LABEL"].astype(int)

    core_features = config.get("core_feature_columns", [])
    behavior_features = config.get("behavior_feature_columns", [])
    subgroup_features = config.get("subgroup_feature_columns", [])
    for column in config.get("feature_columns", []):
        if column not in train.columns:
            train[column] = 0.0
        if column not in valid_df.columns:
            valid_df[column] = 0.0
    for column in subgroup_features:
        if column not in train.columns:
            train[column] = 0.0
        if column not in valid_df.columns:
            valid_df[column] = 0.0

    core_family_map = config.get("core_feature_families", {})
    behavior_family_map = config.get("behavior_feature_families", {})

    def presence(df: pd.DataFrame, cols: list[str]) -> pd.Series:
        usable = [c for c in cols if c in df.columns]
        return df[usable].notna().any(axis=1) if usable else pd.Series(False, index=df.index)

    core_available = presence(valid_df, core_family_map.get("grade", [])) & presence(valid_df, core_family_map.get("course", []))
    behavior_hits = pd.Series(0, index=valid_df.index)
    for cols in behavior_family_map.values():
        behavior_hits = behavior_hits + presence(valid_df, cols).astype(int)
    valid_df["row_level_study_data_mode"] = np.where(~core_available, "degraded_sparse", np.where(behavior_hits >= 1, "core_plus_behavior", "core_only"))

    core_x = valid_df.reindex(columns=core_features).apply(pd.to_numeric, errors="coerce")
    behavior_x = valid_df.reindex(columns=behavior_features).apply(pd.to_numeric, errors="coerce")
    subgroup_x = valid_df.reindex(columns=subgroup_features).apply(pd.to_numeric, errors="coerce")
    core_prob = np.asarray(bundle["core_model"].predict_proba(core_x))[:, 1]
    behavior_prob = np.full(len(valid_df), np.nan)
    subgroup_prob = np.full(len(valid_df), np.nan)
    enhanced_mask = valid_df["row_level_study_data_mode"].eq("core_plus_behavior").to_numpy()
    if behavior_features and bundle.get("behavior_model") is not None and enhanced_mask.any():
        behavior_prob[enhanced_mask] = np.asarray(bundle["behavior_model"].predict_proba(behavior_x.loc[enhanced_mask]))[:, 1]
    if subgroup_features and bundle.get("subgroup_model") is not None and enhanced_mask.any():
        subgroup_prob[enhanced_mask] = np.asarray(bundle["subgroup_model"].predict_proba(subgroup_x.loc[enhanced_mask]))[:, 1]
    weights = config.get("score_combination", {})
    blend_prob = np.asarray(core_prob, dtype=float)
    blend_prob = np.where(
        enhanced_mask & ~pd.isna(behavior_prob),
        core_prob * float(weights.get("core_plus_behavior_core_weight", 0.7)) + behavior_prob * float(weights.get("core_plus_behavior_behavior_weight", 0.3)),
        blend_prob,
    )
    routed_prob = np.asarray(core_prob, dtype=float)
    routed_prob = np.where(enhanced_mask & ~pd.isna(subgroup_prob), subgroup_prob, routed_prob)
    routed_prob = np.where(enhanced_mask & pd.isna(subgroup_prob) & ~pd.isna(behavior_prob), blend_prob, routed_prob)
    routed_prob = np.where(valid_df["row_level_study_data_mode"].eq("degraded_sparse").to_numpy(), core_prob, routed_prob)

    package_a = pd.DataFrame(
        [
            {"score_version": "core_only_score", "auc": safe_auc(y_valid, core_prob)},
            {"score_version": "current_blended_score", "auc": safe_auc(y_valid, blend_prob)},
            {"score_version": "routed_score", "auc": safe_auc(y_valid, routed_prob)},
        ]
    )
    package_a.to_csv(PACKAGE_A_PATH, index=False, encoding="utf-8-sig")

    subgroup_train = train.loc[train_idx].copy()
    subgroup_valid = train.loc[valid_idx].copy()
    train_core_available = presence(subgroup_train, core_family_map.get("grade", [])) & presence(subgroup_train, core_family_map.get("course", []))
    train_behavior_hits = pd.Series(0, index=subgroup_train.index)
    valid_behavior_hits = pd.Series(0, index=subgroup_valid.index)
    for cols in behavior_family_map.values():
        train_behavior_hits = train_behavior_hits + presence(subgroup_train, cols).astype(int)
        valid_behavior_hits = valid_behavior_hits + presence(subgroup_valid, cols).astype(int)
    subgroup_train = subgroup_train.loc[train_core_available & (train_behavior_hits >= 1)].copy()
    valid_core_available = presence(subgroup_valid, core_family_map.get("grade", [])) & presence(subgroup_valid, core_family_map.get("course", []))
    subgroup_valid = subgroup_valid.loc[valid_core_available & (valid_behavior_hits >= 1)].copy()

    targeted_train, targeted_cols = build_targeted_features(agent, subgroup_train)
    targeted_valid, _ = build_targeted_features(agent, subgroup_valid)
    baseline_cols = [col for col in config.get("feature_columns", []) if col in targeted_train.columns]
    targeted_all_cols = list(dict.fromkeys(baseline_cols + [col for col in targeted_cols if col in targeted_train.columns]))

    subgroup_rows: list[dict[str, Any]] = []
    for name, cols, maker in [
        ("baseline_subgroup_model", baseline_cols, make_model),
        ("targeted_subgroup_model", targeted_all_cols, make_model),
        ("targeted_subgroup_linear", targeted_all_cols, make_linear_model),
    ]:
        if subgroup_train.empty or subgroup_valid.empty:
            continue
        x_train = targeted_train.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
        x_valid = targeted_valid.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
        y_train = targeted_train["LABEL"].astype(int)
        y_sub_valid = targeted_valid["LABEL"].astype(int)
        model = maker()
        model.fit(x_train, y_train)
        score = np.asarray(model.predict_proba(x_valid))[:, 1]
        pred = (score >= 0.5).astype(int)
        subgroup_rows.append(
            {
                "experiment_name": name,
                "rows": int(len(y_sub_valid)),
                "positives": int(y_sub_valid.sum()),
                "feature_count": len(cols),
                "auc": safe_auc(y_sub_valid, score),
                "pr_auc": float(average_precision_score(y_sub_valid, score)) if y_sub_valid.nunique() > 1 else None,
                "recall_at_precision_0_70": fixed_precision_recall(y_sub_valid, score, target_precision=0.70),
                "f1": float(f1_score(y_sub_valid, pred, zero_division=0)),
                "recall": float(recall_score(y_sub_valid, pred, zero_division=0)),
                "precision": float(precision_score(y_sub_valid, pred, zero_division=0)),
            }
        )
    package_b = pd.DataFrame(subgroup_rows)
    package_b.to_csv(PACKAGE_B_PATH, index=False, encoding="utf-8-sig")

    report = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "package_a_router_stoploss": package_a.to_dict(orient="records"),
        "package_b_core_plus_behavior_subgroup": subgroup_rows,
        "package_c_targeted_feature_count": len(targeted_cols),
        "package_c_targeted_feature_catalog_path": str(PACKAGE_C_PATH),
    }
    REPORT_JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
