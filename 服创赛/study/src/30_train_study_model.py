from __future__ import annotations

from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from study_common import DM_DIR, collect_feature_columns, ensure_dirs, model_params, term_sort_key, write_json
from study_eval_report import build_eval_reports, write_eval_outputs
from study_expert_train import build_oof_scores, train_behavior_corrector, train_single_fail_expert
from study_feature_engine import (
    add_temporal_features,
    add_interaction_features,
    build_targeted_features,
    add_course_risk_features,
    apply_feature_engineering,
    infer_feature_layer,
    summarize_feature_layers,
)
from study_routing_policy import apply_serving_policy, resolve_policy

EXCLUDED_MODEL_FEATURES = {"FEATURE_MISSING_RATE"}
CORE_FAMILY_PREFIXES = {
    "grade": ("FEATURE_GRADE_", "FEATURE_CET_"),
    "course": ("FEATURE_COURSE_",),
}
BEHAVIOR_FAMILY_PREFIXES = {
    "attendance": ("FEATURE_ATTENDANCE_",),
    "class_task": ("FEATURE_CLASS_",),
    "assignment": ("FEATURE_ASSIGNMENT_",),
    "exam": ("FEATURE_EXAM_",),
    "library": ("FEATURE_LIBRARY_",),
    "online": ("FEATURE_ONLINE_",),
}
FEATURE_LAYER_SUMMARY_PATH = DM_DIR / "study_feature_layer_summary.json"
SELECTED_FEATURES_PATH = DM_DIR / "study_selected_features.csv"
FEATURE_SCREENING_REPORT_PATH = DM_DIR / "study_feature_screening_report.json"
SUBGROUP_FEATURE_SCREENING_REPORT_PATH = DM_DIR / "study_subgroup_feature_screening_report.json"
EVAL_REPORT_PATH = DM_DIR / "study_eval_report.json"
SUBGROUP_METRICS_PATH = DM_DIR / "study_subgroup_metrics.csv"
CONFIDENCE_ZONE_REPORT_PATH = DM_DIR / "study_confidence_zone_report.csv"
ANCHOR_BASELINE_PATH = DM_DIR / "study_frozen_anchor_baseline.json"


def screen_features(train_df: pd.DataFrame, infer_df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[str], dict[str, Any]]:
    kept: list[str] = []
    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        layer = infer_feature_layer(feature)
        train_series = pd.to_numeric(train_df.get(feature, pd.Series(np.nan, index=train_df.index)), errors="coerce")
        infer_series = pd.to_numeric(infer_df.get(feature, pd.Series(np.nan, index=infer_df.index)), errors="coerce")
        train_nonnull = float(train_series.notna().mean()) if len(train_series) else 0.0
        infer_nonnull = float(infer_series.notna().mean()) if len(infer_series) else 0.0
        train_unique = int(train_series.nunique(dropna=True)) if len(train_series) else 0
        infer_unique = int(infer_series.nunique(dropna=True)) if len(infer_series) else 0

        keep = True
        reason = "kept"
        if train_nonnull <= 0.0:
            keep = False
            reason = "dropped_train_all_missing"
        elif train_unique <= 1:
            keep = False
            reason = "dropped_train_constant"
        elif layer in {"behavior", "temporal", "interaction"} and infer_nonnull <= 0.0:
            keep = False
            reason = "dropped_infer_all_missing"
        elif layer in {"temporal", "interaction"} and train_nonnull < 0.10:
            keep = False
            reason = "dropped_low_train_coverage"
        elif layer in {"behavior", "temporal", "interaction"} and infer_nonnull < 0.05:
            keep = False
            reason = "dropped_low_infer_coverage"

        if keep:
            kept.append(feature)

        rows.append(
            {
                "feature_name": feature,
                "feature_layer": layer,
                "train_nonnull_rate": train_nonnull,
                "infer_nonnull_rate": infer_nonnull,
                "train_unique": train_unique,
                "infer_unique": infer_unique,
                "kept": keep,
                "decision_reason": reason,
            }
        )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_feature_count": len(feature_cols),
        "kept_feature_count": len(kept),
        "dropped_feature_count": len(feature_cols) - len(kept),
        "kept_counts_by_layer": {
            layer: sum(1 for row in rows if row["kept"] and row["feature_layer"] == layer)
            for layer in ["core", "behavior", "temporal", "interaction"]
        },
        "dropped_counts_by_reason": {
            reason: sum(1 for row in rows if row["decision_reason"] == reason and not row["kept"])
            for reason in sorted({row["decision_reason"] for row in rows if not row["kept"]})
        },
        "rows": rows,
    }
    return kept, report


def build_primary(random_state: int):
    """Build the primary model - prioritize LightGBM/CatBoost for AUC ceiling."""
    # Try CatBoost first (often best for tabular with categorical features)
    try:
        from catboost import CatBoostClassifier

        return "CatBoostClassifier", CatBoostClassifier(
            random_state=random_state,
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            verbose=False,
            early_stopping_rounds=20,
        )
    except Exception:
        pass

    # Try LightGBM second (fast and effective)
    try:
        from lightgbm import LGBMClassifier

        return "LightGBMClassifier", LGBMClassifier(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    except Exception:
        pass

    # Fallback to HistGradientBoosting (sklearn built-in)
    from sklearn.ensemble import HistGradientBoostingClassifier

    return "HistGradientBoostingClassifier", HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
    )


def build_subgroup_primary(random_state: int):
    """Build a more regularized model for the subgroup expert to combat overfitting."""
    # Try CatBoost first
    try:
        from catboost import CatBoostClassifier

        return "CatBoostClassifier", CatBoostClassifier(
            random_state=random_state,
            iterations=200,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=5,
            verbose=False,
            early_stopping_rounds=15,
            class_weights=[1, 2],  # Higher weight for positive class
        )
    except Exception:
        pass

    # Try LightGBM second
    try:
        from lightgbm import LGBMClassifier

        return "LightGBMClassifier", LGBMClassifier(
            random_state=random_state,
            n_estimators=150,
            max_depth=4,
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.7,
            class_weight="balanced",
        )
    except Exception:
        pass

    # Fallback
    from sklearn.ensemble import HistGradientBoostingClassifier

    return "HistGradientBoostingClassifier", HistGradientBoostingClassifier(
        random_state=random_state, max_depth=3, min_samples_leaf=20
    )


def predict_prob(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    pred = model.predict(x)
    return np.asarray(pred, dtype=float)


def compute_dual_metrics(model, x: pd.DataFrame, y: pd.Series, train_df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute dual metrics: overall + subgroup breakdown.
    
    Returns:
    - overall AUC
    - core_plus_behavior subset AUC
    - single_fail subset AUC  
    - uncertain band AUC
    """
    prob = predict_prob(model, x)
    report = {}
    
    # Overall metrics
    if y.nunique() > 1:
        report["overall_auc"] = float(roc_auc_score(y, prob))
    
    # Core + Behavior subset metrics
    if "LABEL_SUBTYPE" in train_df.columns and train_df.index.equals(x.index):
        # Single fail subset
        single_fail_mask = train_df["LABEL_SUBTYPE"] == "single_fail"
        if single_fail_mask.sum() > 0 and y[single_fail_mask].nunique() > 1:
            report["single_fail_auc"] = float(roc_auc_score(y[single_fail_mask], prob[single_fail_mask]))
        
        # Overall low subset
        overall_low_mask = train_df["LABEL_SUBTYPE"] == "overall_low"
        if overall_low_mask.sum() > 0 and y[overall_low_mask].nunique() > 1:
            report["overall_low_auc"] = float(roc_auc_score(y[overall_low_mask], prob[overall_low_mask]))
    
    # Uncertain band metrics (predictions between 0.3-0.7)
    uncertain_mask = (prob >= 0.3) & (prob <= 0.7)
    if uncertain_mask.sum() > 0 and y[uncertain_mask].nunique() > 1:
        report["uncertain_band_auc"] = float(roc_auc_score(y[uncertain_mask], prob[uncertain_mask]))
        report["uncertain_band_count"] = int(uncertain_mask.sum())
    
    return report


def metric_report(model, x: pd.DataFrame, y: pd.Series) -> dict:
    if len(y) == 0:
        return {}
    prob = predict_prob(model, x)
    pred = (prob >= 0.5).astype(int)
    report = {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
    }
    if y.nunique() > 1:
        report["auc"] = float(roc_auc_score(y, prob))
    return report


def feature_group(columns: list[str], groups: dict[str, tuple[str, ...]]) -> dict[str, list[str]]:
    return {name: [col for col in columns if any(col.startswith(prefix) for prefix in prefixes)] for name, prefixes in groups.items()}


def presence(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    usable = [c for c in cols if c in df.columns]
    return df[usable].notna().any(axis=1) if usable else pd.Series(False, index=df.index)


def coverage_by_family(df: pd.DataFrame, family_map: dict[str, list[str]]) -> dict[str, float]:
    coverage: dict[str, float] = {}
    for family, cols in family_map.items():
        coverage[family] = float(df[cols].notna().any(axis=1).mean()) if cols else 0.0
    return coverage


def classify_behavior_layers(coverage: dict[str, float]) -> dict[str, str]:
    states: dict[str, str] = {}
    for family, rate in coverage.items():
        if rate >= 0.50:
            states[family] = "stable_behavior"
        elif rate >= 0.05:
            states[family] = "recoverable_behavior"
        else:
            states[family] = "unavailable_behavior"
    return states


def fit_model_bundle(train_x: pd.DataFrame, train_y: pd.Series, valid_x: pd.DataFrame, valid_y: pd.Series, random_state: int, model_builder=None, sample_weights: pd.Series | None = None) -> tuple[Any, Any, str, str, dict[str, Any]]:
    if train_x.empty or train_y.nunique() < 2 or len(train_x) < 10:
        primary_name, primary_estimator = "DummyClassifier", DummyClassifier(strategy="most_frequent")
    else:
        if model_builder is not None:
            primary_name, primary_estimator = model_builder(random_state)
        else:
            primary_name, primary_estimator = build_primary(random_state)

    primary = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", primary_estimator)])
    fallback = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    
    # Apply sample weights if provided
    fit_kwargs = {}
    if sample_weights is not None:
        fit_kwargs["model__sample_weight"] = sample_weights.loc[train_x.index].values
    
    primary.fit(train_x, train_y, **fit_kwargs)
    try:
        fallback.fit(train_x, train_y)
        fallback_name = "LogisticRegression"
    except Exception:
        fallback = primary
        fallback_name = primary_name

    metrics = {
        "train": metric_report(primary, train_x, train_y),
        "valid": metric_report(primary, valid_x, valid_y),
    }
    return primary, fallback, primary_name, fallback_name, metrics


def main() -> None:
    ensure_dirs()
    params = model_params()
    train_path = DM_DIR / "study_train_table.csv"
    infer_path = DM_DIR / "study_infer_table.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not infer_path.exists():
        raise FileNotFoundError(infer_path)

    train = pd.read_csv(train_path)
    infer = pd.read_csv(infer_path)

    # Apply unified feature engineering (same as inference path)
    train = apply_feature_engineering(train, include_course_risk=True)
    infer = apply_feature_engineering(infer, include_course_risk=True)

    candidate_feature_cols = [
        c
        for c in train.columns
        if (
            c in collect_feature_columns(train)
            or c.startswith(("prev_", "hist_", "trend_", "volatility_", "stability_", "cross__"))
            or c.startswith(("consecutive_decline_", "dist_from_worst_", "recovery_"))
            or c.startswith(("discordance_", "workload_stress"))
            or c.startswith(("trend_", "personal_", "imbalance_", "feature_"))
            or c.startswith("course_risk_")
        )
        and c not in EXCLUDED_MODEL_FEATURES
        and not c.startswith("personal_z_")  # EXCLUDE: z-score features have leakage risk
    ]
    feature_cols, screening_report = screen_features(train, infer, candidate_feature_cols)
    core_family_map = feature_group(feature_cols, CORE_FAMILY_PREFIXES)
    behavior_family_map = feature_group(feature_cols, BEHAVIOR_FAMILY_PREFIXES)
    core_features = sorted({col for cols in core_family_map.values() for col in cols})
    temporal_features = sorted([col for col in feature_cols if infer_feature_layer(col) == "temporal"])
    interaction_features = sorted([col for col in feature_cols if infer_feature_layer(col) == "interaction"])
    behavior_features = sorted({col for cols in behavior_family_map.values() for col in cols} | set(temporal_features) | set(interaction_features))
    train = train.dropna(subset=["LABEL"])
    y = pd.to_numeric(train["LABEL"], errors="coerce").fillna(0).astype(int)
    x = train[feature_cols].apply(pd.to_numeric, errors="coerce") if feature_cols else pd.DataFrame(index=train.index)

    order = train["TERM_ID"].map(term_sort_key) if "TERM_ID" in train.columns else pd.Series(range(len(train)))
    sorted_idx = order.sort_values().index
    split_at = max(1, int(len(sorted_idx) * (1 - float(params.get("test_size", 0.2)))))
    train_idx, valid_idx = sorted_idx[:split_at], sorted_idx[split_at:]
    if len(valid_idx) == 0:
        valid_idx = train_idx

    random_state = int(params.get("random_state", 42))
    core_x = x.reindex(columns=core_features).copy() if core_features else pd.DataFrame(index=x.index)
    behavior_x = x.reindex(columns=behavior_features).copy() if behavior_features else pd.DataFrame(index=x.index)
    behavior_presence = behavior_x.notna().any(axis=1) if not behavior_x.empty else pd.Series(False, index=x.index)
    behavior_train_idx = [idx for idx in train_idx if bool(behavior_presence.loc[idx])]
    behavior_valid_idx = [idx for idx in valid_idx if bool(behavior_presence.loc[idx])]
    if not behavior_valid_idx:
        behavior_valid_idx = behavior_train_idx or list(valid_idx)

    core_available = presence(train, core_family_map.get("grade", [])) & presence(train, core_family_map.get("course", []))
    behavior_hits = pd.Series(0, index=train.index)
    for cols in behavior_family_map.values():
        behavior_hits = behavior_hits + presence(train, cols).astype(int)
    subgroup_mask = core_available & (behavior_hits >= 1)
    infer_core_available = presence(infer, core_family_map.get("grade", [])) & presence(infer, core_family_map.get("course", []))
    infer_behavior_hits = pd.Series(0, index=infer.index)
    for cols in behavior_family_map.values():
        infer_behavior_hits = infer_behavior_hits + presence(infer, cols).astype(int)
    infer_subgroup_mask = infer_core_available & (infer_behavior_hits >= 1)

    # Subgroup features: all non-core features for enhanced enrichment
    subgroup_candidate_features = [col for col in feature_cols if col not in core_features]
    subgroup_feature_source_train = train.loc[subgroup_mask].copy() if subgroup_mask.any() else train.copy()
    subgroup_feature_source_infer = infer.loc[infer_subgroup_mask].copy() if infer_subgroup_mask.any() else infer.copy()
    subgroup_features, subgroup_screening_report = screen_features(
        subgroup_feature_source_train,
        subgroup_feature_source_infer,
        subgroup_candidate_features,
    )

    # Targeted features are a subset: trend/personal/imbalance features
    subgroup_targeted_features = [col for col in subgroup_features if any(
        col.startswith(prefix) for prefix in ["trend_", "personal_", "imbalance_", "feature_", "course_risk_"]
    )]
    subgroup_x = train.reindex(columns=subgroup_features).apply(pd.to_numeric, errors="coerce") if subgroup_features else pd.DataFrame(index=train.index)
    subgroup_train_idx = [idx for idx in train_idx if bool(subgroup_mask.loc[idx])]
    subgroup_valid_idx = [idx for idx in valid_idx if bool(subgroup_mask.loc[idx])]
    if not subgroup_valid_idx:
        subgroup_valid_idx = subgroup_train_idx or list(valid_idx)

    # ========================================================================
    # Sample weighting: prioritize hard cases (single_fail, uncertain band)
    # ========================================================================
    sample_weights = pd.Series(1.0, index=train.index)
    
    # Higher weight for single_fail samples
    if "LABEL_SUBTYPE" in train.columns:
        single_fail_mask = train["LABEL_SUBTYPE"] == "single_fail"
        sample_weights[single_fail_mask] = 2.0
        
        # Lower weight for overall_low (easy positives)
        overall_low_mask = train["LABEL_SUBTYPE"] == "overall_low"
        sample_weights[overall_low_mask] = 0.7
    
    # Higher weight for uncertain band samples (predictions near 0.5)
    # Use a proxy: samples with LABEL=1 but low fail_count are harder
    if "FEATURE_GRADE_AVG_SCORE" in train.columns:
        avg_score = pd.to_numeric(train["FEATURE_GRADE_AVG_SCORE"], errors="coerce")
        # Uncertain: avg_score between 60-75 (near the pass/fail boundary)
        uncertain_mask = (avg_score >= 60) & (avg_score <= 75) & (y == 1)
        sample_weights[uncertain_mask] = sample_weights[uncertain_mask] * 1.5
    
    # ========================================================================
    # Train core model with sample weights
    # ========================================================================
    core_model, core_fallback_model, core_primary_name, core_fallback_name, core_metrics = fit_model_bundle(
        core_x.loc[train_idx],
        y.loc[train_idx],
        core_x.loc[valid_idx],
        y.loc[valid_idx],
        random_state,
        sample_weights=sample_weights,
    )
    routing_policy = resolve_policy(
        {
            "low_conf_lower": 0.35,
            "low_conf_upper": 0.65,
            "behavior_alpha": 0.15,
            "subgroup_beta": 0.20,
        }
    )

    base_oof = build_oof_scores(core_x.loc[train_idx], y.loc[train_idx], build_primary, random_state=random_state)
    base_train_score = pd.Series(index=train.index, dtype="float64")
    base_train_score.loc[train_idx] = base_oof
    base_valid_score = pd.Series(
        predict_prob(core_model, core_x.loc[valid_idx]),
        index=valid_idx,
        dtype="float64",
    )
    base_train_score.loc[valid_idx] = base_valid_score
    uncertain_mask = (base_train_score > routing_policy.low_conf_lower) & (base_train_score < routing_policy.low_conf_upper)
    behavior_corrector_model, behavior_corrector_features, behavior_metrics = train_behavior_corrector(
        train_df=train,
        feature_columns=behavior_features,
        base_oof=base_train_score.fillna(0.5),
        uncertain_mask=uncertain_mask & behavior_presence,
        random_state=random_state,
    )
    behavior_model = behavior_corrector_model
    behavior_fallback_model = behavior_corrector_model
    behavior_primary_name = "BehaviorResidualCorrector" if behavior_corrector_model is not None else None
    behavior_fallback_name = behavior_primary_name

    single_fail_feature_candidates = sorted(
        {
            col
            for col in subgroup_features
            if col.startswith("course_risk_")
            or col.startswith("trend_")
            or col.startswith("personal_")
            or col.startswith("imbalance_")
            or col.startswith("discordance_")
            or col in core_features
        }
    )
    subgroup_model, subgroup_feature_columns, subgroup_metrics = train_single_fail_expert(
        train_df=train,
        feature_columns=single_fail_feature_candidates,
        base_oof=base_train_score.fillna(0.5),
        uncertain_mask=uncertain_mask,
        random_state=random_state,
    )
    subgroup_fallback_model = subgroup_model
    subgroup_primary_name = "SingleFailExpert" if subgroup_model is not None else None
    subgroup_fallback_name = subgroup_primary_name

    family_coverage = {
        "core": coverage_by_family(train, core_family_map),
        "behavior": coverage_by_family(train, behavior_family_map),
    }
    behavior_layer_status = classify_behavior_layers(family_coverage["behavior"])
    feature_layer_summary = summarize_feature_layers(feature_cols)
    
    # Build validation report for the new base + residual + single_fail expert serving flow
    valid_frame = train.loc[valid_idx].copy()
    valid_frame["BASE_SCORE"] = base_valid_score.reindex(valid_frame.index).astype(float)
    valid_frame["STUDY_DATA_MODE"] = np.where(
        subgroup_mask.reindex(valid_frame.index).fillna(False).to_numpy(),
        "core_plus_behavior",
        "core_only",
    )
    valid_behavior_prob = np.full(len(valid_frame), np.nan)
    valid_subgroup_prob = np.full(len(valid_frame), np.nan)
    valid_uncertain_mask = (valid_frame["BASE_SCORE"] > routing_policy.low_conf_lower) & (valid_frame["BASE_SCORE"] < routing_policy.low_conf_upper)
    if behavior_corrector_model is not None:
        valid_behavior_x = valid_frame.reindex(columns=behavior_corrector_features).copy()
        if "BASE_SCORE_OOF" in valid_behavior_x.columns:
            valid_behavior_x["BASE_SCORE_OOF"] = valid_frame["BASE_SCORE"].to_numpy()
        valid_behavior_prob = np.asarray(behavior_corrector_model.predict_proba(valid_behavior_x))[:, 1]
    if subgroup_model is not None:
        valid_subgroup_x = valid_frame.reindex(columns=subgroup_feature_columns).copy()
        if "BASE_SCORE_OOF" in valid_subgroup_x.columns:
            valid_subgroup_x["BASE_SCORE_OOF"] = valid_frame["BASE_SCORE"].to_numpy()
        valid_subgroup_prob = np.asarray(subgroup_model.predict_proba(valid_subgroup_x))[:, 1]

    routed = apply_serving_policy(
        base_score=valid_frame["BASE_SCORE"],
        behavior_signal=valid_behavior_prob,
        subgroup_signal=valid_subgroup_prob,
        data_mode=valid_frame["STUDY_DATA_MODE"],
        subtype_signal=np.where(valid_frame.get("LABEL_SUBTYPE", pd.Series("", index=valid_frame.index)).astype(str).eq("single_fail"), 1.0, 0.0),
        policy=routing_policy,
    )
    valid_frame = pd.concat([valid_frame, routed], axis=1)
    dual_metrics = compute_dual_metrics(core_model, core_x.loc[valid_idx], y.loc[valid_idx], train.loc[valid_idx])
    eval_report = build_eval_reports(valid_frame[["LABEL", "LABEL_SUBTYPE", "STUDY_DATA_MODE", "BASE_SCORE", "FINAL_SCORE", "CONFIDENCE_ZONE"]].copy())
    write_eval_outputs(eval_report, EVAL_REPORT_PATH, SUBGROUP_METRICS_PATH, CONFIDENCE_ZONE_REPORT_PATH)
    if not ANCHOR_BASELINE_PATH.exists():
        write_json(
            {
                "frozen_at": datetime.now().isoformat(timespec="seconds"),
                "model_version": params.get("model_version", "study_v1"),
                "overall_metrics": eval_report.get("overall_metrics", {}),
            },
            ANCHOR_BASELINE_PATH,
        )
    
    config = {
        "domain": params.get("domain", "study"),
        "model_version": params.get("model_version", "study_v1"),
        "feature_version": params.get("feature_version", "study_feature_v1"),
        "label_name": params.get("label_name", "LABEL"),
        "id_columns": params.get("id_columns", ["XH", "TERM_ID"]),
        "feature_prefix": params.get("feature_prefix", "FEATURE_"),
        "train_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "primary_model": core_primary_name,
        "fallback_model": core_fallback_name,
        "note": "study layered model: stable core + optional behavior enhancement",
        "architecture_version": "study_layered_v2",
        "feature_columns": feature_cols,
        "core_feature_columns": core_features,
        "behavior_feature_columns": behavior_features,
        "behavior_corrector_feature_columns": behavior_corrector_features,
        "subgroup_feature_columns": subgroup_feature_columns,
        "subgroup_targeted_feature_columns": subgroup_targeted_features,
        "subgroup_feature_layer_summary": summarize_feature_layers(subgroup_features),
        "temporal_feature_columns": temporal_features,
        "interaction_feature_columns": interaction_features,
        "core_feature_families": core_family_map,
        "behavior_feature_families": behavior_family_map,
        "feature_layer_summary": feature_layer_summary,
        "behavior_layer_status": behavior_layer_status,
        "behavior_module_enabled": False,
        "behavior_corrector_enabled": bool(behavior_model is not None),
        "subgroup_expert_enabled": bool(subgroup_model is not None),
        "data_mode_rules": {
            "core_available_requires_families": ["grade", "course"],
            "behavior_available_min_family_hits": 1,
            "core_only": "core model runs and behavior module is unavailable for the row",
            "core_plus_behavior": "core model runs and behavior module has enough row-level coverage",
            "degraded_sparse": "core feature families are missing or row-level core coverage is insufficient",
        },
        "routing_policy": {
            "low_conf_lower": routing_policy.low_conf_lower,
            "low_conf_upper": routing_policy.low_conf_upper,
            "behavior_alpha": routing_policy.behavior_alpha,
            "subgroup_beta": routing_policy.subgroup_beta,
        },
        "confidence_routing": {
            "enabled": True,
            "low_threshold": routing_policy.low_conf_lower,
            "high_threshold": routing_policy.low_conf_upper,
            "uncertain_strategy": "residual_then_expert",
        },
        "anchor_baseline_path": str(ANCHOR_BASELINE_PATH),
        "eval_report_path": str(EVAL_REPORT_PATH),
        "subgroup_metrics_path": str(SUBGROUP_METRICS_PATH),
        "confidence_zone_report_path": str(CONFIDENCE_ZONE_REPORT_PATH),
        "metrics": {
            "core_model": core_metrics,
            "behavior_module": behavior_metrics,
            "subgroup_expert": subgroup_metrics,
            "legacy_primary_valid": core_metrics.get("valid", {}),
            "dual_validation": dual_metrics,  # Overall + subgroup breakdown
            "serving_eval_report": eval_report,
        },
    }
    bundle = {
        "primary_model": core_model,
        "fallback_model": core_fallback_model,
        "core_model": core_model,
        "core_fallback_model": core_fallback_model,
        "behavior_model": behavior_model,
        "behavior_fallback_model": behavior_fallback_model,
        "subgroup_model": subgroup_model,
        "subgroup_fallback_model": subgroup_fallback_model,
        "base_oof_train_score": base_train_score.fillna(0.5),
        "config": config,
    }
    joblib.dump(bundle, DM_DIR / "study_model.pkl")
    write_json(config, DM_DIR / "study_model_config.json")
    write_json(config["metrics"], DM_DIR / "study_model_metrics.json")
    write_json(feature_layer_summary | {"generated_at": datetime.now().isoformat(timespec="seconds"), "scope": "layered_train"}, FEATURE_LAYER_SUMMARY_PATH)
    write_json(screening_report, FEATURE_SCREENING_REPORT_PATH)
    write_json(subgroup_screening_report, SUBGROUP_FEATURE_SCREENING_REPORT_PATH)
    pd.DataFrame(
        {
            "feature_name": feature_cols,
            "feature_layer": [infer_feature_layer(name) for name in feature_cols],
        }
    ).to_csv(SELECTED_FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(
        f"model written: {DM_DIR / 'study_model.pkl'} "
        f"core_features={len(core_features)} behavior_features={len(behavior_features)} "
        f"temporal_features={len(temporal_features)} interaction_features={len(interaction_features)} "
        f"subgroup_features={len(subgroup_features)}"
    )


if __name__ == "__main__":
    main()
