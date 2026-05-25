from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from study_common import term_sort_key
from study_expert_train import build_oof_scores, train_behavior_corrector, train_single_fail_expert
from study_feature_engine import apply_feature_engineering
from study_routing_policy import apply_serving_policy, resolve_policy

ROOT = Path(__file__).resolve().parents[1]
DM_DIR = ROOT / "data" / "dm"
MODEL_CONFIG_PATH = ROOT / "data" / "deliverables" / "study" / "model" / "study_model_config.json"
TRAIN_PATH = DM_DIR / "study_train_table.csv"
OUTPUT_PATH = DM_DIR / "study_auc_drop_attribution.json"
CSV_PATH = DM_DIR / "study_auc_drop_attribution.csv"
EXP_A_TRAIN_IDS_PATH = DM_DIR / "exp_A_baseline_rebuild_train_ids.csv"
EXP_A_HOLDOUT_IDS_PATH = DM_DIR / "exp_A_baseline_rebuild_holdout_ids.csv"

CORE8_FEATURES = [
    "FEATURE_GRADE_COURSE_COUNT",
    "FEATURE_GRADE_AVG_SCORE",
    "FEATURE_GRADE_MIN_SCORE",
    "FEATURE_GRADE_FAIL_COUNT",
    "FEATURE_GRADE_CREDIT_SUM",
    "FEATURE_COURSE_SELECTED_COUNT",
    "FEATURE_COURSE_CREDIT_SUM",
    "FEATURE_COURSE_RETAKE_COUNT",
]

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


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def feature_group(columns: list[str], groups: dict[str, tuple[str, ...]]) -> dict[str, list[str]]:
    return {name: [col for col in columns if any(col.startswith(prefix) for prefix in prefixes)] for name, prefixes in groups.items()}


def presence(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    usable = [c for c in cols if c in df.columns]
    return df[usable].notna().any(axis=1) if usable else pd.Series(False, index=df.index)


def safe_auc(y_true: pd.Series, score: np.ndarray) -> float | None:
    if len(y_true) == 0 or y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, score))


def build_lgbm_pipeline() -> tuple[str, dict[str, Any], Pipeline]:
    try:
        from lightgbm import LGBMClassifier

        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 20,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "scale_pos_weight": 1.0,
            "random_state": 42,
            "verbose": -1,
        }
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LGBMClassifier(**params)),
            ]
        )
        return "LightGBMClassifier", params, pipe
    except Exception:
        params = {"max_iter": 1000, "random_state": 42}
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LogisticRegression(**params)),
            ]
        )
        return "LogisticRegression", params, pipe


def lgbm_builder(random_state: int) -> tuple[str, Any]:
    try:
        from lightgbm import LGBMClassifier

        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "num_leaves": 15,
            "min_child_samples": 20,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "scale_pos_weight": 1.0,
            "random_state": random_state,
            "verbose": -1,
        }
        return "LightGBMClassifier", LGBMClassifier(**params)
    except Exception:
        return "LogisticRegression", LogisticRegression(max_iter=1000, random_state=random_state)


def fit_and_score(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, Any]:
    x_train = train_df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
    x_valid = valid_df.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df["LABEL"], errors="coerce").fillna(0).astype(int)
    y_valid = pd.to_numeric(valid_df["LABEL"], errors="coerce").fillna(0).astype(int)
    model_name, model_params, pipeline = build_lgbm_pipeline()
    pipeline.fit(x_train, y_train)
    score = pipeline.predict_proba(x_valid)[:, 1]
    return {
        "auc": safe_auc(y_valid, score),
        "model_name": model_name,
        "model_params": model_params,
        "feature_count": len(feature_cols),
        "valid_rows": int(len(valid_df)),
        "score": score,
    }


def random_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = pd.to_numeric(df["LABEL"], errors="coerce").fillna(0).astype(int)
    idx_train, idx_valid = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
    return df.loc[idx_train].copy(), df.loc[idx_valid].copy()


def split_by_frozen_ids(df: pd.DataFrame, train_ids_path: Path, holdout_ids_path: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not train_ids_path.exists() or not holdout_ids_path.exists():
        return None
    key_cols = ["XH", "TERM_ID"]
    train_ids = pd.read_csv(train_ids_path)
    holdout_ids = pd.read_csv(holdout_ids_path)
    train_keys = set(train_ids[key_cols].astype(str).agg("||".join, axis=1).tolist())
    holdout_keys = set(holdout_ids[key_cols].astype(str).agg("||".join, axis=1).tolist())
    keyed = df.copy()
    keyed["_split_key"] = keyed[key_cols].astype(str).agg("||".join, axis=1)
    train_df = keyed.loc[keyed["_split_key"].isin(train_keys)].drop(columns=["_split_key"]).copy()
    valid_df = keyed.loc[keyed["_split_key"].isin(holdout_keys)].drop(columns=["_split_key"]).copy()
    if train_df.empty or valid_df.empty:
        return None
    return train_df, valid_df


def term_order_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = df["TERM_ID"].map(term_sort_key) if "TERM_ID" in df.columns else pd.Series(range(len(df)), index=df.index)
    sorted_idx = order.sort_values().index
    split_at = max(1, int(len(sorted_idx) * 0.8))
    train_idx, valid_idx = sorted_idx[:split_at], sorted_idx[split_at:]
    if len(valid_idx) == 0:
        valid_idx = train_idx
    return df.loc[train_idx].copy(), df.loc[valid_idx].copy()


def assign_modes(frame: pd.DataFrame) -> pd.Series:
    columns = list(frame.columns)
    core_family_map = feature_group(columns, CORE_FAMILY_PREFIXES)
    behavior_family_map = feature_group(columns, BEHAVIOR_FAMILY_PREFIXES)
    core_available = presence(frame, core_family_map.get("grade", [])) & presence(frame, core_family_map.get("course", []))
    behavior_hits = pd.Series(0, index=frame.index)
    for cols in behavior_family_map.values():
        behavior_hits = behavior_hits + presence(frame, cols).astype(int)
    mode = pd.Series("degraded_sparse", index=frame.index, dtype="string")
    mode.loc[core_available] = "core_only"
    mode.loc[core_available & (behavior_hits >= 1)] = "core_plus_behavior"
    return mode


def summarize_population(df: pd.DataFrame, mode: pd.Series | None = None) -> dict[str, Any]:
    summary = {"rows": int(len(df))}
    if "LABEL" in df.columns:
        y = pd.to_numeric(df["LABEL"], errors="coerce").fillna(0).astype(int)
        summary["positive_rows"] = int(y.sum())
        summary["negative_rows"] = int((y == 0).sum())
    if mode is not None:
        summary["mode_counts"] = mode.astype(str).value_counts(dropna=False).to_dict()
    return summary


def run_routed_experiment(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    core_features = [c for c in model_config.get("core_feature_columns", []) if c in train_df.columns or c in valid_df.columns]
    behavior_corrector_features = model_config.get("behavior_corrector_feature_columns", [])
    subgroup_features = model_config.get("subgroup_feature_columns", [])
    policy = resolve_policy(
        {
            "low_conf_lower": 0.35,
            "low_conf_upper": 0.65,
            "behavior_alpha": 0.15,
            "subgroup_beta": 0.20,
        }
    )

    x_train_core = train_df.reindex(columns=core_features).apply(pd.to_numeric, errors="coerce")
    x_valid_core = valid_df.reindex(columns=core_features).apply(pd.to_numeric, errors="coerce")
    y_train = pd.to_numeric(train_df["LABEL"], errors="coerce").fillna(0).astype(int)
    y_valid = pd.to_numeric(valid_df["LABEL"], errors="coerce").fillna(0).astype(int)

    core_model_name, core_model_params, core_model = build_lgbm_pipeline()
    core_model.fit(x_train_core, y_train)
    base_valid_score = core_model.predict_proba(x_valid_core)[:, 1]

    base_oof = build_oof_scores(x_train_core, y_train, lgbm_builder, random_state=42, n_splits=5)
    uncertain_mask = (base_oof > policy.low_conf_lower) & (base_oof < policy.low_conf_upper)

    behavior_model, behavior_cols, behavior_metrics = train_behavior_corrector(
        train_df=train_df,
        feature_columns=[c for c in behavior_corrector_features if c in train_df.columns or c == "BASE_SCORE_OOF"],
        base_oof=base_oof.fillna(0.5),
        uncertain_mask=uncertain_mask,
        random_state=42,
    )
    subgroup_model, subgroup_cols, subgroup_metrics = train_single_fail_expert(
        train_df=train_df,
        feature_columns=[c for c in subgroup_features if c in train_df.columns or c == "BASE_SCORE_OOF"],
        base_oof=base_oof.fillna(0.5),
        uncertain_mask=uncertain_mask,
        random_state=42,
    )

    valid_mode = assign_modes(valid_df)
    behavior_prob = np.full(len(valid_df), np.nan)
    subgroup_prob = np.full(len(valid_df), np.nan)

    if behavior_model is not None:
        behavior_input = valid_df.reindex(columns=behavior_cols).copy()
        if "BASE_SCORE_OOF" in behavior_input.columns:
            behavior_input["BASE_SCORE_OOF"] = base_valid_score
        behavior_prob = np.asarray(behavior_model.predict_proba(behavior_input))[:, 1]

    if subgroup_model is not None:
        subgroup_input = valid_df.reindex(columns=subgroup_cols).copy()
        if "BASE_SCORE_OOF" in subgroup_input.columns:
            subgroup_input["BASE_SCORE_OOF"] = base_valid_score
        subgroup_prob = np.asarray(subgroup_model.predict_proba(subgroup_input))[:, 1]

    subtype_signal = np.where(valid_df.get("LABEL_SUBTYPE", pd.Series("", index=valid_df.index)).astype(str).eq("single_fail"), 1.0, 0.0)
    routed = apply_serving_policy(
        base_score=pd.Series(base_valid_score, index=valid_df.index),
        behavior_signal=behavior_prob,
        subgroup_signal=subgroup_prob,
        data_mode=valid_mode,
        policy=policy,
        subtype_signal=subtype_signal,
    )
    final_score = routed["FINAL_SCORE"].to_numpy()

    return {
        "auc": safe_auc(y_valid, final_score),
        "core_model_name": core_model_name,
        "core_model_params": core_model_params,
        "base_auc_before_routing": safe_auc(y_valid, base_valid_score),
        "behavior_model_trained": behavior_model is not None,
        "subgroup_model_trained": subgroup_model is not None,
        "behavior_metrics": behavior_metrics,
        "subgroup_metrics": subgroup_metrics,
        "confidence_zone_counts": routed["CONFIDENCE_ZONE"].value_counts(dropna=False).to_dict(),
        "routing_reason_counts": routed["ROUTING_REASON"].value_counts(dropna=False).to_dict(),
        "valid_mode_counts": valid_mode.astype(str).value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    train_raw = pd.read_csv(TRAIN_PATH)
    model_config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8")) if MODEL_CONFIG_PATH.exists() else {}
    engineered = apply_feature_engineering(train_raw, include_course_risk=True)

    label_mask = pd.to_numeric(engineered["LABEL"], errors="coerce").notna()
    engineered = engineered.loc[label_mask].copy()
    raw_labeled = train_raw.loc[engineered.index].copy()

    behavior_features_raw = [
        c
        for c in raw_labeled.columns
        if any(x in c.lower() for x in ["attendance", "library", "assignment", "exam", "class_task"])
        and c not in CORE8_FEATURES
    ]
    behavior_features_raw = [c for c in behavior_features_raw if raw_labeled[c].notna().mean() > 0.1]
    behavior_nonnull = raw_labeled[behavior_features_raw].notna().sum(axis=1) if behavior_features_raw else pd.Series(0, index=raw_labeled.index)
    core_only_mask = behavior_nonnull < 2

    l0_df = raw_labeled.loc[core_only_mask].copy()
    frozen_split = split_by_frozen_ids(l0_df, EXP_A_TRAIN_IDS_PATH, EXP_A_HOLDOUT_IDS_PATH)
    l0_train, l0_valid = frozen_split if frozen_split is not None else random_split(l0_df)
    l0 = fit_and_score(l0_train, l0_valid, CORE8_FEATURES)

    l1_df = raw_labeled.loc[core_only_mask].copy()
    l1_train, l1_valid = term_order_split(l1_df)
    l1 = fit_and_score(l1_train, l1_valid, CORE8_FEATURES)

    l2_df = engineered.copy()
    l2_mode = assign_modes(l2_df)
    l2_train, l2_valid = term_order_split(l2_df)
    l2 = fit_and_score(l2_train, l2_valid, CORE8_FEATURES)

    l3_features = [c for c in model_config.get("feature_columns", []) if c in engineered.columns]
    l3_train, l3_valid = term_order_split(l2_df)
    l3 = fit_and_score(l3_train, l3_valid, l3_features)

    l4_train, l4_valid = term_order_split(l2_df)
    l4 = run_routed_experiment(l4_train, l4_valid, model_config)

    rows = []
    report = {
        "experiment_name": "study_auc_drop_attribution",
        "generated_at": now_iso(),
        "label_definition": "LABEL",
        "reference": {"historical_core_only_random_auc": 0.852089926589149},
        "levels": {},
    }

    def add_level(name: str, payload: dict[str, Any], *, population_summary: dict[str, Any], split: str, features: str) -> None:
        report["levels"][name] = {
            "split": split,
            "features": features,
            "population_summary": population_summary,
            **{k: v for k, v in payload.items() if k != "score"},
        }
        rows.append(
            {
                "level": name,
                "split": split,
                "features": features,
                "rows": population_summary.get("rows"),
                "auc": payload.get("auc"),
            }
        )

    add_level("L0", l0, population_summary=summarize_population(l0_df), split="random_stratified", features="8_core")
    add_level("L1", l1, population_summary=summarize_population(l1_df), split="term_order_holdout", features="8_core")
    add_level("L2", l2, population_summary=summarize_population(l2_df, l2_mode), split="term_order_holdout", features="8_core")
    add_level("L3", l3, population_summary=summarize_population(l2_df, l2_mode), split="term_order_holdout", features="current_serving_features_no_routing")
    add_level("L4", l4, population_summary=summarize_population(l2_df, l2_mode), split="term_order_holdout", features="current_layered_routed_serving")

    ordered = ["L0", "L1", "L2", "L3", "L4"]
    for idx, level in enumerate(ordered):
        current_auc = report["levels"][level].get("auc")
        report["levels"][level]["delta_vs_L0"] = None if current_auc is None else float(current_auc - report["levels"]["L0"]["auc"])
        if idx == 0:
            report["levels"][level]["delta_vs_prev"] = None
        else:
            prev_auc = report["levels"][ordered[idx - 1]].get("auc")
            report["levels"][level]["delta_vs_prev"] = None if current_auc is None or prev_auc is None else float(current_auc - prev_auc)

    interpretation = []
    if report["levels"]["L1"]["auc"] is not None and report["levels"]["L0"]["auc"] is not None:
        interpretation.append(
            {
                "question": "split_difficulty_effect",
                "answer": float(report["levels"]["L1"]["auc"] - report["levels"]["L0"]["auc"]),
            }
        )
    if report["levels"]["L2"]["auc"] is not None and report["levels"]["L1"]["auc"] is not None:
        interpretation.append(
            {
                "question": "population_expansion_effect",
                "answer": float(report["levels"]["L2"]["auc"] - report["levels"]["L1"]["auc"]),
            }
        )
    if report["levels"]["L3"]["auc"] is not None and report["levels"]["L2"]["auc"] is not None:
        interpretation.append(
            {
                "question": "feature_expansion_effect",
                "answer": float(report["levels"]["L3"]["auc"] - report["levels"]["L2"]["auc"]),
            }
        )
    if report["levels"]["L4"]["auc"] is not None and report["levels"]["L3"]["auc"] is not None:
        interpretation.append(
            {
                "question": "routing_effect",
                "answer": float(report["levels"]["L4"]["auc"] - report["levels"]["L3"]["auc"]),
            }
        )
    report["interpretation_deltas"] = interpretation

    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
