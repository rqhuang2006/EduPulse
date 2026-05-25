"""
Study Domain AUC Improvement Experiments
=========================================
Three focused experiments to validate AUC improvement directions:

1. single_fail专项实验: Validate hardest subgroup can be modeled separately
2. 课程级临界风险特征: Course-level critical risk features vs generic temporal
3. enhanced组独立主模型: Separate core_only and core_plus_behavior models
"""
from __future__ import annotations

import json
import warnings
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

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

ROOT = Path(__file__).resolve().parents[1]
DM_DIR = ROOT / "data" / "dm"
EXPERIMENT_RESULTS_PATH = DM_DIR / "study_auc_experiments.json"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_auc(y_true, y_pred):
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return 0.5


def compute_metrics(y_true, y_pred):
    return {
        "auc": safe_auc(y_true, y_pred),
        "rows": len(y_true),
        "positive_count": int(y_true.sum()),
        "negative_count": int((~y_true).sum()),
    }


def train_lgbm_pipeline(X_train, y_train, X_valid, y_valid, **kwargs):
    default_params = {
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
    default_params.update(kwargs)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LGBMClassifier(**default_params)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict_proba(X_valid)[:, 1]
    return pipe, pred


# ============================================================================
# Experiment 1: single_fail专项实验
# ============================================================================
def experiment_single_fail_specialist(train_df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate: Can we model single_fail subgroup better when separated?

    Approach:
    - Filter to normal + single_fail only (exclude overall_low)
    - Train separate models for each subtype
    - Compare AUC vs mixed-label baseline
    """
    print("\n" + "=" * 60)
    print("Experiment 1: single_fail Specialist")
    print("=" * 60)

    if "LABEL_SUBTYPE" not in train_df.columns:
        print("  SKIP: LABEL_SUBTYPE not found in training data")
        return {"status": "skipped", "reason": "LABEL_SUBTYPE missing"}

    # Core features only for this experiment
    core_features = [c for c in train_df.columns if c.startswith("FEATURE_GRADE_") or c.startswith("FEATURE_COURSE_")]
    core_features = [c for c in core_features if train_df[c].notna().mean() > 0.5]

    # Baseline: mixed label (all positives)
    mixed_df = train_df[train_df["LABEL"].isin([0, 1])].copy()
    mixed_df = mixed_df.dropna(subset=core_features)
    y_mixed = mixed_df["LABEL"].astype(int)
    X_mixed = mixed_df[core_features]

    idx_train, idx_valid = train_test_split(X_mixed.index, test_size=0.2, random_state=42, stratify=y_mixed)
    X_train, X_valid = X_mixed.loc[idx_train], X_mixed.loc[idx_valid]
    y_train, y_valid = y_mixed.loc[idx_train], y_mixed.loc[idx_valid]

    if LGBMClassifier is not None:
        _, mixed_pred = train_lgbm_pipeline(X_train, y_train, X_valid, y_valid)
    else:
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, random_state=42))])
        pipe.fit(X_train, y_train)
        mixed_pred = pipe.predict_proba(X_valid)[:, 1]

    mixed_metrics = compute_metrics(y_valid.values, mixed_pred)
    print(f"  Baseline (mixed): AUC={mixed_metrics['auc']:.4f}, pos={mixed_metrics['positive_count']}")

    # Experiment: single_fail only
    sf_df = train_df[train_df["LABEL_SUBTYPE"].isin(["normal", "single_fail"])].copy()
    sf_df = sf_df.dropna(subset=core_features)
    y_sf = (sf_df["LABEL_SUBTYPE"] == "single_fail").astype(int)
    X_sf = sf_df[core_features]

    idx_train, idx_valid = train_test_split(X_sf.index, test_size=0.2, random_state=42, stratify=y_sf)
    X_train, X_valid = X_sf.loc[idx_train], X_sf.loc[idx_valid]
    y_train, y_valid = y_sf.loc[idx_train], y_sf.loc[idx_valid]

    if LGBMClassifier is not None:
        _, sf_pred = train_lgbm_pipeline(X_train, y_train, X_valid, y_valid, scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1))
    else:
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))])
        pipe.fit(X_train, y_train)
        sf_pred = pipe.predict_proba(X_valid)[:, 1]

    sf_metrics = compute_metrics(y_valid.values, sf_pred)
    print(f"  single_fail specialist: AUC={sf_metrics['auc']:.4f}, pos={sf_metrics['positive_count']}")

    # Experiment: overall_low only
    ol_df = train_df[train_df["LABEL_SUBTYPE"].isin(["normal", "overall_low"])].copy()
    ol_df = ol_df.dropna(subset=core_features)
    y_ol = (ol_df["LABEL_SUBTYPE"] == "overall_low").astype(int)
    X_ol = ol_df[core_features]

    if len(y_ol) > 10 and y_ol.nunique() == 2:
        idx_train, idx_valid = train_test_split(X_ol.index, test_size=0.2, random_state=42, stratify=y_ol)
        X_train, X_valid = X_ol.loc[idx_train], X_ol.loc[idx_valid]
        y_train, y_valid = y_ol.loc[idx_train], y_ol.loc[idx_valid]

        if LGBMClassifier is not None:
            _, ol_pred = train_lgbm_pipeline(X_train, y_train, X_valid, y_valid, scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1))
        else:
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))])
            pipe.fit(X_train, y_train)
            ol_pred = pipe.predict_proba(X_valid)[:, 1]

        ol_metrics = compute_metrics(y_valid.values, ol_pred)
    else:
        ol_metrics = {"auc": 0.0, "rows": 0, "note": "insufficient data"}

    print(f"  overall_low specialist: AUC={ol_metrics['auc']:.4f}, pos={ol_metrics.get('positive_count', 'N/A')}")

    return {
        "status": "completed",
        "baseline_mixed": mixed_metrics,
        "single_fail_specialist": sf_metrics,
        "overall_low_specialist": ol_metrics,
        "conclusion": "single_fail specialist AUC improvement vs baseline",
    }


# ============================================================================
# Experiment 2: 课程级临界风险特征
# ============================================================================
def add_course_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add course-level critical risk features:
    - min_course_score: worst course score
    - second_min_course_score: second worst
    - fail_course_count: courses with score < 60
    - near_fail_course_count: courses with score 60-65
    - score_dispersion: std of course scores
    - recent_worst_course_delta: change in worst course score
    """
    frame = df.copy()

    # These features would normally come from course-level data
    # For this experiment, we derive them from existing grade features
    if "FEATURE_GRADE_MIN_SCORE" in frame.columns:
        frame["course_risk_min_score"] = pd.to_numeric(frame["FEATURE_GRADE_MIN_SCORE"], errors="coerce")

    if "FEATURE_GRADE_AVG_SCORE" in frame.columns and "FEATURE_GRADE_MIN_SCORE" in frame.columns:
        avg = pd.to_numeric(frame["FEATURE_GRADE_AVG_SCORE"], errors="coerce")
        min_s = pd.to_numeric(frame["FEATURE_GRADE_MIN_SCORE"], errors="coerce")
        frame["course_risk_avg_min_gap"] = avg - min_s

    if "FEATURE_GRADE_FAIL_COUNT" in frame.columns:
        frame["course_risk_fail_count"] = pd.to_numeric(frame["FEATURE_GRADE_FAIL_COUNT"], errors="coerce")

    # Distance to 60 (critical threshold)
    if "FEATURE_GRADE_AVG_SCORE" in frame.columns:
        frame["course_risk_distance_to_60"] = pd.to_numeric(frame["FEATURE_GRADE_AVG_SCORE"], errors="coerce") - 60

    # Near-fail indicator (60-65 range)
    if "FEATURE_GRADE_MIN_SCORE" in frame.columns:
        min_s = pd.to_numeric(frame["FEATURE_GRADE_MIN_SCORE"], errors="coerce")
        frame["course_risk_near_fail"] = ((min_s >= 60) & (min_s <= 65)).astype(int)

    # Score dispersion (if we have multiple score features)
    score_cols = [c for c in frame.columns if "SCORE" in c and c.startswith("FEATURE_")]
    if len(score_cols) >= 2:
        score_matrix = pd.to_numeric(frame[score_cols[0]], errors="coerce")
        frame["course_risk_dispersion"] = score_matrix.std() if len(score_matrix) > 1 else 0.0

    return frame


def experiment_course_risk_features(train_df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate: Do course-level critical risk features outperform generic temporal features?

    Approach:
    - Compare: core only vs core+old_temporal vs core+course_risk vs core+course_risk+selected_behavior
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Course-Level Critical Risk Features")
    print("=" * 60)

    # Add course risk features
    enhanced_df = add_course_risk_features(train_df)

    core_features = [c for c in enhanced_df.columns if c.startswith("FEATURE_GRADE_") or c.startswith("FEATURE_COURSE_")]
    core_features = [c for c in core_features if enhanced_df[c].notna().mean() > 0.5]

    course_risk_features = [c for c in enhanced_df.columns if c.startswith("course_risk_")]
    course_risk_features = [c for c in course_risk_features if enhanced_df[c].notna().mean() > 0.1]

    temporal_features = [c for c in enhanced_df.columns if c.startswith(("prev_", "hist_", "delta_", "ratio_"))]
    temporal_features = [c for c in temporal_features if enhanced_df[c].notna().mean() > 0.1]

    print(f"  Core features: {len(core_features)}")
    print(f"  Course risk features: {len(course_risk_features)}: {course_risk_features}")
    print(f"  Temporal features: {len(temporal_features)}")

    y = enhanced_df["LABEL"].astype(int)
    valid_mask = y.notna() & enhanced_df[core_features].notna().any(axis=1)
    y = y[valid_mask]
    X_base = enhanced_df.loc[valid_mask, core_features]

    if y.nunique() < 2:
        print("  SKIP: insufficient label classes")
        return {"status": "skipped", "reason": "insufficient labels"}

    idx_train, idx_valid = train_test_split(X_base.index, test_size=0.2, random_state=42, stratify=y)
    y_train, y_valid = y.loc[idx_train], y.loc[idx_valid]

    results = {}

    # Model 1: Core only
    if LGBMClassifier is not None:
        _, pred_core = train_lgbm_pipeline(X_base.loc[idx_train], y_train, X_base.loc[idx_valid], y_valid)
    else:
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, random_state=42))])
        pipe.fit(X_base.loc[idx_train], y_train)
        pred_core = pipe.predict_proba(X_base.loc[idx_valid])[:, 1]

    results["core_only"] = compute_metrics(y_valid.values, pred_core)
    print(f"  Core only: AUC={results['core_only']['auc']:.4f}")

    # Model 2: Core + temporal
    if temporal_features:
        X_temporal = enhanced_df.loc[valid_mask, core_features + temporal_features]
        if LGBMClassifier is not None:
            _, pred_temporal = train_lgbm_pipeline(X_temporal.loc[idx_train], y_train, X_temporal.loc[idx_valid], y_valid)
        else:
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, random_state=42))])
            pipe.fit(X_temporal.loc[idx_train], y_train)
            pred_temporal = pipe.predict_proba(X_temporal.loc[idx_valid])[:, 1]
        results["core_plus_temporal"] = compute_metrics(y_valid.values, pred_temporal)
        print(f"  Core + temporal: AUC={results['core_plus_temporal']['auc']:.4f}")

    # Model 3: Core + course_risk
    if course_risk_features:
        X_course = enhanced_df.loc[valid_mask, core_features + course_risk_features]
        if LGBMClassifier is not None:
            _, pred_course = train_lgbm_pipeline(X_course.loc[idx_train], y_train, X_course.loc[idx_valid], y_valid)
        else:
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, random_state=42))])
            pipe.fit(X_course.loc[idx_train], y_train)
            pred_course = pipe.predict_proba(X_course.loc[idx_valid])[:, 1]
        results["core_plus_course_risk"] = compute_metrics(y_valid.values, pred_course)
        print(f"  Core + course_risk: AUC={results['core_plus_course_risk']['auc']:.4f}")

    return {
        "status": "completed",
        "results": results,
        "feature_counts": {
            "core": len(core_features),
            "course_risk": len(course_risk_features),
            "temporal": len(temporal_features),
        },
    }


# ============================================================================
# Experiment 3: enhanced组独立主模型
# ============================================================================
def experiment_enhanced_separate_model(train_df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate: Should core_only and core_plus_behavior be modeled separately?

    Approach:
    - Split samples by data mode (core_only vs core_plus_behavior)
    - Train separate models for each group
    - Compare AUC vs unified model
    """
    print("\n" + "=" * 60)
    print("Experiment 3: Enhanced Group Independent Model")
    print("=" * 60)

    core_features = [c for c in train_df.columns if c.startswith("FEATURE_GRADE_") or c.startswith("FEATURE_COURSE_")]
    core_features = [c for c in core_features if train_df[c].notna().mean() > 0.5]

    behavior_features = [c for c in train_df.columns if any(x in c.lower() for x in ["attendance", "library", "assignment", "exam", "class_task"])]
    behavior_features = [c for c in behavior_features if c not in core_features and train_df[c].notna().mean() > 0.1]

    print(f"  Core features: {len(core_features)}")
    print(f"  Behavior features: {len(behavior_features)}")

    y = train_df["LABEL"].astype(int)
    valid_mask = y.notna()

    # Determine data mode by behavior feature availability
    behavior_nonnull = train_df[behavior_features].notna().sum(axis=1) if behavior_features else pd.Series(0, index=train_df.index)
    data_mode = np.where(behavior_nonnull >= 2, "core_plus_behavior", "core_only")

    results = {}

    # Unified model (baseline)
    valid_idx = train_df.index[valid_mask & train_df[core_features].notna().any(axis=1)]
    y_all = y.loc[valid_idx]
    X_all = train_df.loc[valid_idx, core_features]

    if y_all.nunique() < 2:
        print("  SKIP: insufficient label classes")
        return {"status": "skipped", "reason": "insufficient labels"}

    idx_train, idx_valid = train_test_split(X_all.index, test_size=0.2, random_state=42, stratify=y_all)
    y_train, y_valid = y_all.loc[idx_train], y_all.loc[idx_valid]

    if LGBMClassifier is not None:
        _, pred_unified = train_lgbm_pipeline(X_all.loc[idx_train], y_train, X_all.loc[idx_valid], y_valid)
    else:
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, random_state=42))])
        pipe.fit(X_all.loc[idx_train], y_train)
        pred_unified = pipe.predict_proba(X_all.loc[idx_valid])[:, 1]

    results["unified_model"] = compute_metrics(y_valid.values, pred_unified)
    print(f"  Unified model: AUC={results['unified_model']['auc']:.4f}")

    # Separate models
    core_only_idx = valid_idx[data_mode[valid_idx] == "core_only"]
    enhanced_idx = valid_idx[data_mode[valid_idx] == "core_plus_behavior"]

    print(f"  Core_only samples: {len(core_only_idx)}")
    print(f"  Enhanced samples: {len(enhanced_idx)}")

    # Core_only model
    if len(core_only_idx) > 100:
        y_co = y.loc[core_only_idx]
        X_co = train_df.loc[core_only_idx, core_features]
        idx_train, idx_valid = train_test_split(X_co.index, test_size=0.2, random_state=42, stratify=y_co)
        y_train, y_valid = y_co.loc[idx_train], y_co.loc[idx_valid]

        if LGBMClassifier is not None:
            _, pred_co = train_lgbm_pipeline(X_co.loc[idx_train], y_train, X_co.loc[idx_valid], y_valid)
        else:
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, random_state=42))])
            pipe.fit(X_co.loc[idx_train], y_train)
            pred_co = pipe.predict_proba(X_co.loc[idx_valid])[:, 1]

        results["core_only_model"] = compute_metrics(y_valid.values, pred_co)
        print(f"  Core_only model: AUC={results['core_only_model']['auc']:.4f}")

    # Enhanced model (with behavior features)
    if len(enhanced_idx) > 100:
        y_en = y.loc[enhanced_idx]
        X_en = train_df.loc[enhanced_idx, core_features + behavior_features]
        X_en = X_en.fillna(0)

        if y_en.nunique() >= 2:
            idx_train, idx_valid = train_test_split(X_en.index, test_size=0.2, random_state=42, stratify=y_en)
            y_train, y_valid = y_en.loc[idx_train], y_en.loc[idx_valid]

            if LGBMClassifier is not None:
                _, pred_en = train_lgbm_pipeline(X_en.loc[idx_train], y_train, X_en.loc[idx_valid], y_valid)
            else:
                pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", LogisticRegression(max_iter=1000, random_state=42))])
                pipe.fit(X_en.loc[idx_train], y_train)
                pred_en = pipe.predict_proba(X_en.loc[idx_valid])[:, 1]

            results["enhanced_model"] = compute_metrics(y_valid.values, pred_en)
            print(f"  Enhanced model: AUC={results['enhanced_model']['auc']:.4f}")

    return {
        "status": "completed",
        "results": results,
        "sample_counts": {
            "core_only": int((data_mode == "core_only").sum()),
            "core_plus_behavior": int((data_mode == "core_plus_behavior").sum()),
        },
    }


# ============================================================================
# Main
# ============================================================================
def main():
    print("Study Domain AUC Improvement Experiments")
    print(f"Started at: {now_iso()}")

    # Load training data
    train_path = DM_DIR / "study_train_table.csv"
    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        print("Run 20_build_label.py and 21_build_study_train_infer.py first.")
        return

    train_df = pd.read_csv(train_path)
    print(f"Loaded training data: {len(train_df)} rows")
    print(f"Label distribution: {train_df['LABEL'].value_counts().to_dict()}")
    if "LABEL_SUBTYPE" in train_df.columns:
        print(f"Label subtype: {train_df['LABEL_SUBTYPE'].value_counts().to_dict()}")

    # Run experiments
    results = {
        "timestamp": now_iso(),
        "data_rows": len(train_df),
        "experiments": {},
    }

    # Experiment 1: single_fail specialist
    exp1 = experiment_single_fail_specialist(train_df)
    results["experiments"]["single_fail_specialist"] = exp1

    # Experiment 2: Course-level critical risk features
    exp2 = experiment_course_risk_features(train_df)
    results["experiments"]["course_risk_features"] = exp2

    # Experiment 3: Enhanced group independent model
    exp3 = experiment_enhanced_separate_model(train_df)
    results["experiments"]["enhanced_separate_model"] = exp3

    # Write results
    EXPERIMENT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENT_RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nExperiment results written to {EXPERIMENT_RESULTS_PATH}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    exp1_data = results["experiments"].get("single_fail_specialist", {})
    if exp1_data.get("status") == "completed":
        baseline = exp1_data.get("baseline_mixed", {}).get("auc", 0)
        sf = exp1_data.get("single_fail_specialist", {}).get("auc", 0)
        ol = exp1_data.get("overall_low_specialist", {}).get("auc", 0)
        print(f"\n1. single_fail Specialist:")
        print(f"   Baseline (mixed): {baseline:.4f}")
        print(f"   single_fail only: {sf:.4f}")
        print(f"   overall_low only: {ol:.4f}")

    exp2_data = results["experiments"].get("course_risk_features", {})
    if exp2_data.get("status") == "completed":
        print(f"\n2. Course Risk Features:")
        for key, val in exp2_data.get("results", {}).items():
            print(f"   {key}: AUC={val.get('auc', 0):.4f}")

    exp3_data = results["experiments"].get("enhanced_separate_model", {})
    if exp3_data.get("status") == "completed":
        print(f"\n3. Enhanced Separate Model:")
        for key, val in exp3_data.get("results", {}).items():
            print(f"   {key}: AUC={val.get('auc', 0):.4f}")


if __name__ == "__main__":
    main()
