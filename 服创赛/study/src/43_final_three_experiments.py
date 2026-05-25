"""
Final Three Experiments - Unified Comparison Framework
======================================================
Runs three final experiments with EXACTLY the same:
- holdout IDs
- label definition
- feature engineering order (AFTER split)
- population
- model class

Experiments:
A. Pure core 8 features strong baseline
B. Core + audited clean incremental features
C. Light serving single model (no complex routing)

Usage:
    python src/43_final_three_experiments.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from study_common import DM_DIR
from study_feature_engine import apply_feature_engineering


# ============================================================================
# Unified Experiment Configuration (FROZEN)
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_CLASS = "LightGBMClassifier"  # Fixed model class

# Feature sets
CORE_8_FEATURES = [
    "FEATURE_GRADE_AVG_SCORE",
    "FEATURE_GRADE_MIN_SCORE",
    "FEATURE_GRADE_FAIL_COUNT",
    "FEATURE_GRADE_CREDIT_SUM",
    "FEATURE_COURSE_SELECTED_COUNT",
    "FEATURE_COURSE_CREDIT_SUM",
    "FEATURE_COURSE_RETAKE_COUNT",
    "FEATURE_LIBRARY_VISIT_COUNT",
]

# Clean incremental features (audited - NO z-scores, NO leakage)
INCREMENTAL_FEATURES = [
    # Core course risk
    "course_risk_min_score",
    "course_risk_avg_min_gap",
    "course_risk_fail_count",
    "course_risk_distance_to_60",
    "course_risk_min_distance_to_60",
    "course_risk_near_fail",
    "course_risk_marginal_pass_count",
    "course_risk_multi_course_danger",
    "course_risk_dispersion",
    "course_risk_dispersion_coef",
    "course_risk_weakest_gap_to_60",
    "course_risk_weakest_gap_to_70",
    "course_risk_bottleneck_severity",
    "course_risk_bottleneck_ratio",
    "course_risk_fail_rate",
    "course_risk_consecutive_decline",
    # Discordance features
    "discordance_attendance_grade",
    "discordance_effort_result",
    "discordance_library_grade",
    # Temporal (clean - computed after split)
    "prev_grade_avg_score",
    "hist_grade_avg_score",
    "delta_grade_avg_score",
    "ratio_grade_avg_score",
    "consecutive_decline_grade_avg_score",
    "dist_from_worst_grade_avg_score",
    "recovery_grade_avg_score",
]


# ============================================================================
# Experiment Functions
# ============================================================================

def load_and_split_data():
    """
    Load data and split with PROPER order:
    1. Load raw data
    2. Split FIRST (train/holdout)
    3. THEN engineer features on each split separately
    
    This prevents ALL forms of data leakage.
    """
    train_path = DM_DIR / "study_train_table.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Train table not found: {train_path}")
    
    raw = pd.read_csv(train_path)
    raw = raw.dropna(subset=["LABEL"])
    y = pd.to_numeric(raw["LABEL"], errors="coerce").fillna(0).astype(int)
    
    # Split FIRST - this is critical
    train_idx, holdout_idx = train_test_split(
        raw.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    train_raw = raw.loc[train_idx].copy()
    holdout_raw = raw.loc[holdout_idx].copy()
    
    # Engineer AFTER split (prevents leakage)
    train = apply_feature_engineering(train_raw, include_course_risk=True)
    holdout = apply_feature_engineering(holdout_raw, include_course_risk=True)
    
    y_train = y.loc[train_idx]
    y_holdout = y.loc[holdout_idx]
    
    return train, holdout, y_train, y_holdout, train_idx, holdout_idx


def fit_and_evaluate(x_train, x_holdout, y_train, y_holdout, exp_name, feature_set_name):
    """
    Train model and compute comprehensive metrics.
    """
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
        )
    
    # Fill NaN after feature engineering
    x_train = x_train.fillna(0)
    x_holdout = x_holdout.fillna(0)
    
    # Train
    model.fit(x_train, y_train)
    
    # Predict
    prob = model.predict_proba(x_holdout)[:, 1]
    pred = (prob >= 0.5).astype(int)
    
    # Metrics
    metrics = {
        "experiment": exp_name,
        "feature_set": feature_set_name,
        "feature_count": len(x_train.columns),
        "model_class": MODEL_CLASS,
        "holdout_size": len(y_holdout),
        "holdout_positive_rate": float(y_holdout.mean()),
    }
    
    if y_holdout.nunique() > 1:
        metrics["holdout_auc"] = float(roc_auc_score(y_holdout, prob))
    
    metrics["holdout_accuracy"] = float(np.mean(pred == y_holdout))
    metrics["holdout_f1"] = float(f1_score(y_holdout, pred, zero_division=0))
    metrics["holdout_recall"] = float(recall_score(y_holdout, pred, zero_division=0))
    metrics["holdout_precision"] = float(precision_score(y_holdout, pred, zero_division=0))
    
    # Subgroup metrics
    # (Will be computed if LABEL_SUBTYPE exists in holdout_raw)
    
    return metrics, prob


def experiment_a_core_baseline(train, holdout, y_train, y_holdout, train_idx, holdout_idx):
    """
    Experiment A: Pure Core 8 Features Strong Baseline
    
    This should reproduce ~0.85 AUC if conditions are right.
    """
    print("\n" + "=" * 80)
    print("Experiment A: Pure Core 8 Features Strong Baseline")
    print("=" * 80)
    
    # Filter to existing columns
    features = [f for f in CORE_8_FEATURES if f in train.columns]
    
    x_train = train[features]
    x_holdout = holdout[features]
    
    print(f"Features ({len(features)}): {features}")
    print(f"Train samples: {len(x_train)}")
    print(f"Holdout samples: {len(x_holdout)}")
    
    metrics, prob = fit_and_evaluate(x_train, x_holdout, y_train, y_holdout, "A", "core_8")
    
    print(f"\nResults:")
    print(f"  AUC: {metrics.get('holdout_auc', 'N/A'):.4f}" if metrics.get('holdout_auc') else "  AUC: N/A")
    print(f"  F1: {metrics['holdout_f1']:.4f}")
    print(f"  Recall: {metrics['holdout_recall']:.4f}")
    print(f"  Precision: {metrics['holdout_precision']:.4f}")
    
    return metrics, prob


def experiment_b_core_plus_clean(train, holdout, y_train, y_holdout, train_idx, holdout_idx):
    """
    Experiment B: Core + Audited Clean Incremental Features
    
    Adds only features that passed leakage audit.
    """
    print("\n" + "=" * 80)
    print("Experiment B: Core + Audited Clean Incremental Features")
    print("=" * 80)
    
    # Core features
    core_features = [f for f in CORE_8_FEATURES if f in train.columns]
    
    # Incremental features (only those that exist)
    incr_features = [f for f in INCREMENTAL_FEATURES if f in train.columns]
    
    # Combined
    all_features = sorted(set(core_features + incr_features))
    
    x_train = train[all_features]
    x_holdout = holdout[all_features]
    
    print(f"Core features: {len(core_features)}")
    print(f"Incremental features: {len(incr_features)}")
    print(f"Total features: {len(all_features)}")
    print(f"Train samples: {len(x_train)}")
    print(f"Holdout samples: {len(x_holdout)}")
    
    metrics, prob = fit_and_evaluate(x_train, x_holdout, y_train, y_holdout, "B", "core_plus_clean")
    
    print(f"\nResults:")
    print(f"  AUC: {metrics.get('holdout_auc', 'N/A'):.4f}" if metrics.get('holdout_auc') else "  AUC: N/A")
    print(f"  F1: {metrics['holdout_f1']:.4f}")
    print(f"  Recall: {metrics['holdout_recall']:.4f}")
    print(f"  Precision: {metrics['holdout_precision']:.4f}")
    
    return metrics, prob


def experiment_c_light_serving(train, holdout, y_train, y_holdout, train_idx, holdout_idx):
    """
    Experiment C: Light Serving Single Model (No Complex Routing)
    
    Uses all clean features but trains as a SINGLE model.
    No routing, no behavior correction, no subgroup experts.
    """
    print("\n" + "=" * 80)
    print("Experiment C: Light Serving Single Model (No Routing)")
    print("=" * 80)
    
    # Get all available clean features
    all_features = [c for c in train.columns if c.startswith((
        "FEATURE_", "course_risk_", "prev_", "hist_", "delta_", "ratio_",
        "trend_", "personal_", "imbalance_", "consecutive_decline_",
        "dist_from_worst_", "recovery_", "discordance_", "workload_stress",
        "cross__", "feature_"
    )) and c not in {"FEATURE_MISSING_RATE"}
    and not c.startswith("personal_z_")]  # Exclude leaked features
    
    x_train = train[all_features]
    x_holdout = holdout[all_features]
    
    print(f"Total features: {len(all_features)}")
    print(f"Train samples: {len(x_train)}")
    print(f"Holdout samples: {len(x_holdout)}")
    print(f"Model: Single {MODEL_CLASS} (no routing)")
    
    metrics, prob = fit_and_evaluate(x_train, x_holdout, y_train, y_holdout, "C", "light_serving_all_clean")
    
    print(f"\nResults:")
    print(f"  AUC: {metrics.get('holdout_auc', 'N/A'):.4f}" if metrics.get('holdout_auc') else "  AUC: N/A")
    print(f"  F1: {metrics['holdout_f1']:.4f}")
    print(f"  Recall: {metrics['holdout_recall']:.4f}")
    print(f"  Precision: {metrics['holdout_precision']:.4f}")
    
    return metrics, prob


def main():
    print("Final Three Experiments - Unified Comparison Framework")
    print("=" * 80)
    print(f"Random state: {RANDOM_STATE}")
    print(f"Test size: {TEST_SIZE}")
    print(f"Model class: {MODEL_CLASS}")
    print(f"Feature engineering order: AFTER split (leakage-free)")
    
    # Load and split data (ONCE - unified)
    train, holdout, y_train, y_holdout, train_idx, holdout_idx = load_and_split_data()
    
    print(f"\nData loaded:")
    print(f"  Total samples: {len(train) + len(holdout)}")
    print(f"  Train samples: {len(train)}")
    print(f"  Holdout samples: {len(holdout)}")
    print(f"  Train positive rate: {y_train.mean():.4f}")
    print(f"  Holdout positive rate: {y_holdout.mean():.4f}")
    
    # Save train/holdout IDs for reproducibility
    id_record = {
        "train_ids": train_idx.tolist(),
        "holdout_ids": holdout_idx.tolist(),
        "train_count": len(train_idx),
        "holdout_count": len(holdout_idx),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "stratify": "LABEL",
    }
    id_path = DM_DIR / "study_final_experiment_ids.json"
    id_path.write_text(json.dumps(id_record, indent=2))
    print(f"\nExperiment IDs saved to: {id_path}")
    
    # Run three experiments
    metrics_a, prob_a = experiment_a_core_baseline(train, holdout, y_train, y_holdout, train_idx, holdout_idx)
    metrics_b, prob_b = experiment_b_core_plus_clean(train, holdout, y_train, y_holdout, train_idx, holdout_idx)
    metrics_c, prob_c = experiment_c_light_serving(train, holdout, y_train, y_holdout, train_idx, holdout_idx)
    
    # Unified comparison
    print("\n" + "=" * 80)
    print("UNIFIED COMPARISON RESULTS")
    print("=" * 80)
    
    results = [metrics_a, metrics_b, metrics_c]
    
    # Create comparison table
    comparison = pd.DataFrame(results)
    print("\nComparison Table:")
    print(comparison[["experiment", "feature_set", "feature_count", "holdout_auc", "holdout_f1", "holdout_recall", "holdout_precision"]].to_string(index=False))
    
    # Delta from baseline (A)
    if metrics_a.get("holdout_auc"):
        print(f"\nDelta vs Baseline (A):")
        for name, metrics in [("B", metrics_b), ("C", metrics_c)]:
            auc_delta = metrics.get("holdout_auc", 0) - metrics_a.get("holdout_auc", 0)
            f1_delta = metrics.get("holdout_f1", 0) - metrics_a.get("holdout_f1", 0)
            recall_delta = metrics.get("holdout_recall", 0) - metrics_a.get("holdout_recall", 0)
            precision_delta = metrics.get("holdout_precision", 0) - metrics_a.get("holdout_precision", 0)
            
            print(f"  {name}: AUC delta={auc_delta:+.4f}, F1 delta={f1_delta:+.4f}, "
                  f"Recall delta={recall_delta:+.4f}, Precision delta={precision_delta:+.4f}")
    
    # Save results
    output_path = DM_DIR / "study_final_three_experiments.json"
    output_data = {
        "experiment_A": metrics_a,
        "experiment_B": metrics_b,
        "experiment_C": metrics_c,
        "unified_framework": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "model_class": MODEL_CLASS,
            "feature_engineering_order": "after_split",
            "population": "all_labeled",
            "train_ids_path": str(id_path),
            "holdout_ids_path": str(id_path),
        },
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nFull results saved to: {output_path}")
    
    # Final conclusions
    print("\n" + "=" * 80)
    print("PRELIMINARY CONCLUSIONS")
    print("=" * 80)
    
    auc_a = metrics_a.get("holdout_auc", 0)
    auc_b = metrics_b.get("holdout_auc", 0)
    auc_c = metrics_c.get("holdout_auc", 0)
    
    print(f"1. Can we reach 0.85 AUC in unified framework?")
    max_auc = max(auc_a, auc_b, auc_c)
    if max_auc >= 0.85:
        print(f"   YES - Experiment with highest AUC: {max_auc:.4f}")
    else:
        print(f"   NO - Best AUC achieved: {max_auc:.4f}")
        print(f"   Gap to 0.85: {0.85 - max_auc:.4f}")
    
    print(f"\n2. Recommended final version:")
    if auc_c >= auc_a and auc_c >= auc_b:
        print("   Experiment C (Light Serving) - best overall AUC")
    elif auc_b >= auc_a:
        print("   Experiment B (Core + Clean Incremental) - good balance")
    else:
        print("   Experiment A (Core Baseline) - simplest, most stable")


if __name__ == "__main__":
    main()
