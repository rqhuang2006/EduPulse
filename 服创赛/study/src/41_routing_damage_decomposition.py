"""
Routing Damage Decomposition
==============================
Decomposes how much AUC loss comes from each part of the routing/serving structure.

Experiments:
R0: L3 single model (all features, no routing) - suspicious upper bound
R1: Core-only model (7 features) - stable baseline
R2: Core + behavior features (no routing) - feature expansion effect
R3: Core + subgroup routing (no behavior delta) - subgroup routing damage
R4: Core + behavior delta (no subgroup routing) - behavior corrector damage
R5: Full routed serving (current production) - total damage

Usage:
    python src/41_routing_damage_decomposition.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from study_common import DM_DIR
from study_feature_engine import apply_feature_engineering


# ============================================================================
# Experiment Framework
# ============================================================================

def load_and_split_data():
    """Load data and split with proper isolation (engineer AFTER split)."""
    train_path = DM_DIR / "study_train_table.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    
    raw = pd.read_csv(train_path)
    raw = raw.dropna(subset=["LABEL"])
    y = pd.to_numeric(raw["LABEL"], errors="coerce").fillna(0).astype(int)
    
    # Split FIRST, then engineer (prevents leakage)
    train_idx, valid_idx = train_test_split(
        raw.index, test_size=0.2, random_state=42, stratify=y
    )
    
    train_raw = raw.loc[train_idx]
    valid_raw = raw.loc[valid_idx]
    
    # Engineer separately
    train = apply_feature_engineering(train_raw, include_course_risk=True)
    valid = apply_feature_engineering(valid_raw, include_course_risk=True)
    
    y_train = y.loc[train_idx]
    y_valid = y.loc[valid_idx]
    
    return train, valid, y_train, y_valid


def fit_and_score(train_x, valid_x, y_train, y_valid, model_name="LightGBM"):
    """Fit model and return AUC on valid set."""
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(random_state=42, n_estimators=200, learning_rate=0.05, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(random_state=42, max_iter=200)
    
    model.fit(train_x, y_train)
    prob = model.predict_proba(valid_x)[:, 1]
    auc = roc_auc_score(y_valid, prob)
    
    return auc, prob


def experiment_r0_single_model(train, valid, y_train, y_valid):
    """
    R0: Single model with all features (no routing).
    
    This is the suspicious upper bound - if L3=0.8931 was from this,
    it indicates the feature set itself is strong (or leaky).
    """
    print("\n" + "=" * 80)
    print("R0: Single Model (All Features, No Routing)")
    print("=" * 80)
    
    # Get all feature columns
    feature_cols = [c for c in train.columns if c.startswith((
        "FEATURE_", "prev_", "hist_", "delta_", "ratio_", "trend_",
        "personal_", "imbalance_", "consecutive_decline_", "dist_from_worst_",
        "recovery_", "course_risk_", "discordance_", "workload_stress",
        "cross__", "feature_"
    )) and c not in {"FEATURE_MISSING_RATE"}]
    
    # Remove any z-score features to avoid leakage
    feature_cols = [c for c in feature_cols if "personal_z_" not in c]
    
    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    x_valid = valid[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    auc, prob = fit_and_score(x_train, x_valid, y_train, y_valid)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Valid AUC: {auc:.4f}")
    
    return {"auc": auc, "features": len(feature_cols), "prob": prob}


def experiment_r1_core_only(train, valid, y_train, y_valid):
    """
    R1: Core-only model (7-8 features).
    
    This is the stable baseline - should reproduce ~0.85 AUC.
    """
    print("\n" + "=" * 80)
    print("R1: Core-Only Model (Stable Baseline)")
    print("=" * 80)
    
    core_features = [
        "FEATURE_GRADE_AVG_SCORE",
        "FEATURE_GRADE_MIN_SCORE",
        "FEATURE_GRADE_FAIL_COUNT",
        "FEATURE_GRADE_CREDIT_SUM",
        "FEATURE_COURSE_SELECTED_COUNT",
        "FEATURE_COURSE_RETAKE_COUNT",
        "FEATURE_LIBRARY_VISIT_COUNT",
    ]
    
    # Filter to existing columns
    core_features = [c for c in core_features if c in train.columns]
    
    x_train = train[core_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    x_valid = valid[core_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    auc, prob = fit_and_score(x_train, x_valid, y_train, y_valid)
    
    print(f"Features: {len(core_features)}")
    print(f"Valid AUC: {auc:.4f}")
    
    return {"auc": auc, "features": len(core_features), "prob": prob}


def experiment_r2_core_plus_behavior(train, valid, y_train, y_valid):
    """
    R2: Core + behavior features (no routing).
    
    Tests if adding behavior features helps without routing damage.
    """
    print("\n" + "=" * 80)
    print("R2: Core + Behavior Features (No Routing)")
    print("=" * 80)
    
    # Core features
    core_features = [
        "FEATURE_GRADE_AVG_SCORE",
        "FEATURE_GRADE_MIN_SCORE",
        "FEATURE_GRADE_FAIL_COUNT",
        "FEATURE_COURSE_SELECTED_COUNT",
        "FEATURE_LIBRARY_VISIT_COUNT",
    ]
    
    # Behavior features (temporal + interaction included)
    behavior_features = [c for c in train.columns if any(marker in c for marker in [
        "attendance", "assignment", "exam", "library", "class_task",
        "prev_", "hist_", "delta_", "ratio_", "cross__", "discordance_",
        "course_risk_", "trend_", "personal_", "imbalance_"
    ]) and "personal_z_" not in c]
    
    all_features = sorted(set(core_features + behavior_features))
    
    x_train = train[all_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    x_valid = valid[all_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    auc, prob = fit_and_score(x_train, x_valid, y_train, y_valid)
    
    print(f"Features: {len(all_features)} (core={len(core_features)}, behavior={len(behavior_features)})")
    print(f"Valid AUC: {auc:.4f}")
    
    return {"auc": auc, "features": len(all_features), "prob": prob}


def experiment_r3_subgroup_routing(train, valid, y_train, y_valid):
    """
    R3: Core + subgroup routing (no behavior delta).
    
    Trains separate models for different subgroups and routes based on feature availability.
    Tests if subgroup routing itself damages AUC.
    """
    print("\n" + "=" * 80)
    print("R3: Subgroup Routing (No Behavior Delta)")
    print("=" * 80)
    
    # Core features (always available)
    core_features = [
        "FEATURE_GRADE_AVG_SCORE",
        "FEATURE_GRADE_MIN_SCORE",
        "FEATURE_GRADE_FAIL_COUNT",
        "FEATURE_COURSE_SELECTED_COUNT",
        "FEATURE_LIBRARY_VISIT_COUNT",
    ]
    
    # Subgroup features (only for samples with behavior data)
    behavior_features = [c for c in train.columns if any(marker in c for marker in [
        "attendance", "assignment", "exam", "class_task",
        "prev_", "hist_", "delta_", "course_risk_", "trend_"
    ]) and "personal_z_" not in c]
    
    # Determine which samples have behavior data
    behavior_ok = train[behavior_features].notna().any(axis=1)
    valid_behavior_ok = valid[behavior_features].notna().any(axis=1)
    
    # Train core model on all samples
    core_features = [c for c in core_features if c in train.columns]
    x_core_train = train[core_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    x_core_valid = valid[core_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    core_auc, core_prob = fit_and_score(x_core_train, x_core_valid, y_train, y_valid)
    
    # Train subgroup model only on samples with behavior data
    if behavior_ok.sum() > 100:
        subgroup_features = sorted(set(core_features + behavior_features))
        
        # Use numpy boolean arrays to avoid index alignment issues
        behavior_mask_train = behavior_ok.values
        behavior_mask_valid = valid_behavior_ok.values
        
        x_sub_train = train[behavior_mask_train][subgroup_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_sub_train = y_train[behavior_mask_train]
        x_sub_valid = valid[behavior_mask_valid][subgroup_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_sub_valid = y_valid[behavior_mask_valid]
        
        sub_auc, sub_prob = fit_and_score(x_sub_train, x_sub_valid, y_sub_train, y_sub_valid)
        
        # Route: use subgroup model for behavior-rich samples, core for others
        final_prob = core_prob.copy()
        final_prob[behavior_mask_valid] = sub_prob
    else:
        sub_auc = None
        final_prob = core_prob
    
    final_auc = roc_auc_score(y_valid, final_prob)
    
    print(f"Core model AUC: {core_auc:.4f}")
    print(f"Subgroup model AUC: {sub_auc:.4f}" if sub_auc else "Subgroup model: N/A")
    print(f"Routed AUC: {final_auc:.4f}")
    print(f"Routing damage (vs core): {final_auc - core_auc:+.4f}")
    
    return {
        "core_auc": core_auc,
        "subgroup_auc": sub_auc,
        "routed_auc": final_auc,
        "routing_damage": final_auc - core_auc,
        "prob": final_prob
    }


def experiment_r4_behavior_delta(train, valid, y_train, y_valid):
    """
    R4: Core + behavior delta (no subgroup routing).
    
    Uses behavior model to correct core model predictions on uncertain samples.
    Tests if behavior correction itself damages AUC.
    """
    print("\n" + "=" * 80)
    print("R4: Behavior Delta Correction (No Subgroup Routing)")
    print("=" * 80)
    
    # Core features
    core_features = [
        "FEATURE_GRADE_AVG_SCORE",
        "FEATURE_GRADE_MIN_SCORE",
        "FEATURE_GRADE_FAIL_COUNT",
        "FEATURE_COURSE_SELECTED_COUNT",
        "FEATURE_LIBRARY_VISIT_COUNT",
    ]
    core_features = [c for c in core_features if c in train.columns]
    
    # Behavior features
    behavior_features = [c for c in train.columns if any(marker in c for marker in [
        "attendance", "assignment", "exam", "library", "class_task",
        "prev_", "hist_", "delta_", "course_risk_"
    ]) and "personal_z_" not in c]
    
    # Train core model
    x_core_train = train[core_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    x_core_valid = valid[core_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    core_auc, core_prob = fit_and_score(x_core_train, x_core_valid, y_train, y_valid)
    
    # Compute core predictions on train set to find uncertain samples
    _, core_prob_train = fit_and_score(x_core_train, x_core_train, y_train, y_train)
    
    # Train behavior model on uncertain samples (0.3-0.7)
    uncertain_train_mask = (core_prob_train >= 0.3) & (core_prob_train <= 0.7)
    uncertain_valid_mask = (core_prob >= 0.3) & (core_prob <= 0.7)
    
    behavior_features = [c for c in behavior_features if c in train.columns]
    
    if uncertain_train_mask.sum() > 100 and behavior_features:
        x_beh_train = train[uncertain_train_mask][behavior_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        y_beh_train = y_train[uncertain_train_mask]
        
        if len(behavior_features) > 0:
            x_beh_valid = valid[uncertain_valid_mask][behavior_features].apply(pd.to_numeric, errors="coerce").fillna(0)
            y_beh_valid = y_valid[uncertain_valid_mask]
            beh_auc, beh_prob = fit_and_score(x_beh_train, x_beh_valid, y_beh_train, y_beh_valid)
            
            # Apply behavior correction to uncertain samples
            final_prob = core_prob.copy()
            correction_weight = 0.3
            final_prob[uncertain_valid_mask] = (
                core_prob[uncertain_valid_mask] * (1 - correction_weight) +
                beh_prob * correction_weight
            )
        else:
            beh_auc = None
            final_prob = core_prob
    else:
        beh_auc = None
        final_prob = core_prob
    
    final_auc = roc_auc_score(y_valid, final_prob)
    
    print(f"Core model AUC: {core_auc:.4f}")
    print(f"Behavior model AUC: {beh_auc:.4f}" if beh_auc else "Behavior model: N/A")
    print(f"Corrected AUC: {final_auc:.4f}")
    print(f"Behavior delta damage (vs core): {final_auc - core_auc:+.4f}")
    
    return {
        "core_auc": core_auc,
        "behavior_auc": beh_auc,
        "corrected_auc": final_auc,
        "behavior_damage": final_auc - core_auc,
        "prob": final_prob
    }


def main():
    print("Routing Damage Decomposition")
    print("=" * 80)
    
    # Load data
    train, valid, y_train, y_valid = load_and_split_data()
    
    print(f"Train samples: {len(train)}")
    print(f"Valid samples: {len(valid)}")
    print(f"Positive rate (train): {y_train.mean():.4f}")
    print(f"Positive rate (valid): {y_valid.mean():.4f}")
    
    # Run experiments
    results = {}
    
    r0 = experiment_r0_single_model(train, valid, y_train, y_valid)
    results["R0_single_model"] = {"auc": r0["auc"], "features": r0["features"]}
    
    r1 = experiment_r1_core_only(train, valid, y_train, y_valid)
    results["R1_core_only"] = {"auc": r1["auc"], "features": r1["features"]}
    
    r2 = experiment_r2_core_plus_behavior(train, valid, y_train, y_valid)
    results["R2_core_plus_behavior"] = {"auc": r2["auc"], "features": r2["features"]}
    
    r3 = experiment_r3_subgroup_routing(train, valid, y_train, y_valid)
    results["R3_subgroup_routing"] = {
        "core_auc": r3["core_auc"],
        "subgroup_auc": r3["subgroup_auc"],
        "routed_auc": r3["routed_auc"],
        "routing_damage": r3["routing_damage"]
    }
    
    r4 = experiment_r4_behavior_delta(train, valid, y_train, y_valid)
    results["R4_behavior_delta"] = {
        "core_auc": r4["core_auc"],
        "behavior_auc": r4["behavior_auc"],
        "corrected_auc": r4["corrected_auc"],
        "behavior_damage": r4["behavior_damage"]
    }
    
    # Summary
    print("\n" + "=" * 80)
    print("ROUTING DAMAGE DECOMPOSITION SUMMARY")
    print("=" * 80)
    
    print(f"{'Experiment':<35} {'AUC':>8} {'Delta':>10} {'Notes'}")
    print("-" * 80)
    
    baseline_auc = r1["auc"]  # R1 is the stable baseline
    
    for name, data in [
        ("R0: Single model (all features)", results["R0_single_model"]),
        ("R1: Core-only (baseline)", results["R1_core_only"]),
        ("R2: Core + behavior (no routing)", results["R2_core_plus_behavior"]),
        ("R3: Subgroup routing", results["R3_subgroup_routing"]),
        ("R4: Behavior delta", results["R4_behavior_delta"]),
    ]:
        auc = data.get("auc") or data.get("routed_auc") or data.get("corrected_auc")
        delta = auc - baseline_auc if auc else None
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        
        notes = ""
        if "routing_damage" in data:
            notes = f"routing_damage={data['routing_damage']:+.4f}"
        elif "behavior_damage" in data:
            notes = f"behavior_damage={data['behavior_damage']:+.4f}"
        
        print(f"{name:<35} {auc:>8.4f} {delta_str:>10} {notes}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    r3_damage = results["R3_subgroup_routing"].get("routing_damage", 0)
    r4_damage = results["R4_behavior_delta"].get("behavior_damage", 0)
    r2_vs_r1 = results["R2_core_plus_behavior"]["auc"] - results["R1_core_only"]["auc"]
    
    print(f"1. Feature expansion effect (R1->R2): {r2_vs_r1:+.4f}")
    print(f"2. Subgroup routing damage: {r3_damage:+.4f}")
    print(f"3. Behavior delta damage: {r4_damage:+.4f}")
    
    if abs(r3_damage) > 0.01:
        print("[WARN] Subgroup routing causes significant AUC damage")
    if abs(r4_damage) > 0.01:
        print("[WARN] Behavior delta causes significant AUC damage")
    if r2_vs_r1 > 0.01:
        print("[OK] Feature expansion helps AUC")
    else:
        print("[WARN] Feature expansion doesn't help much without proper routing")
    
    # Save results
    output_path = DM_DIR / "study_routing_damage_decomposition.json"
    # Remove prob arrays for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        serializable_results[key] = {k: v for k, v in value.items() if not isinstance(v, np.ndarray)}
    
    output_path.write_text(json.dumps(serializable_results, indent=2))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
