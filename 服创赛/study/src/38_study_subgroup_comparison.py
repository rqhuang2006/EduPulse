"""
Study Model Subgroup Comparison
================================
Outputs detailed comparison metrics across subgroups:
- overall AUC
- single_fail AUC
- core_plus_behavior AUC
- uncertain band AUC

Run after training to verify real improvements.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from study_common import DM_DIR
from study_feature_engine import apply_feature_engineering


def predict_prob(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    pred = model.predict(x)
    return np.asarray(pred, dtype=float)


def compute_subgroup_auc(model, x: pd.DataFrame, y: pd.Series, train_df: pd.DataFrame) -> dict:
    """Compute AUC for each subgroup."""
    prob = predict_prob(model, x)
    results = {}
    
    # Overall
    if y.nunique() > 1:
        results["overall_auc"] = float(roc_auc_score(y, prob))
        results["overall_count"] = len(y)
    
    # Single fail subgroup - need to align indices
    if "LABEL_SUBTYPE" in train_df.columns:
        # Align LABEL_SUBTYPE to x index
        aligned_subtype = train_df["LABEL_SUBTYPE"].reindex(x.index)
        
        single_fail_mask = aligned_subtype == "single_fail"
        if single_fail_mask.sum() > 0 and y[single_fail_mask].nunique() > 1:
            results["single_fail_auc"] = float(roc_auc_score(y[single_fail_mask], prob[single_fail_mask]))
            results["single_fail_count"] = int(single_fail_mask.sum())
        
        overall_low_mask = aligned_subtype == "overall_low"
        if overall_low_mask.sum() > 0 and y[overall_low_mask].nunique() > 1:
            results["overall_low_auc"] = float(roc_auc_score(y[overall_low_mask], prob[overall_low_mask]))
            results["overall_low_count"] = int(overall_low_mask.sum())
    
    # Core + Behavior subgroup (has behavior features)
    behavior_cols = [c for c in x.columns if any(marker in c.lower() for marker in 
        ["attendance", "library", "online", "assignment", "exam", "class_task"])]
    if behavior_cols:
        behavior_ok = x[behavior_cols].notna().any(axis=1)
        if behavior_ok.sum() > 0 and y[behavior_ok].nunique() > 1:
            results["core_plus_behavior_auc"] = float(roc_auc_score(y[behavior_ok], prob[behavior_ok]))
            results["core_plus_behavior_count"] = int(behavior_ok.sum())
    
    # Uncertain band (predictions between 0.3-0.7)
    uncertain_mask = (prob >= 0.3) & (prob <= 0.7)
    if uncertain_mask.sum() > 0 and y[uncertain_mask].nunique() > 1:
        results["uncertain_band_auc"] = float(roc_auc_score(y[uncertain_mask], prob[uncertain_mask]))
        results["uncertain_band_count"] = int(uncertain_mask.sum())
    
    return results


def main():
    # Load trained model
    model_path = DM_DIR / "study_model.pkl"
    if not model_path.exists():
        print("ERROR: No trained model found. Run training first.")
        return
    
    import joblib
    bundle = joblib.load(model_path)
    core_model = bundle["core_model"]
    config = bundle["config"]
    
    # Load train data
    train_path = DM_DIR / "study_train_table.csv"
    if not train_path.exists():
        print("ERROR: No train table found.")
        return
    
    train = pd.read_csv(train_path)
    train = apply_feature_engineering(train, include_course_risk=True)
    train = train.dropna(subset=["LABEL"])
    
    y = pd.to_numeric(train["LABEL"], errors="coerce").fillna(0).astype(int)
    
    # Get feature columns from config - use ALL features for comprehensive comparison
    core_features = config.get("core_feature_columns", [])
    behavior_features = config.get("behavior_feature_columns", [])
    temporal_features = config.get("temporal_feature_columns", [])
    interaction_features = config.get("interaction_feature_columns", [])
    
    # Use all available features for the comparison
    all_features = sorted(set(core_features + behavior_features + temporal_features + interaction_features))
    x_all = train[all_features].apply(pd.to_numeric, errors="coerce")
    
    # Also create core-only version for comparison
    x_core = train[core_features].apply(pd.to_numeric, errors="coerce")
    
    # Compute subgroup metrics
    print("=" * 80)
    print("Study Model Subgroup Comparison")
    print("=" * 80)
    print(f"Model: {config.get('primary_model')}")
    print(f"Architecture: {config.get('architecture_version')}")
    print(f"Core features: {len(core_features)}")
    print(f"All features: {len(all_features)}")
    print()
    
    # Core model only (uses core features)
    print("Core Model (core features only):")
    core_metrics = compute_subgroup_auc(core_model, x_core, y, train)
    for key in ["overall_auc", "single_fail_auc", "overall_low_auc", 
                "core_plus_behavior_auc", "uncertain_band_auc"]:
        if key in core_metrics:
            count_key = key.replace("_auc", "_count")
            count = core_metrics.get(count_key, "N/A")
            print(f"  {key:<35} {core_metrics[key]:>10.4f} {count:>10}")
    
    print("-" * 80)
    print()
    print("=" * 80)
    
    # Save to file
    output_path = DM_DIR / "study_subgroup_comparison.json"
    output_data = {
        "core_model": core_metrics,
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"Results saved to: {output_path}")
    
    # Compare with baseline if available
    baseline_metrics = config.get("metrics", {}).get("core_model", {}).get("valid", {})
    if baseline_metrics:
        print()
        print("Comparison with Baseline (from training):")
        print(f"  Baseline valid AUC: {baseline_metrics.get('auc', 'N/A'):.4f}")
        print(f"  Current overall AUC: {core_metrics.get('overall_auc', 'N/A'):.4f}")
        diff = core_metrics.get('overall_auc', 0) - baseline_metrics.get('auc', 0)
        print(f"  Difference: {diff:+.4f}")


if __name__ == "__main__":
    main()
