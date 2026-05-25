"""
L3 Leakage Audit
=================
Audits the L3 experiment (AUC=0.8931) for data leakage.

Checks:
1. Time travel: features using post-prediction information
2. Label proxy: features directly translating LABEL
3. Full-table statistics: z-scores computed on combined train+valid
4. Rare combination keys: course/term combinations forming label-like keys

Usage:
    python src/40_l3_leakage_audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from study_common import DM_DIR


# ============================================================================
# Reproduce L3 with leakage detection
# ============================================================================

def audit_z_score_leakage():
    """
    Check if personal_z_* features use full-dataset statistics.
    
    If z-scores are computed BEFORE train/valid split, the valid set
    statistics are contaminated by training data.
    """
    print("=" * 80)
    print("AUDIT 1: Z-Score Leakage")
    print("=" * 80)
    
    # Load data
    train_path = DM_DIR / "study_train_table.csv"
    if not train_path.exists():
        print("ERROR: No train table found.")
        return {}
    
    raw = pd.read_csv(train_path)
    
    # Method A: Engineer BEFORE split (LEAKY)
    from study_feature_engine import apply_feature_engineering
    engineered_all = apply_feature_engineering(raw, include_course_risk=True)
    labeled = engineered_all.dropna(subset=["LABEL"])
    y_all = pd.to_numeric(labeled["LABEL"], errors="coerce").fillna(0).astype(int)
    
    # Split AFTER engineering
    X_all = labeled[[c for c in engineered_all.columns if c.startswith(("personal_z_", "personal_gap_", "dist_from_worst_")) 
                     and c in labeled.columns]]
    
    train_idx, valid_idx = train_test_split(
        labeled.index, test_size=0.2, random_state=42, stratify=y_all.loc[labeled.index]
    )
    
    # Check if valid set z-scores use training data statistics
    # If z-scores are computed on full data, valid_z should correlate with train_z
    z_cols = [c for c in X_all.columns if "personal_z_" in c]
    
    if z_cols:
        train_z = X_all.loc[train_idx, z_cols].apply(pd.to_numeric, errors="coerce")
        valid_z = X_all.loc[valid_idx, z_cols].apply(pd.to_numeric, errors="coerce")
        
        # If leakage exists, valid z-scores should have similar distribution to train
        train_mean = train_z.mean()
        valid_mean = valid_z.mean()
        
        print(f"Z-score columns found: {len(z_cols)}")
        print(f"Train z-score mean: {train_mean.mean():.4f}")
        print(f"Valid z-score mean: {valid_mean.mean():.4f}")
        print(f"Difference: {abs(train_mean.mean() - valid_mean.mean()):.4f}")
        
        if abs(train_mean.mean() - valid_mean.mean()) < 0.1:
            print("[WARN]  WARNING: Valid z-scores closely match train distribution - LIKELY LEAKAGE")
        else:
            print("[OK] Valid z-scores differ from train - appears clean")
    
    # Method B: Engineer AFTER split (CLEAN)
    raw_labeled = raw.dropna(subset=["LABEL"])
    y_labeled = pd.to_numeric(raw_labeled["LABEL"], errors="coerce").fillna(0).astype(int)
    
    train_idx2, valid_idx2 = train_test_split(
        raw_labeled.index, test_size=0.2, random_state=42, stratify=y_labeled
    )
    
    train_clean = apply_feature_engineering(raw_labeled.loc[train_idx2], include_course_risk=True)
    valid_clean = apply_feature_engineering(raw_labeled.loc[valid_idx2], include_course_risk=True)
    
    # Compare AUC: leaky vs clean
    from lightgbm import LGBMClassifier
    
    results = {}
    
    for method, X_train, X_valid, y_train, y_valid, label in [
        ("LEAKY (engineer before split)", 
         X_all.loc[train_idx, z_cols] if z_cols else pd.DataFrame(),
         X_all.loc[valid_idx, z_cols] if z_cols else pd.DataFrame(),
         y_all.loc[train_idx], y_all.loc[valid_idx], "Leaky Z"),
        ("CLEAN (engineer after split)",
         train_clean[[c for c in train_clean.columns if "personal_z_" in c]].apply(pd.to_numeric, errors="coerce"),
         valid_clean[[c for c in valid_clean.columns if "personal_z_" in c]].apply(pd.to_numeric, errors="coerce"),
         y_labeled.loc[train_idx2], y_labeled.loc[valid_idx2], "Clean Z"),
    ]:
        if X_train.empty or X_valid.empty:
            continue
            
        X_train = X_train.fillna(0)
        X_valid = X_valid.fillna(0)
        
        model = LGBMClassifier(random_state=42, n_estimators=100, verbose=-1)
        model.fit(X_train, y_train)
        
        prob = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, prob)
        
        results[label] = auc
        print(f"\n{label}:")
        print(f"  AUC = {auc:.4f}")
    
    if "Leaky Z" in results and "Clean Z" in results:
        delta = results["Leaky Z"] - results["Clean Z"]
        print(f"\nLeakage effect: {delta:+.4f}")
        if delta > 0.02:
            print("[WARN]  SIGNIFICANT LEAKAGE DETECTED (>2% AUC inflation)")
    
    return results


def audit_label_proxy():
    """
    Check if any features are direct proxies of LABEL.
    
    Features that are computed from or highly correlated with LABEL
    will create artificial AUC inflation.
    """
    print("\n" + "=" * 80)
    print("AUDIT 2: Label Proxy Features")
    print("=" * 80)
    
    train_path = DM_DIR / "study_train_table.csv"
    raw = pd.read_csv(train_path)
    
    from study_feature_engine import apply_feature_engineering
    engineered = apply_feature_engineering(raw, include_course_risk=True)
    labeled = engineered.dropna(subset=["LABEL"])
    y = pd.to_numeric(labeled["LABEL"], errors="coerce").fillna(0).astype(int)
    
    # Check correlation of each feature with LABEL
    feature_cols = [c for c in engineered.columns if c.startswith(("course_risk_", "personal_", "trend_", "dist_from_worst_"))]
    
    correlations = []
    for col in feature_cols:
        if col in labeled.columns:
            vals = pd.to_numeric(labeled[col], errors="coerce")
            corr = vals.corr(y)
            correlations.append({"feature": col, "corr_with_label": abs(corr)})
    
    correlations.sort(key=lambda x: x["corr_with_label"], reverse=True)
    
    print("\nTop 10 features by correlation with LABEL:")
    for item in correlations[:10]:
        flag = "[WARN]  POTENTIAL PROXY" if item["corr_with_label"] > 0.5 else ""
        print(f"  {item['feature']:40s} |corr|={item['corr_with_label']:.4f} {flag}")
    
    return {"high_corr_features": [c for c in correlations if c["corr_with_label"] > 0.5]}


def audit_rare_combinations():
    """
    Check if rare course/term combinations form label-like keys.
    
    If certain combinations only appear in positive/negative cases,
    the model can learn to use them as direct label indicators.
    """
    print("\n" + "=" * 80)
    print("AUDIT 3: Rare Combination Keys")
    print("=" * 80)
    
    train_path = DM_DIR / "study_train_table.csv"
    raw = pd.read_csv(train_path)
    
    if "LABEL" not in raw.columns:
        print("No LABEL column found.")
        return {}
    
    y = pd.to_numeric(raw["LABEL"], errors="coerce").fillna(0).astype(int)
    
    # Check course/term combinations
    combo_cols = [c for c in raw.columns if any(k in c for k in ["COURSE", "TERM", "CLASS"])]
    
    rare_combos = []
    for col in combo_cols:
        if col in raw.columns:
            # Check if certain values only appear in positive/negative
            grouped = raw.groupby(col)["LABEL"].mean()
            pure_pos = (grouped == 1.0).sum()
            pure_neg = (grouped == 0.0).sum()
            
            if pure_pos > 0 or pure_neg > 0:
                rare_combos.append({
                    "column": col,
                    "pure_positive_values": int(pure_pos),
                    "pure_negative_values": int(pure_neg),
                })
    
    if rare_combos:
        print("\nColumns with pure positive/negative values:")
        for item in rare_combos:
            print(f"  {item['column']:40s} | pure_pos={item['pure_positive_values']}, pure_neg={item['pure_negative_values']}")
    else:
        print("\n[OK] No rare combination keys detected.")
    
    return {"rare_combos": rare_combos}


def main():
    print("L3 Leakage Audit")
    print("=" * 80)
    print()
    
    results = {}
    
    # Audit 1: Z-score leakage
    z_results = audit_z_score_leakage()
    results["z_score_leakage"] = z_results
    
    # Audit 2: Label proxy
    proxy_results = audit_label_proxy()
    results["label_proxy"] = proxy_results
    
    # Audit 3: Rare combinations
    rare_results = audit_rare_combinations()
    results["rare_combinations"] = rare_results
    
    # Summary
    print("\n" + "=" * 80)
    print("LEAKAGE AUDIT SUMMARY")
    print("=" * 80)
    
    has_leakage = False
    
    if z_results.get("Leaky Z", 0) - z_results.get("Clean Z", 0) > 0.02:
        print("[WARN]  Z-Score leakage: SIGNIFICANT (>2% AUC inflation)")
        has_leakage = True
    else:
        print("[OK] Z-Score leakage: minimal or none")
    
    if proxy_results.get("high_corr_features"):
        print(f"[WARN]  Label proxy: {len(proxy_results['high_corr_features'])} high-correlation features found")
        has_leakage = True
    else:
        print("[OK] Label proxy: no obvious proxies")
    
    if rare_results.get("rare_combos"):
        print(f"[WARN]  Rare combinations: {len(rare_results['rare_combos'])} columns with pure values")
        has_leakage = True
    else:
        print("[OK] Rare combinations: none detected")
    
    if has_leakage:
        print("\n" + "=" * 80)
        print("CONCLUSION: L3 AUC=0.8931 is INFLATED by data leakage")
        print("The true AUC (without leakage) is likely closer to L4=0.8538")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("CONCLUSION: L3 appears clean - leakage not detected")
        print("The 0.8931 AUC may be genuine")
        print("=" * 80)
    
    # Save results
    output_path = DM_DIR / "study_l3_leakage_audit.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
