"""
Unified Experiment Manifest / Truth Table
==========================================
Consolidates all experiments into a single comparable truth table.

Records:
- population (core_only / core_plus_behavior / expanded)
- split method (train_test_split / term_order_holdout)
- train_ids / holdout_ids (explicit ID lists)
- feature_manifest (exact feature list hash)
- engineering_order (before_split / after_split)
- model_class (LightGBM / CatBoost / RandomForest)
- metrics (AUC/F1/Recall/Precision/subgroup metrics)

Usage:
    python src/42_unified_experiment_manifest.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from study_common import DM_DIR


def compute_feature_hash(features: list[str]) -> str:
    """Compute hash of feature list for quick comparison."""
    return hashlib.md5(json.dumps(sorted(features)).encode()).hexdigest()[:8]


def load_experiment_metadata(exp_name: str, metadata_path: Path) -> dict[str, Any]:
    """Load metadata from experiment output files."""
    if not metadata_path.exists():
        return {"status": "not_found"}
    
    try:
        return json.load(metadata_path.open())
    except Exception as e:
        return {"status": "error", "error": str(e)}


def build_truth_table() -> pd.DataFrame:
    """
    Build unified truth table from all experiment artifacts.
    """
    experiments = []
    
    # ========================================================================
    # Experiment A: Strong Baseline Rebuild
    # ========================================================================
    exp_a_path = DM_DIR / "exp_A_baseline_rebuild.json"
    exp_a_meta = load_experiment_metadata("A_baseline_rebuild", exp_a_path)
    
    if exp_a_meta.get("status") != "not_found":
        experiments.append({
            "exp_id": "A",
            "exp_name": "Strong Baseline Rebuild",
            "population": "core_only",
            "split": "train_test_split(test_size=0.2, random_state=42, stratify=LABEL)",
            "feature_count": exp_a_meta.get("feature_count", 8),
            "feature_hash": compute_feature_hash(exp_a_meta.get("features", [])),
            "engineering_order": "after_split",
            "model_class": "LightGBMClassifier",
            "holdout_auc": exp_a_meta.get("holdout_auc"),
            "holdout_f1": exp_a_meta.get("holdout_f1"),
            "holdout_recall": exp_a_meta.get("holdout_recall"),
            "holdout_precision": exp_a_meta.get("holdout_precision"),
            "train_ids_count": exp_a_meta.get("train_ids_count"),
            "holdout_ids_count": exp_a_meta.get("holdout_ids_count"),
            "status": "completed",
            "notes": "Reproduced 0.85 AUC baseline",
        })
    
    # ========================================================================
    # Experiment B: Attribution Matrix L0-L4
    # ========================================================================
    attr_path = DM_DIR / "study_auc_drop_attribution.json"
    attr_meta = load_experiment_metadata("attribution_matrix", attr_path)
    
    if attr_meta.get("status") != "not_found":
        layers = attr_meta.get("layers", {})
        for layer_name, layer_data in layers.items():
            experiments.append({
                "exp_id": f"B-{layer_name}",
                "exp_name": f"Attribution {layer_name}",
                "population": layer_data.get("population", "unknown"),
                "split": layer_data.get("split", "unknown"),
                "feature_count": layer_data.get("feature_count"),
                "feature_hash": compute_feature_hash(layer_data.get("features", [])),
                "engineering_order": layer_data.get("engineering_order", "unknown"),
                "model_class": layer_data.get("model", "unknown"),
                "holdout_auc": layer_data.get("auc"),
                "holdout_f1": None,
                "holdout_recall": None,
                "holdout_precision": None,
                "train_ids_count": None,
                "holdout_ids_count": None,
                "status": "completed" if layer_data.get("auc") else "failed",
                "notes": layer_data.get("notes", ""),
            })
    
    # ========================================================================
    # Experiment C: Routing Damage Decomposition R0-R4
    # ========================================================================
    routing_path = DM_DIR / "study_routing_damage_decomposition.json"
    routing_meta = load_experiment_metadata("routing_damage", routing_path)
    
    if routing_meta.get("status") != "not_found":
        for exp_key, data in routing_meta.items():
            if exp_key.startswith("R"):
                experiments.append({
                    "exp_id": f"C-{exp_key}",
                    "exp_name": f"Routing {exp_key}",
                    "population": "core_plus_behavior",
                    "split": "train_test_split(test_size=0.2, random_state=42)",
                    "feature_count": data.get("features"),
                    "feature_hash": None,
                    "engineering_order": "after_split",
                    "model_class": "LightGBMClassifier",
                    "holdout_auc": data.get("auc") or data.get("routed_auc") or data.get("corrected_auc"),
                    "holdout_f1": None,
                    "holdout_recall": None,
                    "holdout_precision": None,
                    "train_ids_count": 14062,
                    "holdout_ids_count": 3516,
                    "status": "completed",
                    "notes": f"routing_damage={data.get('routing_damage') or data.get('behavior_damage')}",
                })
    
    # ========================================================================
    # Experiment D: Frozen Baseline vs Candidate
    # ========================================================================
    candidate_path = DM_DIR / "study_evolution_publish_candidate.json"
    candidate_meta = load_experiment_metadata("publish_candidate", candidate_path)
    
    if candidate_meta.get("status") != "not_found":
        experiments.append({
            "exp_id": "D-frozen_baseline",
            "exp_name": "Frozen Baseline",
            "population": candidate_meta.get("task_scope", "unknown"),
            "split": "term_order_holdout",
            "feature_count": None,
            "feature_hash": None,
            "engineering_order": "unknown",
            "model_class": candidate_meta.get("model_name", "unknown"),
            "holdout_auc": None,  # Baseline metrics from comparison
            "holdout_f1": None,
            "holdout_recall": None,
            "holdout_precision": None,
            "train_ids_count": None,
            "holdout_ids_count": None,
            "status": "completed",
            "notes": "Frozen snapshot baseline",
        })
        
        experiments.append({
            "exp_id": "D-candidate",
            "exp_name": "Evolution Candidate",
            "population": candidate_meta.get("task_scope", "unknown"),
            "split": "term_order_holdout",
            "feature_count": None,
            "feature_hash": None,
            "engineering_order": "unknown",
            "model_class": candidate_meta.get("model_name", "unknown"),
            "holdout_auc": candidate_meta.get("metrics", {}).get("auc"),
            "holdout_f1": candidate_meta.get("metrics", {}).get("f1"),
            "holdout_recall": candidate_meta.get("metrics", {}).get("recall"),
            "holdout_precision": candidate_meta.get("metrics", {}).get("precision"),
            "train_ids_count": None,
            "holdout_ids_count": None,
            "status": "completed",
            "notes": f"candidate vs baseline AUC delta",
        })
    
    # ========================================================================
    # Experiment E: Current Retrained Model (post-leakage-fix)
    # ========================================================================
    model_config_path = DM_DIR / "study_model_config.json"
    model_metrics_path = DM_DIR / "study_model_metrics.json"
    
    if model_config_path.exists() and model_metrics_path.exists():
        config = json.load(model_config_path.open())
        metrics = json.load(model_metrics_path.open())
        
        experiments.append({
            "exp_id": "E-retrained",
            "exp_name": "Retrained (post-leakage-fix)",
            "population": config.get("data_mode_rules", {}).get("core_available_requires_families", ["unknown"]),
            "split": "term_order_holdout",
            "feature_count": len(config.get("feature_columns", [])),
            "feature_hash": compute_feature_hash(config.get("feature_columns", [])),
            "engineering_order": "after_split",
            "model_class": config.get("primary_model", "unknown"),
            "holdout_auc": metrics.get("core_model", {}).get("valid", {}).get("auc"),
            "holdout_f1": metrics.get("core_model", {}).get("valid", {}).get("f1"),
            "holdout_recall": metrics.get("core_model", {}).get("valid", {}).get("recall"),
            "holdout_precision": metrics.get("core_model", {}).get("valid", {}).get("precision"),
            "train_ids_count": None,
            "holdout_ids_count": None,
            "status": "completed",
            "notes": "Leakage-fixed model",
        })
    
    return pd.DataFrame(experiments)


def main():
    print("Building Unified Experiment Manifest")
    print("=" * 80)
    
    # Build truth table
    truth_table = build_truth_table()
    
    # Sort by experiment ID
    truth_table = truth_table.sort_values("exp_id").reset_index(drop=True)
    
    # Display
    print("\nUnified Experiment Truth Table")
    print("=" * 80)
    
    # Key columns only for display
    display_cols = [
        "exp_id", "exp_name", "population", "feature_count", 
        "engineering_order", "model_class", "holdout_auc", "status"
    ]
    display_cols = [c for c in display_cols if c in truth_table.columns]
    
    print(truth_table[display_cols].to_string(index=False))
    
    # Save to file
    output_path = DM_DIR / "study_unified_experiment_manifest.csv"
    truth_table.to_csv(output_path, index=False)
    print(f"\nManifest saved to: {output_path}")
    
    # Save as JSON for programmatic access
    json_path = DM_DIR / "study_unified_experiment_manifest.json"
    truth_table.to_json(json_path, orient="records", indent=2)
    print(f"JSON saved to: {json_path}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Total experiments: {len(truth_table)}")
    print(f"Completed: {(truth_table['status'] == 'completed').sum()}")
    print(f"Failed: {(truth_table['status'] == 'failed').sum()}")
    print(f"Not found: {(truth_table['status'] == 'not_found').sum()}")
    
    if "holdout_auc" in truth_table.columns:
        auc_exps = truth_table.dropna(subset=["holdout_auc"])
        if len(auc_exps) > 0:
            print(f"\nAUC Range: {auc_exps['holdout_auc'].min():.4f} - {auc_exps['holdout_auc'].max():.4f}")
            print(f"Mean AUC: {auc_exps['holdout_auc'].mean():.4f}")
            print(f"Std AUC: {auc_exps['holdout_auc'].std():.4f}")


if __name__ == "__main__":
    main()
