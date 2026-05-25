from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from study_common import DM_DIR, DELIVERABLE_DIR, DWD_DIR, ensure_dirs, term_sort_key, write_json

DIAGNOSTIC_REPORT_PATH = DM_DIR / "study_diagnostic_report.json"
DIAGNOSTIC_DETAIL_PATH = DM_DIR / "study_diagnostic_detail.csv"
DELIVERABLE_MODEL_DIR = DELIVERABLE_DIR / "study" / "model"

EXCLUDED_MODEL_FEATURES = {
    "XH",
    "TERM_ID",
    "LABEL",
    "LABEL_REASON",
    "LABEL_SUBTYPE",
    "NEXT_TERM_ID",
    "PREV_TERM_ID",
    "FEATURE_MISSING_RATE",
    "_SORT",
    "_TERM_SORT_KEY",
    "_ROW_ID",
}


def safe_auc(y_true, scores) -> float:
    yt = np.asarray(y_true, dtype=float)
    sc = np.asarray(scores, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(sc)
    yt, sc = yt[mask], sc[mask]
    if len(yt) == 0 or pd.Series(yt).nunique() < 2:
        return None
    return float(roc_auc_score(yt, sc))


def load_model_bundle() -> tuple[dict[str, Any], Any]:
    config_path = DELIVERABLE_MODEL_DIR / "study_model_config.json"
    model_path = DELIVERABLE_MODEL_DIR / "study_model.pkl"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError("Model bundle not found in deliverables/study/model/. Run 30_train_study_model.py first.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bundle = joblib.load(model_path)
    return config, bundle


def load_data() -> pd.DataFrame:
    train_path = DM_DIR / "study_train_table.csv"
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    return pd.read_csv(train_path)


def reconstruct_split(df: pd.DataFrame, test_size: float = 0.2):
    order = df["TERM_ID"].map(term_sort_key) if "TERM_ID" in df.columns else pd.Series(range(len(df)))
    sorted_idx = order.sort_values().index
    split_at = max(1, int(len(sorted_idx) * (1 - test_size)))
    train_idx, valid_idx = sorted_idx[:split_at], sorted_idx[split_at:]
    return list(train_idx), list(valid_idx)


def classify_label_subtype(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "LABEL" not in frame.columns:
        frame["LABEL_SUBTYPE"] = "unknown"
        return frame

    avg_col = "FEATURE_GRADE_AVG_SCORE" if "FEATURE_GRADE_AVG_SCORE" in frame.columns else None
    fail_col = "FEATURE_GRADE_FAIL_COUNT" if "FEATURE_GRADE_FAIL_COUNT" in frame.columns else None

    if avg_col and fail_col:
        avg = pd.to_numeric(frame[avg_col], errors="coerce")
        fail = pd.to_numeric(frame[fail_col], errors="coerce").fillna(0)
        frame["LABEL_SUBTYPE"] = np.where(
            frame["LABEL"] == 0,
            "normal",
            np.where(avg < 60, "overall_low", "single_fail"),
        )
    else:
        frame["LABEL_SUBTYPE"] = np.where(frame["LABEL"] == 0, "normal", "single_fail")
    return frame


def compute_confidence_zone(scores: np.ndarray, low: float = 0.3, high: float = 0.7) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    return np.where(
        s > high, "high_positive",
        np.where(s < low, "high_negative", "uncertain"),
    )


def safe_predict_proba(pipeline, x: pd.DataFrame) -> np.ndarray:
    """Work around sklearn version incompatibility: manually impute, then call underlying model."""
    try:
        return pipeline.predict_proba(x)[:, 1]
    except AttributeError:
        # sklearn 1.7.2 -> 1.8.0 SimpleImputer compatibility issue
        imputer = pipeline.named_steps.get("imputer")
        model = pipeline.named_steps.get("model")
        if imputer is not None and model is not None:
            # Manually impute with median
            imputed = x.copy()
            for col in imputed.columns:
                col_data = pd.to_numeric(imputed[col], errors="coerce")
                if col_data.isna().all():
                    imputed[col] = 0.0
                else:
                    median_val = col_data.median()
                    imputed[col] = col_data.fillna(median_val)
            if hasattr(model, "predict_proba"):
                return model.predict_proba(imputed)[:, 1]
            return np.asarray(model.predict(imputed), dtype=float)
        raise


def safe_predict_with_model(pipeline, x: pd.DataFrame) -> np.ndarray:
    """Predict with pipeline, handling sklearn version incompatibility and feature count mismatches."""
    try:
        return pipeline.predict_proba(x)[:, 1]
    except (AttributeError, ValueError):
        imputer = pipeline.named_steps.get("imputer")
        model = pipeline.named_steps.get("model")
        if imputer is None or model is None:
            raise

        # Get imputer's known feature names and medians
        feature_names = list(imputer.feature_names_in_)
        medians = imputer.statistics_

        # Determine actual number of model features (may differ from imputer if training bug)
        model_n_features = model.n_features_
        if len(feature_names) > model_n_features:
            # Truncate to match model - drop extra features
            feature_names = feature_names[:model_n_features]
            medians = medians[:model_n_features]

        imputed = pd.DataFrame(index=x.index)
        for i, fname in enumerate(feature_names):
            fill_val = float(medians[i]) if i < len(medians) else 0.0
            if fname in x.columns:
                col_data = pd.to_numeric(x[fname], errors="coerce")
                imputed[fname] = col_data.fillna(fill_val)
            else:
                imputed[fname] = fill_val
        imputed = imputed[feature_names]

        if hasattr(model, "predict_proba"):
            return model.predict_proba(imputed)[:, 1]
        return np.asarray(model.predict(imputed), dtype=float)


def main() -> None:
    ensure_dirs()
    config, bundle = load_model_bundle()
    raw_train = load_data()

    # === Section A: Metric consistency verification ===
    section_a: dict[str, Any] = {}

    # Reconstruct train/valid split
    train_idx, valid_idx = reconstruct_split(raw_train)
    valid_df = raw_train.loc[valid_idx].copy()
    valid_df = valid_df.dropna(subset=["LABEL"])
    valid_df = classify_label_subtype(valid_df)
    y_valid = pd.to_numeric(valid_df["LABEL"], errors="coerce").fillna(0).astype(int)

    # Extract feature columns from config
    core_features = config.get("core_feature_columns", [])
    behavior_features = config.get("behavior_feature_columns", [])
    subgroup_features = config.get("subgroup_feature_columns", [])
    temporal_features = config.get("temporal_feature_columns", [])
    interaction_features = config.get("interaction_feature_columns", [])

    # Ensure all config features exist in valid_df
    for col in core_features + behavior_features + subgroup_features:
        if col not in valid_df.columns:
            valid_df[col] = 0.0

    # Extract feature matrices
    core_x = valid_df.reindex(columns=core_features).apply(pd.to_numeric, errors="coerce")
    behavior_x = valid_df.reindex(columns=behavior_features).apply(pd.to_numeric, errors="coerce")
    subgroup_x = valid_df.reindex(columns=subgroup_features).apply(pd.to_numeric, errors="coerce")

    # Data mode classification
    core_family_map = config.get("core_feature_families", {})
    behavior_family_map = config.get("behavior_feature_families", {})

    def presence(df: pd.DataFrame, cols: list[str]) -> pd.Series:
        usable = [c for c in cols if c in df.columns]
        return df[usable].notna().any(axis=1) if usable else pd.Series(False, index=df.index)

    core_available_mask = presence(valid_df, core_family_map.get("grade", [])) & presence(valid_df, core_family_map.get("course", []))
    behavior_hits = pd.Series(0, index=valid_df.index)
    for cols in behavior_family_map.values():
        behavior_hits = behavior_hits + presence(valid_df, cols).astype(int)
    valid_df["row_level_study_data_mode"] = np.where(
        ~core_available_mask,
        "degraded_sparse",
        np.where(behavior_hits >= 1, "core_plus_behavior", "core_only"),
    )

    # Compute component scores
    core_prob = safe_predict_with_model(bundle["core_model"], core_x)
    behavior_prob = np.full(len(valid_df), np.nan)
    subgroup_prob = np.full(len(valid_df), np.nan)

    behavior_available_rows = valid_df["row_level_study_data_mode"].eq("core_plus_behavior").to_numpy()
    subgroup_available_rows = behavior_available_rows.copy()

    if behavior_features and bundle.get("behavior_model") is not None and behavior_available_rows.any():
        behavior_prob[behavior_available_rows] = safe_predict_with_model(
            bundle["behavior_model"], behavior_x.loc[behavior_available_rows]
        )

    if subgroup_features and bundle.get("subgroup_model") is not None and subgroup_available_rows.any():
        subgroup_prob[subgroup_available_rows] = safe_predict_with_model(
            bundle["subgroup_model"], subgroup_x.loc[subgroup_available_rows]
        )

    # Blend scores
    weights = config.get("score_combination", {})
    core_w = float(weights.get("core_plus_behavior_core_weight", 0.7))
    behavior_w = float(weights.get("core_plus_behavior_behavior_weight", 0.3))
    blend_prob = np.asarray(core_prob, dtype=float).copy()
    blend_prob = np.where(
        behavior_available_rows & ~pd.isna(behavior_prob),
        core_prob * core_w + behavior_prob * behavior_w,
        blend_prob,
    )

    # Routed scores
    routed_prob = np.asarray(core_prob, dtype=float).copy()
    routed_prob = np.where(
        subgroup_available_rows & ~pd.isna(subgroup_prob),
        np.asarray(subgroup_prob, dtype=float),
        routed_prob,
    )
    routed_prob = np.where(
        ~subgroup_available_rows & behavior_available_rows & ~pd.isna(behavior_prob),
        np.asarray(blend_prob, dtype=float),
        routed_prob,
    )
    routed_prob = np.where(
        valid_df["row_level_study_data_mode"].eq("degraded_sparse").to_numpy(),
        np.asarray(core_prob, dtype=float),
        routed_prob,
    )

    # Pathway AUCs on full valid set
    section_a["pathway_auc"] = {
        "core_model": safe_auc(y_valid, core_prob),
        "behavior_module": safe_auc(y_valid[behavior_available_rows], behavior_prob[behavior_available_rows]) if behavior_available_rows.any() else None,
        "subgroup_expert": safe_auc(y_valid[subgroup_available_rows], subgroup_prob[subgroup_available_rows]) if subgroup_available_rows.any() else None,
        "blend_score": safe_auc(y_valid, blend_prob),
        "routed_score": safe_auc(y_valid, routed_prob),
    }

    # Compare with reported metrics
    reported_metrics = config.get("metrics", {})
    section_a["comparison_with_reported"] = {
        "reported_core_valid_auc": reported_metrics.get("core_model", {}).get("valid", {}).get("auc"),
        "computed_core_valid_auc": section_a["pathway_auc"].get("core_model"),
        "reported_behavior_valid_auc": reported_metrics.get("behavior_module", {}).get("valid", {}).get("auc"),
        "reported_subgroup_valid_auc": reported_metrics.get("subgroup_expert", {}).get("valid", {}).get("auc"),
    }

    # Discrepancy note
    computed_core = section_a["pathway_auc"].get("core_model")
    reported_core = section_a["comparison_with_reported"].get("reported_core_valid_auc")
    if computed_core is not None and reported_core is not None:
        delta = abs(computed_core - reported_core)
        if delta > 0.01:
            section_a["discrepancy_note"] = (
                f"Core model AUC discrepancy: computed={computed_core:.4f} vs reported={reported_core:.4f}, delta={delta:.4f}. "
                f"This may be caused by differences in feature computation between training and diagnostic paths, "
                f"or by different validation set compositions. Investigate by checking feature column values."
            )
        else:
            section_a["discrepancy_note"] = (
                f"Core model AUC consistent: computed={computed_core:.4f} vs reported={reported_core:.4f}, delta={delta:.4f}"
            )

    # Per-data-mode AUC
    mode_auc_results = []
    for mode in ["core_only", "core_plus_behavior", "degraded_sparse"]:
        mask = valid_df["row_level_study_data_mode"] == mode
        if mask.sum() < 5:
            continue
        mode_y = y_valid[mask]
        row = {
            "data_mode": mode,
            "rows": int(mask.sum()),
            "positives": int(mode_y.sum()),
            "positive_rate": float(mode_y.mean()),
            "core_auc": safe_auc(mode_y, core_prob[mask]),
            "blend_auc": safe_auc(mode_y, blend_prob[mask]),
            "routed_auc": safe_auc(mode_y, routed_prob[mask]),
        }
        if subgroup_available_rows.any():
            row["subgroup_auc"] = safe_auc(mode_y, subgroup_prob[mask])
        if behavior_available_rows.any():
            row["behavior_auc"] = safe_auc(mode_y, behavior_prob[mask])
        mode_auc_results.append(row)
    section_a["per_data_mode_auc"] = mode_auc_results

    # === Section B: Label subtype analysis ===
    section_b: dict[str, Any] = {}
    if "LABEL_SUBTYPE" in valid_df.columns:
        subtype_dist = valid_df["LABEL_SUBTYPE"].value_counts().to_dict()
        section_b["subtype_distribution"] = {k: int(v) for k, v in subtype_dist.items()}

        # Per-subtype AUC
        subtype_auc_results = []
        for subtype in ["overall_low", "single_fail"]:
            pos_mask = valid_df["LABEL_SUBTYPE"] == subtype
            neg_mask = valid_df["LABEL_SUBTYPE"] == "normal"
            if not pos_mask.any() or not neg_mask.any():
                continue
            combined_mask = pos_mask | neg_mask
            subtype_y = y_valid[combined_mask]
            row = {
                "subtype": subtype,
                "positive_rows": int(pos_mask.sum()),
                "negative_rows": int(neg_mask.sum()),
                "core_auc": safe_auc(subtype_y, core_prob[combined_mask]),
                "blend_auc": safe_auc(subtype_y, blend_prob[combined_mask]),
                "routed_auc": safe_auc(subtype_y, routed_prob[combined_mask]),
            }
            if subgroup_available_rows.any() and (pos_mask[behavior_available_rows] | neg_mask[behavior_available_rows]).any():
                sub_mask = combined_mask & behavior_available_rows
                if sub_mask.sum() >= 5 and pd.Series(subtype_y[sub_mask]).nunique() >= 2:
                    row["subgroup_auc"] = safe_auc(subtype_y[sub_mask], subgroup_prob[sub_mask])
            subtype_auc_results.append(row)
        section_b["per_subtype_auc"] = subtype_auc_results

        # Feature means per subtype
        numeric_feature_cols = [c for c in core_features if c in valid_df.columns]
        section_b["feature_means_by_subtype"] = {}
        for col in numeric_feature_cols[:5]:
            vals = pd.to_numeric(valid_df[col], errors="coerce")
            means = {}
            for st in ["normal", "overall_low", "single_fail"]:
                mask = valid_df["LABEL_SUBTYPE"] == st
                if mask.any():
                    means[st] = float(vals[mask].mean())
            if means:
                section_b["feature_means_by_subtype"][col] = means

    # === Section C: Confidence zone analysis ===
    section_c: dict[str, Any] = {}
    conf_cfg = config.get("confidence_routing", {})
    low_thresh = float(conf_cfg.get("low_threshold", 0.3))
    high_thresh = float(conf_cfg.get("high_threshold", 0.7))

    zones = compute_confidence_zone(core_prob, low_thresh, high_thresh)
    valid_df["CONFIDENCE_ZONE"] = zones

    zone_stats = []
    for zone in ["high_negative", "uncertain", "high_positive"]:
        mask = zones == zone
        if not mask.any():
            continue
        zone_y = y_valid[mask]
        zone_pred = (core_prob[mask] >= 0.5).astype(int)
        tp = int(((zone_pred == 1) & (zone_y == 1)).sum())
        fp = int(((zone_pred == 1) & (zone_y == 0)).sum())
        fn = int(((zone_pred == 0) & (zone_y == 1)).sum())
        tn = int(((zone_pred == 0) & (zone_y == 0)).sum())
        zone_stats.append({
            "zone": zone,
            "sample_count": int(mask.sum()),
            "positive_count": int(zone_y.sum()),
            "positive_rate": float(zone_y.mean()),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn + 1e-9)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        })
    section_c["confidence_zones"] = zone_stats

    # Cross-tab: confidence zone x data mode
    cross_tab = []
    for zone in ["high_negative", "uncertain", "high_positive"]:
        for mode in ["core_only", "core_plus_behavior", "degraded_sparse"]:
            mask = (zones == zone) & (valid_df["row_level_study_data_mode"] == mode)
            if mask.sum() == 0:
                continue
            cross_tab.append({
                "confidence_zone": zone,
                "data_mode": mode,
                "count": int(mask.sum()),
            })
    section_c["confidence_zone_x_data_mode"] = cross_tab

    # Fraction of errors in uncertain zone
    zone_preds = (core_prob >= 0.5).astype(int)
    errors = zone_preds != y_valid.values
    total_errors = int(errors.sum())
    uncertain_errors = int((errors & (zones == "uncertain")).sum())
    section_c["error_distribution"] = {
        "total_errors": total_errors,
        "errors_in_uncertain_zone": uncertain_errors,
        "uncertain_zone_error_rate": float(uncertain_errors / total_errors) if total_errors > 0 else 0,
    }

    # === Build full report ===
    report = {
        "generated_at": datetime.now().isoformat(),
        "valid_set_size": len(valid_df),
        "label_distribution": {
            "positive": int(y_valid.sum()),
            "negative": int((1 - y_valid).sum()),
            "positive_rate": float(y_valid.mean()),
        },
        "section_a_metric_consistency": section_a,
        "section_b_label_subtype": section_b,
        "section_c_confidence_zones": section_c,
    }
    write_json(report, DIAGNOSTIC_REPORT_PATH)
    print(f"Diagnostic report written to {DIAGNOSTIC_REPORT_PATH}")

    # === Write detail CSV ===
    detail = pd.DataFrame({
        "XH": valid_df["XH"],
        "TERM_ID": valid_df["TERM_ID"],
        "LABEL": y_valid.values,
        "LABEL_SUBTYPE": valid_df.get("LABEL_SUBTYPE", pd.Series("unknown", index=valid_df.index)),
        "core_prob": core_prob,
        "behavior_prob": behavior_prob,
        "subgroup_prob": subgroup_prob,
        "blend_prob": blend_prob,
        "routed_prob": routed_prob,
        "row_level_study_data_mode": valid_df["row_level_study_data_mode"],
        "CONFIDENCE_ZONE": zones,
    })
    detail.to_csv(DIAGNOSTIC_DETAIL_PATH, index=False, encoding="utf-8-sig")
    print(f"Diagnostic detail written to {DIAGNOSTIC_DETAIL_PATH}")

    # === Print summary ===
    print("\n=== Diagnostic Summary ===")
    print(f"Valid set: {len(valid_df)} rows, {y_valid.sum()} positive ({y_valid.mean():.1%})")
    for pathway, auc_val in section_a["pathway_auc"].items():
        print(f"  {pathway}: AUC={auc_val:.4f}" if auc_val else f"  {pathway}: N/A")
    print(f"\n  Reported core AUC: {reported_core}")
    print(f"  Computed core AUC: {computed_core}")
    if section_b.get("per_subtype_auc"):
        print("\nPer-subtype core AUC:")
        for item in section_b["per_subtype_auc"]:
            print(f"  {item['subtype']}: AUC={item['core_auc']:.4f} (n={item['positive_rows']} positive)")
    print("\nConfidence zones:")
    for zs in section_c.get("confidence_zones", []):
        print(f"  {zs['zone']}: {zs['sample_count']} rows, {zs['fp']} FP, {zs['fn']} FN")
    err_info = section_c.get("error_distribution", {})
    if err_info.get("total_errors", 0) > 0:
        print(f"  {err_info['uncertain_zone_error_rate']:.1%} of all errors ({err_info['errors_in_uncertain_zone']}/{err_info['total_errors']}) are in uncertain zone")


if __name__ == "__main__":
    main()
