from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DM_DIR = ROOT / "data" / "dm"
TRAIN_PATH = DM_DIR / "study_train_table.csv"
OUTPUT_PATH = DM_DIR / "exp_A_baseline_rebuild.json"
TRAIN_IDS_PATH = DM_DIR / "exp_A_baseline_rebuild_train_ids.csv"
HOLDOUT_IDS_PATH = DM_DIR / "exp_A_baseline_rebuild_holdout_ids.csv"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def safe_auc(y_true, y_score) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def build_model():
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


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(TRAIN_PATH)

    train_df = pd.read_csv(TRAIN_PATH)

    core_features = [c for c in train_df.columns if c.startswith("FEATURE_GRADE_") or c.startswith("FEATURE_COURSE_")]
    core_features = [c for c in core_features if train_df[c].notna().mean() > 0.5]

    behavior_features = [
        c
        for c in train_df.columns
        if any(x in c.lower() for x in ["attendance", "library", "assignment", "exam", "class_task"])
    ]
    behavior_features = [c for c in behavior_features if c not in core_features and train_df[c].notna().mean() > 0.1]

    y_all = pd.to_numeric(train_df["LABEL"], errors="coerce").fillna(0).astype(int)
    valid_mask = y_all.notna()
    valid_idx = train_df.index[valid_mask & train_df[core_features].notna().any(axis=1)]

    behavior_nonnull = train_df[behavior_features].notna().sum(axis=1) if behavior_features else pd.Series(0, index=train_df.index)
    data_mode = pd.Series("core_only", index=train_df.index)
    data_mode.loc[behavior_nonnull >= 2] = "core_plus_behavior"
    core_only_idx = valid_idx[data_mode.loc[valid_idx] == "core_only"]

    subset = train_df.loc[core_only_idx].copy()
    y = pd.to_numeric(subset["LABEL"], errors="coerce").fillna(0).astype(int)
    x = subset[core_features].apply(pd.to_numeric, errors="coerce")

    idx_train, idx_valid = train_test_split(x.index, test_size=0.2, random_state=42, stratify=y)
    x_train, x_valid = x.loc[idx_train], x.loc[idx_valid]
    y_train, y_valid = y.loc[idx_train], y.loc[idx_valid]

    model_name, model_params, pipeline = build_model()
    pipeline.fit(x_train, y_train)
    pred = pipeline.predict_proba(x_valid)[:, 1]
    auc = safe_auc(y_valid.values, pred)

    subset.loc[idx_train, ["XH", "TERM_ID"]].to_csv(TRAIN_IDS_PATH, index=False, encoding="utf-8-sig")
    subset.loc[idx_valid, ["XH", "TERM_ID"]].to_csv(HOLDOUT_IDS_PATH, index=False, encoding="utf-8-sig")

    recovered = 0.848 <= auc <= 0.853
    report = {
        "experiment_name": "exp_A_baseline_rebuild",
        "generated_at": now_iso(),
        "goal": "rebuild the strongest historical baseline under fixed core-only configuration",
        "historical_reference_auc": 0.852089926589149,
        "data_source": str(TRAIN_PATH),
        "subset_definition": {
            "population": "core_only",
            "rule": "behavior feature non-null count < 2",
            "rows": int(len(subset)),
            "positive_rows": int(y.sum()),
            "negative_rows": int((y == 0).sum()),
        },
        "split_definition": {
            "method": "train_test_split",
            "test_size": 0.2,
            "random_state": 42,
            "stratify": "LABEL",
            "train_rows": int(len(idx_train)),
            "holdout_rows": int(len(idx_valid)),
            "train_ids_path": str(TRAIN_IDS_PATH),
            "holdout_ids_path": str(HOLDOUT_IDS_PATH),
        },
        "feature_config": {
            "feature_count": len(core_features),
            "feature_list": core_features,
        },
        "model_config": {
            "model_name": model_name,
            "model_params": model_params,
            "threshold_note": "AUC experiment only; threshold intentionally not optimized",
        },
        "results": {
            "holdout_auc": auc,
            "recovered_to_target_band": recovered,
            "interpretation": (
                "strong baseline recovered"
                if recovered
                else "historical baseline not reproduced in current environment"
            ),
        },
    }

    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
