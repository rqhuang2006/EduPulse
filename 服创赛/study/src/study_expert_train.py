from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ModelBuilder = Callable[[int], tuple[str, Any]]


def _safe_auc(y_true: pd.Series, score: np.ndarray) -> float | None:
    if len(y_true) == 0 or y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, score))


def build_oof_scores(
    x: pd.DataFrame,
    y: pd.Series,
    model_builder: ModelBuilder,
    random_state: int = 42,
    n_splits: int = 5,
) -> pd.Series:
    if x.empty or len(x) < n_splits or y.nunique() < 2:
        return pd.Series(0.5, index=x.index, dtype="float64")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = pd.Series(np.nan, index=x.index, dtype="float64")
    for fold, (train_pos, valid_pos) in enumerate(splitter.split(x, y), start=1):
        _, estimator = model_builder(random_state + fold)
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
        train_index = x.index[train_pos]
        valid_index = x.index[valid_pos]
        model.fit(x.loc[train_index], y.loc[train_index])
        oof.loc[valid_index] = np.asarray(model.predict_proba(x.loc[valid_index]))[:, 1]
    return oof.fillna(0.5)


def train_behavior_corrector(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    base_oof: pd.Series,
    uncertain_mask: pd.Series,
    random_state: int = 42,
) -> tuple[Any | None, list[str], dict[str, Any]]:
    if not feature_columns:
        return None, [], {"train_rows": 0, "valid_rows": 0}

    working = train_df.loc[uncertain_mask].copy()
    if working.empty:
        return None, [], {"train_rows": 0, "valid_rows": 0}

    x = working.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    x["BASE_SCORE_OOF"] = base_oof.loc[working.index].astype(float)
    y = pd.to_numeric(working["LABEL"], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2 or len(y) < 20:
        return None, x.columns.tolist(), {"train_rows": int(len(y)), "valid_rows": 0}

    split_at = max(10, int(len(x) * 0.8))
    train_index = x.index[:split_at]
    valid_index = x.index[split_at:] if split_at < len(x) else x.index[:]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=random_state)),
        ]
    )
    model.fit(x.loc[train_index], y.loc[train_index])
    valid_score = np.asarray(model.predict_proba(x.loc[valid_index]))[:, 1]
    metrics = {
        "train_rows": int(len(train_index)),
        "valid_rows": int(len(valid_index)),
        "valid_auc": _safe_auc(y.loc[valid_index], valid_score),
        "feature_count": int(x.shape[1]),
        "uncertain_training_only": True,
    }
    return model, x.columns.tolist(), metrics


def train_single_fail_expert(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    base_oof: pd.Series,
    uncertain_mask: pd.Series,
    random_state: int = 42,
) -> tuple[Any | None, list[str], dict[str, Any]]:
    if not feature_columns or "LABEL_SUBTYPE" not in train_df.columns:
        return None, [], {"train_rows": 0, "valid_rows": 0}

    subtype_mask = train_df["LABEL_SUBTYPE"].isin(["normal", "single_fail"])
    working = train_df.loc[uncertain_mask & subtype_mask].copy()
    if working.empty:
        return None, [], {"train_rows": 0, "valid_rows": 0}

    x = working.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    x["BASE_SCORE_OOF"] = base_oof.loc[working.index].astype(float)
    y = (working["LABEL_SUBTYPE"].astype(str) == "single_fail").astype(int)
    if y.nunique() < 2 or len(y) < 20:
        return None, x.columns.tolist(), {"train_rows": int(len(y)), "valid_rows": 0}

    split_at = max(10, int(len(x) * 0.8))
    train_index = x.index[:split_at]
    valid_index = x.index[split_at:] if split_at < len(x) else x.index[:]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=random_state)),
        ]
    )
    model.fit(x.loc[train_index], y.loc[train_index])
    valid_score = np.asarray(model.predict_proba(x.loc[valid_index]))[:, 1]
    metrics = {
        "train_rows": int(len(train_index)),
        "valid_rows": int(len(valid_index)),
        "valid_auc": _safe_auc(y.loc[valid_index], valid_score),
        "feature_count": int(x.shape[1]),
        "subtype": "single_fail",
        "uncertain_training_only": True,
    }
    return model, x.columns.tolist(), metrics
