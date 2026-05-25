from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def evaluate_model(y_true_score: pd.Series, y_pred_score: np.ndarray) -> dict[str, Any]:
    y_true = (pd.to_numeric(y_true_score, errors="coerce").fillna(0.0) < 60.0).astype(int)
    y_pred = (pd.Series(y_pred_score).astype(float) < 60.0).astype(int)
    auc: float | None = None
    if y_true.nunique() >= 2:
        try:
            auc = float(roc_auc_score(y_true, -pd.Series(y_pred_score).astype(float)))
        except Exception:
            auc = None
    return {
        "auc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
