import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =========================
# 基础工具
# =========================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _term_sort_key(term: Any) -> tuple[int, int, int]:
    """
    支持 TERM_ID 形如:
    2023-2024-1
    2023_2024_1
    202320241
    """
    if pd.isna(term):
        return (0, 0, 0)

    s = str(term).strip()

    m = re.match(r"(\d{4})[-_]?(\d{4})[-_]?(\d+)", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    nums = re.findall(r"\d+", s)
    if len(nums) >= 3:
        return (int(nums[0]), int(nums[1]), int(nums[2]))

    if len(nums) == 1:
        n = nums[0]
        if len(n) >= 9:
            return (int(n[:4]), int(n[4:8]), int(n[8:]))

    return (0, 0, 0)


def _find_cols_by_keywords(df: pd.DataFrame, keywords: list[str]) -> list[str]:
    cols = []
    for c in df.columns:
        name = str(c).lower()
        if any(k.lower() in name for k in keywords):
            cols.append(c)
    return cols


def _numeric_feature_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


# =========================
# 特征增强
# =========================

def add_study_enhanced_features(
    df: pd.DataFrame,
    id_col: str = "XH",
    term_col: str = "TERM_ID",
    label_col: str = "label",
) -> pd.DataFrame:
    out = df.copy()

    exclude = {id_col, term_col, label_col}

    # 1) 识别学习域内三类核心字段
    grade_cols = _find_cols_by_keywords(
        out,
        [
            "grade", "score", "gpa", "cj", "成绩", "均分", "分数",
            "feature_grade", "study_grade"
        ],
    )
    course_cols = _find_cols_by_keywords(
        out,
        [
            "course", "credit", "选课", "课程", "学分", "课时",
            "feature_course", "study_course"
        ],
    )
    library_cols = _find_cols_by_keywords(
        out,
        [
            "library", "borrow", "图书馆", "借阅", "visit",
            "feature_library", "study_library"
        ],
    )

    # 去掉主键类
    grade_cols = [c for c in grade_cols if c not in exclude]
    course_cols = [c for c in course_cols if c not in exclude]
    library_cols = [c for c in library_cols if c not in exclude]

    # 2) 如果识别太少，就从数值列里兜底
    num_cols = _numeric_feature_cols(out, exclude)
    if len(grade_cols) == 0:
        grade_cols = [c for c in num_cols if "grade" in str(c).lower() or "score" in str(c).lower()]
    if len(course_cols) == 0:
        course_cols = [c for c in num_cols if "course" in str(c).lower() or "credit" in str(c).lower()]
    if len(library_cols) == 0:
        library_cols = [c for c in num_cols if "library" in str(c).lower() or "borrow" in str(c).lower()]

    # 3) 缺失模式特征
    core_union = sorted(set(grade_cols + course_cols + library_cols))
    if core_union:
        out["STUDY_CORE_MISSING_CNT"] = out[core_union].isna().sum(axis=1)
        out["STUDY_CORE_MISSING_RATIO"] = out[core_union].isna().mean(axis=1)
        out["STUDY_CORE_NON_NULL_CNT"] = out[core_union].notna().sum(axis=1)
    else:
        out["STUDY_CORE_MISSING_CNT"] = 0
        out["STUDY_CORE_MISSING_RATIO"] = 0.0
        out["STUDY_CORE_NON_NULL_CNT"] = 0

    def add_group_stats(prefix: str, cols: list[str]) -> None:
        if not cols:
            out[f"{prefix}_PRESENT"] = 0
            out[f"{prefix}_MEAN"] = 0.0
            out[f"{prefix}_STD"] = 0.0
            out[f"{prefix}_MAX"] = 0.0
            out[f"{prefix}_MIN"] = 0.0
            out[f"{prefix}_MISSING_RATIO"] = 1.0
            return

        block = out[cols]
        out[f"{prefix}_PRESENT"] = 1
        out[f"{prefix}_MEAN"] = block.mean(axis=1, skipna=True)
        out[f"{prefix}_STD"] = block.std(axis=1, skipna=True).fillna(0.0)
        out[f"{prefix}_MAX"] = block.max(axis=1, skipna=True)
        out[f"{prefix}_MIN"] = block.min(axis=1, skipna=True)
        out[f"{prefix}_MISSING_RATIO"] = block.isna().mean(axis=1)

    add_group_stats("GRADE_BLOCK", grade_cols)
    add_group_stats("COURSE_BLOCK", course_cols)
    add_group_stats("LIBRARY_BLOCK", library_cols)

    # 4) 交叉特征：学习表现 × 课程负荷 × 学习行为
    out["GRADE_X_COURSE"] = out["GRADE_BLOCK_MEAN"] * out["COURSE_BLOCK_MEAN"]
    out["GRADE_X_LIBRARY"] = out["GRADE_BLOCK_MEAN"] * out["LIBRARY_BLOCK_MEAN"]
    out["COURSE_X_LIBRARY"] = out["COURSE_BLOCK_MEAN"] * out["LIBRARY_BLOCK_MEAN"]

    out["GRADE_STD_X_LIBRARY"] = out["GRADE_BLOCK_STD"] * out["LIBRARY_BLOCK_MEAN"]
    out["COURSE_LOAD_MINUS_GRADE"] = out["COURSE_BLOCK_MEAN"] - out["GRADE_BLOCK_MEAN"]
    out["COURSE_LOAD_PER_LIBRARY"] = out["COURSE_BLOCK_MEAN"] / (out["LIBRARY_BLOCK_MEAN"].abs() + 1.0)

    # 5) 相对风险型特征：高缺失+低行为组合
    out["LOW_LIBRARY_HIGH_LOAD_FLAG"] = (
        (out["LIBRARY_BLOCK_MEAN"] <= out["LIBRARY_BLOCK_MEAN"].median())
        & (out["COURSE_BLOCK_MEAN"] >= out["COURSE_BLOCK_MEAN"].median())
    ).astype(int)

    out["LOW_GRADE_HIGH_LOAD_FLAG"] = (
        (out["GRADE_BLOCK_MEAN"] <= out["GRADE_BLOCK_MEAN"].median())
        & (out["COURSE_BLOCK_MEAN"] >= out["COURSE_BLOCK_MEAN"].median())
    ).astype(int)

    # 6) 按学生历史做 term-level lag / delta 特征
    if id_col in out.columns and term_col in out.columns:
        out["_TERM_SORT_KEY_"] = out[term_col].map(_term_sort_key)
        out = out.sort_values([id_col, "_TERM_SORT_KEY_"]).reset_index(drop=True)

        lag_base_cols = [
            "GRADE_BLOCK_MEAN",
            "GRADE_BLOCK_STD",
            "COURSE_BLOCK_MEAN",
            "LIBRARY_BLOCK_MEAN",
            "STUDY_CORE_MISSING_RATIO",
        ]

        for c in lag_base_cols:
            out[f"{c}_PREV"] = out.groupby(id_col)[c].shift(1)
            out[f"{c}_DELTA"] = out[c] - out[f"{c}_PREV"]

        out["HAS_PREV_TERM"] = out.groupby(id_col).cumcount().gt(0).astype(int)
        out = out.drop(columns=["_TERM_SORT_KEY_"])

    # 7) 把 inf 清掉
    out = out.replace([np.inf, -np.inf], np.nan)

    return out


# =========================
# 评估与阈值搜索
# =========================

def _search_threshold(
    y_true: np.ndarray,
    prob: np.ndarray,
    recall_floor: float = 0.0,
) -> tuple[float, dict[str, float]]:
    best_t = 0.5
    best_score = -1e18
    best_metrics = {"f1": 0.0, "recall": 0.0, "precision": 0.0}

    for t in np.linspace(0.20, 0.80, 61):
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)

        penalty = 0.0
        if recall < recall_floor:
            penalty = (recall_floor - recall) * 2.0

        score = 0.55 * f1 + 0.35 * recall + 0.10 * precision - penalty
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_metrics = {"f1": float(f1), "recall": float(recall), "precision": float(precision)}

    return best_t, best_metrics


def _evaluate_binary(
    y_true: np.ndarray,
    prob: np.ndarray,
    recall_floor: float = 0.0,
) -> dict[str, float]:
    auc = roc_auc_score(y_true, prob)
    threshold, th_metrics = _search_threshold(y_true, prob, recall_floor=recall_floor)

    pred = (prob >= threshold).astype(int)
    return {
        "auc": float(auc),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "threshold": float(threshold),
        "coverage": 1.0,
        "degraded_proxy": 0.0,
    }


def _composite_score(m: dict[str, float]) -> float:
    # 这里故意不是只看 AUC
    return (
        0.40 * _safe_float(m.get("auc"))
        + 0.25 * _safe_float(m.get("recall"))
        + 0.25 * _safe_float(m.get("f1"))
        + 0.10 * _safe_float(m.get("precision"))
        - 0.05 * _safe_float(m.get("degraded_proxy"))
    )


# =========================
# 模型池
# =========================

def _make_model_pool(random_state: int = 42) -> dict[str, Any]:
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    et = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        l2_regularization=0.1,
        random_state=random_state,
    )

    lr = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    rf_cal = CalibratedClassifierCV(estimator=rf, method="sigmoid", cv=3)
    et_cal = CalibratedClassifierCV(estimator=et, method="sigmoid", cv=3)

    stack = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("hgb", Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", hgb)])),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    pool = {
        "random_forest": rf,
        "extra_trees": et,
        "hist_gbdt": Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", hgb)]),
        "logistic_balanced": lr,
        "rf_calibrated": Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", rf_cal)]),
        "et_calibrated": Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", et_cal)]),
        "stacking_v1": Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", stack)]),
    }

    # 可选：如果环境里装了 lightgbm / catboost，就自动加进来
    try:
        from lightgbm import LGBMClassifier
        pool["lightgbm_v2"] = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.2,
            reg_lambda=0.5,
            class_weight="balanced",
            random_state=random_state,
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier
        pool["catboost_v2"] = CatBoostClassifier(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            eval_metric="AUC",
            loss_function="Logloss",
            random_seed=random_state,
            verbose=False,
            auto_class_weights="Balanced",
        )
    except Exception:
        pass

    return pool


# =========================
# 训练主流程
# =========================

@dataclass
class CandidateSummary:
    version_id: str
    model_name: str
    threshold: float
    metrics: dict[str, float]
    composite_score: float
    model_path: str
    feature_columns_path: str
    comparison_path: str
    selection_path: str


def train_study_candidate_pool(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    label_col: str = "label",
    id_col: str = "XH",
    term_col: str = "TERM_ID",
    out_dir: str = "data/dm",
    baseline_recall: float = 0.0,
    random_state: int = 42,
) -> CandidateSummary:
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_df = add_study_enhanced_features(train_df, id_col=id_col, term_col=term_col, label_col=label_col)
    valid_df = add_study_enhanced_features(valid_df, id_col=id_col, term_col=term_col, label_col=label_col)

    exclude = {label_col, id_col, term_col}
    feature_cols = [c for c in train_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])]
    feature_cols = [c for c in feature_cols if c in valid_df.columns]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].astype(int).values

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[label_col].astype(int).values

    pool = _make_model_pool(random_state=random_state)

    compare_rows = []
    best = None
    best_artifact = None
    best_model = None

    for model_name, model in pool.items():
        try:
            model.fit(X_train, y_train)

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_valid)[:, 1]
            else:
                raw = model.decision_function(X_valid)
                prob = 1 / (1 + np.exp(-raw))

            metrics = _evaluate_binary(y_valid, prob, recall_floor=baseline_recall)
            score = _composite_score(metrics)

            row = {
                "model_name": model_name,
                "auc": metrics["auc"],
                "f1": metrics["f1"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "threshold": metrics["threshold"],
                "coverage": metrics["coverage"],
                "degraded_proxy": metrics["degraded_proxy"],
                "composite_score": score,
            }
            compare_rows.append(row)

            if best is None or score > best["composite_score"]:
                best = row
                best_model = model
        except Exception as e:
            compare_rows.append(
                {
                    "model_name": model_name,
                    "auc": np.nan,
                    "f1": np.nan,
                    "recall": np.nan,
                    "precision": np.nan,
                    "threshold": np.nan,
                    "coverage": np.nan,
                    "degraded_proxy": np.nan,
                    "composite_score": -999.0,
                    "error": str(e),
                }
            )

    if best_model is None:
        raise RuntimeError("all candidate models failed in study candidate pool")

    version_id = f"study_candidate_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
    model_path = os.path.join(out_dir, f"{version_id}.joblib")
    feature_columns_path = os.path.join(out_dir, f"{version_id}_feature_columns.json")
    comparison_path = os.path.join(out_dir, "study_evolution_comparison.csv")
    selection_path = os.path.join(out_dir, "study_evolution_selection.json")

    joblib.dump(best_model, model_path)
    with open(feature_columns_path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    compare_df = pd.DataFrame(compare_rows).sort_values("composite_score", ascending=False)
    compare_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    selection = {
        "version_id": version_id,
        "best_model_name": best["model_name"],
        "selection_rule": "study_best_model_v1",
        "metrics": {
            "auc": best["auc"],
            "f1": best["f1"],
            "recall": best["recall"],
            "precision": best["precision"],
            "coverage": best["coverage"],
            "degraded_proxy": best["degraded_proxy"],
        },
        "threshold": best["threshold"],
        "composite_score": best["composite_score"],
        "model_path": model_path,
        "feature_columns_path": feature_columns_path,
        "comparison_path": comparison_path,
    }
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)

    return CandidateSummary(
        version_id=version_id,
        model_name=best["model_name"],
        threshold=best["threshold"],
        metrics={
            "auc": best["auc"],
            "f1": best["f1"],
            "recall": best["recall"],
            "precision": best["precision"],
            "coverage": best["coverage"],
            "degraded_proxy": best["degraded_proxy"],
        },
        composite_score=best["composite_score"],
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        comparison_path=comparison_path,
        selection_path=selection_path,
    )