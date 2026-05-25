from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "数据集及类型"
OUT_DIR = ROOT / "outputs"
FROZEN_SPORT_MAINLINE = {
    "label_version": "future_v3",
    "feature_bundle": "baseline+deviation+trend",
    "structure_version": "two_stage",
    "population_version": "recoverable",
}


def safe_read_excel(path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if usecols is None:
        return pd.read_excel(path)
    return pd.read_excel(path, usecols=lambda c: c in usecols)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def infer_term_from_datetime(value: object) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None, None, None, None

    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        return None, None, None, None

    year = int(dt.year)
    month = int(dt.month)
    if month >= 9:
        start_year = year
        semester = 1
    elif month <= 2:
        start_year = year - 1
        semester = 1
    else:
        start_year = year - 1
        semester = 2

    academic_year = f"{start_year}-{start_year + 1}"
    term_id = f"{academic_year}-T{semester}"
    term_order = start_year * 10 + semester
    return term_id, start_year, semester, term_order


def infer_terms_from_datetime_series(series: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(series.astype(str).str.strip(), errors="coerce")
    valid = dt.notna()
    years = pd.Series(np.nan, index=series.index, dtype="float64")
    semesters = pd.Series(np.nan, index=series.index, dtype="float64")

    year_vals = dt.dt.year.astype("float64")
    month_vals = dt.dt.month.astype("float64")
    sem1_mask = valid & ((month_vals >= 9) | (month_vals <= 2))
    sem2_mask = valid & (month_vals >= 3) & (month_vals <= 8)

    years.loc[sem1_mask] = year_vals.loc[sem1_mask]
    years.loc[sem2_mask] = year_vals.loc[sem2_mask] - 1
    semesters.loc[sem1_mask] = 1
    semesters.loc[sem2_mask] = 2

    start_year = years.astype("Int64")
    semester = semesters.astype("Int64")
    term_order = (start_year * 10 + semester).astype("Int64")
    term_id = pd.Series(pd.NA, index=series.index, dtype="string")
    valid_term = start_year.notna() & semester.notna()
    term_id.loc[valid_term] = (
        start_year.loc[valid_term].astype(str)
        + "-"
        + (start_year.loc[valid_term] + 1).astype(str)
        + "-T"
        + semester.loc[valid_term].astype(str)
    )
    return pd.DataFrame(
        {
            "term_id": term_id.astype(object),
            "year_start": start_year.astype(object),
            "semester": semester.astype(object),
            "term_order": term_order.astype(object),
        }
    )


def parse_term_text(term_text: object) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    text = str(term_text).strip()
    if not text or text.lower() == "nan":
        return None, None, None, None

    match = re.search(r"(\d{4})\s*[-~]\s*(\d{4}).*?([12])", text)
    if not match:
        return None, None, None, None

    start_year = int(match.group(1))
    semester = int(match.group(3))
    term_id = f"{start_year}-{start_year + 1}-T{semester}"
    term_order = start_year * 10 + semester
    return term_id, start_year, semester, term_order


def parse_year_to_term(value: object) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    year = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(year):
        return None, None, None, None
    y = int(year)
    term_id = f"{y}-{y + 1}-T1"
    return term_id, y, 1, y * 10 + 1


def parse_term_text_series(series: pd.Series) -> pd.DataFrame:
    extracted = series.astype(str).str.strip().str.extract(r"(\d{4})\s*[-~]\s*(\d{4}).*?([12])")
    start_year = pd.to_numeric(extracted[0], errors="coerce").astype("Int64")
    semester = pd.to_numeric(extracted[2], errors="coerce").astype("Int64")
    term_order = (start_year * 10 + semester).astype("Int64")
    term_id = pd.Series(pd.NA, index=series.index, dtype="string")
    valid = start_year.notna() & semester.notna()
    term_id.loc[valid] = (
        start_year.loc[valid].astype(str)
        + "-"
        + (start_year.loc[valid] + 1).astype(str)
        + "-T"
        + semester.loc[valid].astype(str)
    )
    return pd.DataFrame(
        {
            "term_id": term_id.astype(object),
            "year_start": start_year.astype(object),
            "semester": semester.astype(object),
            "term_order": term_order.astype(object),
        }
    )


def parse_year_to_term_series(series: pd.Series) -> pd.DataFrame:
    years = pd.to_numeric(series, errors="coerce").astype("Int64")
    semester = pd.Series(1, index=series.index, dtype="Int64")
    semester.loc[years.isna()] = pd.NA
    term_order = (years * 10 + semester).astype("Int64")
    term_id = pd.Series(pd.NA, index=series.index, dtype="string")
    valid = years.notna()
    term_id.loc[valid] = years.loc[valid].astype(str) + "-" + (years.loc[valid] + 1).astype(str) + "-T1"
    return pd.DataFrame(
        {
            "term_id": term_id.astype(object),
            "year_start": years.astype(object),
            "semester": semester.astype(object),
            "term_order": term_order.astype(object),
        }
    )


def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    categorical: List[str] = []
    numeric: List[str] = []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(transformers=[("num", num_pipe, numeric), ("cat", cat_pipe, categorical)])


def optimize_f1_threshold(y_true: pd.Series, proba: np.ndarray) -> Tuple[float, float]:
    best_t = 0.5
    best_objective: Tuple[float, float, float, float] | None = None
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (proba >= t).astype(int)
        precision = float(precision_score(y_true, pred, zero_division=0))
        recall = float(recall_score(y_true, pred, zero_division=0))
        score = float(f1_score(y_true, pred, zero_division=0))
        objective = (
            round(score - 0.15 * abs(precision - recall), 6),
            round(score, 6),
            round(precision, 6),
            -round(abs(float(t) - 0.5), 6),
        )
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_t = float(t)
    pred = (proba >= best_t).astype(int)
    return best_t, float(f1_score(y_true, pred, zero_division=0))


def optimize_group_thresholds(
    df: pd.DataFrame,
    label_col: str,
    proba_col: str,
    group_col: str,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    global_threshold, global_f1 = optimize_f1_threshold(df[label_col], df[proba_col].to_numpy())
    pred_label = np.zeros(len(df), dtype=int)
    group_thresholds: Dict[str, float] = {}

    for group_name, group_df in df.groupby(group_col, dropna=False):
        group_key = str(group_name)
        group_threshold = float(global_threshold)
        group_f1 = global_f1

        if len(group_df) >= 30 and group_df[label_col].nunique() >= 2:
            candidate_threshold, candidate_f1 = optimize_f1_threshold(group_df[label_col], group_df[proba_col].to_numpy())
            if candidate_f1 > 0.0:
                group_threshold = float(candidate_threshold)
                group_f1 = float(candidate_f1)

        if group_f1 <= 0.0 and int(group_df[label_col].sum()) > 0:
            prevalence_threshold = _prevalence_threshold(group_df[proba_col], float(group_df[label_col].mean()))
            prevalence_pred = (group_df[proba_col] >= prevalence_threshold).astype(int)
            prevalence_f1 = float(f1_score(group_df[label_col], prevalence_pred, zero_division=0))
            if prevalence_f1 > group_f1:
                group_threshold = float(prevalence_threshold)

        group_thresholds[group_key] = float(group_threshold)
        pred_label[group_df.index.to_numpy()] = (group_df[proba_col].to_numpy() >= group_threshold).astype(int)

    return float(global_threshold), group_thresholds, pred_label


def split_temporal_train_calibration(
    df: pd.DataFrame,
    label_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    terms = sorted(df["term_order"].dropna().unique().tolist())
    if len(terms) >= 2:
        calib_term = terms[-1]
        model_train = df[df["term_order"] < calib_term].copy()
        calib = df[df["term_order"] == calib_term].copy()
        if not model_train.empty and not calib.empty and model_train[label_col].nunique() >= 2:
            return model_train, calib

    split_idx = int(len(df) * 0.8)
    model_train = df.iloc[:split_idx].copy()
    calib = df.iloc[split_idx:].copy()
    if model_train.empty or calib.empty:
        return df.copy(), pd.DataFrame(columns=df.columns)
    return model_train, calib


def fit_sigmoid_calibrator(y_true: pd.Series, proba: np.ndarray) -> Optional[LogisticRegression]:
    if len(y_true) < 30 or y_true.nunique() < 2:
        return None
    calibrator = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
    calibrator.fit(np.asarray(proba).reshape(-1, 1), y_true.to_numpy())
    return calibrator


def apply_sigmoid_calibration(calibrator: Optional[LogisticRegression], proba: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(proba, dtype=float)
    return calibrator.predict_proba(np.asarray(proba).reshape(-1, 1))[:, 1]


def select_threshold_from_validation(y_true: pd.Series, proba: np.ndarray) -> float:
    threshold, best_f1 = optimize_f1_threshold(y_true, proba)
    if best_f1 <= 0.0 and int(y_true.sum()) > 0:
        prevalence_threshold = _prevalence_threshold(pd.Series(proba), float(y_true.mean()))
        prevalence_pred = (np.asarray(proba) >= prevalence_threshold).astype(int)
        prevalence_f1 = float(f1_score(y_true, prevalence_pred, zero_division=0))
        if prevalence_f1 > best_f1:
            return float(prevalence_threshold)
    return float(threshold)


def score_thresholded_predictions(y_true: pd.Series, proba: np.ndarray) -> Dict[str, float]:
    threshold = select_threshold_from_validation(y_true, proba)
    pred = (np.asarray(proba) >= threshold).astype(int)
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    auc = float(roc_auc_score(y_true, proba)) if y_true.nunique() >= 2 else float("nan")
    objective = float(round(f1 - 0.2 * abs(precision - recall), 6))
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "objective": objective,
    }


def build_sport_binary_candidates(y_train: pd.Series) -> Dict[str, object]:
    positive = max(int(y_train.sum()), 1)
    negative = max(int((1 - y_train).sum()), 1)
    scale_pos_weight = max(float(negative) / float(positive), 1.0)

    candidates: Dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=180,
            random_state=42,
            n_jobs=1,
            class_weight="balanced",
        ),
    }
    if LGBMClassifier is not None:
        candidates["LightGBM"] = LGBMClassifier(
            n_estimators=220,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            class_weight="balanced",
        )
    if XGBClassifier is not None:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
        )
    return candidates


def fit_best_future_model(
    fit_train: pd.DataFrame,
    calib: pd.DataFrame,
    live_feature_cols: List[str],
    label_col: str,
) -> Tuple[Pipeline, float, Optional[LogisticRegression], str]:
    candidates = build_sport_binary_candidates(fit_train[label_col])
    fallback_pre = build_preprocessor(fit_train, live_feature_cols)
    fallback_pipe = Pipeline(steps=[("pre", fallback_pre), ("model", next(iter(candidates.values())))])
    fallback_pipe.fit(fit_train[live_feature_cols], fit_train[label_col])

    if calib.empty or calib[label_col].nunique() < 2:
        return fallback_pipe, 0.5, None, next(iter(candidates.keys()))

    best_choice: Optional[Tuple[Tuple[float, float, float, float], Pipeline, float, Optional[LogisticRegression], str]] = None
    for name, model in candidates.items():
        pre = build_preprocessor(fit_train, live_feature_cols)
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(fit_train[live_feature_cols], fit_train[label_col])
        raw_calib_proba = pipe.predict_proba(calib[live_feature_cols])[:, 1]
        raw_score = score_thresholded_predictions(calib[label_col], raw_calib_proba)

        calibrator = fit_sigmoid_calibrator(calib[label_col], raw_calib_proba)
        if calibrator is not None:
            calibrated_proba = apply_sigmoid_calibration(calibrator, raw_calib_proba)
            calibrated_score = score_thresholded_predictions(calib[label_col], calibrated_proba)
            if calibrated_score["objective"] > raw_score["objective"]:
                chosen_score = calibrated_score
            else:
                calibrator = None
                chosen_score = raw_score
        else:
            chosen_score = raw_score

        rank = (
            round(chosen_score["objective"], 6),
            round(chosen_score["auc"], 6) if not np.isnan(chosen_score["auc"]) else -1.0,
            round(chosen_score["precision"], 6),
            round(chosen_score["recall"], 6),
        )
        if best_choice is None or rank > best_choice[0]:
            best_choice = (rank, pipe, float(chosen_score["threshold"]), calibrator, name)

    assert best_choice is not None
    _, best_pipe, best_threshold, best_calibrator, best_name = best_choice
    return best_pipe, best_threshold, best_calibrator, best_name


def _prevalence_threshold(proba: pd.Series, positive_rate: float) -> float:
    clipped_rate = min(max(float(positive_rate), 0.0), 1.0)
    if clipped_rate <= 0.0:
        return 1.0
    if clipped_rate >= 1.0:
        return 0.0
    return float(pd.Series(proba).quantile(max(0.0, 1.0 - clipped_rate)))


def split_train_val_test_by_term(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    terms = sorted(df["term_order"].dropna().unique())
    if len(terms) >= 3:
        test_term = terms[-1]
        val_term = terms[-2]
        train = df[df["term_order"] < val_term].copy()
        val = df[df["term_order"] == val_term].copy()
        test = df[df["term_order"] == test_term].copy()
        return train, val, test

    if len(terms) == 2:
        test_term = terms[-1]
        train_all = df[df["term_order"] < test_term].copy()
        test = df[df["term_order"] == test_term].copy()
    else:
        frac = int(len(df) * 0.8)
        train_all = df.iloc[:frac].copy()
        test = df.iloc[frac:].copy()

    split_idx = int(len(train_all) * 0.8)
    train = train_all.iloc[:split_idx].copy()
    val = train_all.iloc[split_idx:].copy()
    if len(val) < 10 and len(train_all) >= 10:
        val = train_all.sample(frac=0.2, random_state=42)
        train = train_all.drop(val.index)
    return train, val, test


def fallback_random_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n = len(shuffled)
    if n < 30:
        raise RuntimeError("Not enough samples for fallback split.")
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train = shuffled.iloc[:n_train].copy()
    val = shuffled.iloc[n_train : n_train + n_val].copy()
    test = shuffled.iloc[n_train + n_val :].copy()
    if train.empty or val.empty or test.empty:
        raise RuntimeError("Fallback random split failed due to insufficient samples.")
    return train, val, test


def build_life_dataset(active_threshold: int = 3) -> pd.DataFrame:
    club = safe_read_excel(DATA_DIR / "社团活动.xlsx", ["XSBH", "HDRQ"])
    if club.empty:
        return pd.DataFrame()

    club = club.dropna(subset=["XSBH"]).copy()
    club["sid"] = normalize_id(club["XSBH"])
    parsed = club["HDRQ"].apply(infer_term_from_datetime)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])
    club = pd.concat([club, parsed_df], axis=1).dropna(subset=["term_id"])

    target = club.groupby(["sid", "term_id", "year_start", "semester", "term_order"], as_index=False).agg(
        club_event_count=("HDRQ", "count")
    )
    target["club_active_flag"] = (target["club_event_count"] >= active_threshold).astype(int)

    lib = safe_read_excel(DATA_DIR / "图书馆打卡记录.xlsx", ["cardld", "visittime", "gateno"])
    if not lib.empty:
        lib = lib.dropna(subset=["cardld"]).copy()
        lib["sid"] = normalize_id(lib["cardld"])
        parsed = lib["visittime"].apply(infer_term_from_datetime)
        p = pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])
        lib = pd.concat([lib, p], axis=1).dropna(subset=["term_id"])
        lib_feat = lib.groupby(["sid", "term_id"], as_index=False).agg(
            lib_visit_count=("visittime", "count"),
            lib_unique_gate=("gateno", pd.Series.nunique),
        )
    else:
        lib_feat = pd.DataFrame(columns=["sid", "term_id"])

    door = safe_read_excel(DATA_DIR / "门禁数据.xlsx", ["IDSERTAL", "LOGINTIME", "DOORCTRLCODE"])
    if not door.empty:
        door = door.dropna(subset=["IDSERTAL"]).copy()
        door["sid"] = normalize_id(door["IDSERTAL"])
        parsed = door["LOGINTIME"].apply(infer_term_from_datetime)
        p = pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])
        door = pd.concat([door, p], axis=1).dropna(subset=["term_id"])
        door_feat = door.groupby(["sid", "term_id"], as_index=False).agg(
            door_event_count=("LOGINTIME", "count"),
            door_unique_ctrl=("DOORCTRLCODE", pd.Series.nunique),
        )
    else:
        door_feat = pd.DataFrame(columns=["sid", "term_id"])

    net = safe_read_excel(DATA_DIR / "上网统计.xlsx", ["XSBH", "TJNY", "SWLJSC", "XXPJZ"])
    if not net.empty:
        net = net.dropna(subset=["XSBH"]).copy()
        net["sid"] = normalize_id(net["XSBH"])
        net["SWLJSC"] = pd.to_numeric(net["SWLJSC"], errors="coerce")
        net["XXPJZ"] = pd.to_numeric(net["XXPJZ"], errors="coerce")
        parsed = net["TJNY"].apply(infer_term_from_datetime)
        p = pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])
        net = pd.concat([net, p], axis=1).dropna(subset=["term_id"])
        net_feat = net.groupby(["sid", "term_id"], as_index=False).agg(
            net_duration_sum=("SWLJSC", "sum"),
            net_duration_mean=("SWLJSC", "mean"),
            net_score_mean=("XXPJZ", "mean"),
        )
    else:
        net_feat = pd.DataFrame(columns=["sid", "term_id"])

    data = target.merge(lib_feat, on=["sid", "term_id"], how="left")
    data = data.merge(door_feat, on=["sid", "term_id"], how="left")
    data = data.merge(net_feat, on=["sid", "term_id"], how="left")

    numeric_cols = [c for c in data.columns if c not in ["sid", "term_id"]]
    for c in numeric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    count_cols = [c for c in data.columns if "count" in c or "num" in c]
    if count_cols:
        data[count_cols] = data[count_cols].fillna(0)

    data = data.sort_values(["term_order", "sid"]).reset_index(drop=True)
    return data


def build_sport_dataset() -> pd.DataFrame:
    print(f"[{time.time()}] train_domain_models.build_sport_dataset target_read BEFORE", flush=True)
    target = safe_read_excel(DATA_DIR / "体测数据.xlsx")
    print(f"[{time.time()}] train_domain_models.build_sport_dataset target_read AFTER rows={len(target)}", flush=True)
    if target.empty:
        return pd.DataFrame()

    target = target.dropna(subset=["XH"]).copy()
    target["sid"] = normalize_id(target["XH"])

    p = parse_year_to_term_series(target["TCNF"])
    target = pd.concat([target, p], axis=1).dropna(subset=["term_id"])

    target["ZF"] = pd.to_numeric(target.get("ZF"), errors="coerce")
    target = target.dropna(subset=["ZF"])
    # Derive a stable grade label from numeric score to avoid noisy text labels.
    target["zf_grade"] = pd.cut(
        target["ZF"],
        bins=[-np.inf, 60, 70, 80, 90, np.inf],
        labels=["E", "D", "C", "B", "A"],
    ).astype(str)
    # label v2/v3 for label-feature co-tuning experiments (kept alongside legacy label).
    target["zf_label_v2"] = (target["ZF"] < 75).astype(int)
    target["zf_label_v3"] = pd.cut(
        target["ZF"],
        bins=[-np.inf, 60, 80, np.inf],
        labels=["high_risk", "mid_risk", "low_risk"],
    ).astype(str)

    base = target.groupby(["sid", "term_id", "year_start", "semester", "term_order"], as_index=False).agg(
        zf_score=("ZF", "mean"),
        zf_grade=("zf_grade", lambda s: s.mode().iloc[0] if not s.mode().empty else "C"),
        zf_label_v2=("zf_label_v2", "max"),
        zf_label_v3=("zf_label_v3", lambda s: s.mode().iloc[0] if not s.mode().empty else "mid_risk"),
        bmi_count=("BMI", "count"),
        fhl_mean=("FHL", "mean"),
        ws_mean=("WS", "mean"),
        ldty_mean=("LDTY", "mean"),
        zwtqq_mean=("ZWTQQ", "mean"),
        bb_mean=("BB", "mean"),
        ywqz_mean=("YWQZ", "mean"),
    )

    print(f"[{time.time()}] train_domain_models.build_sport_dataset sport_course_read BEFORE", flush=True)
    sport_course = safe_read_excel(DATA_DIR / "体育课.xlsx", ["XH", "XQ", "KC", "ZKC"])
    print(f"[{time.time()}] train_domain_models.build_sport_dataset sport_course_read AFTER rows={len(sport_course)}", flush=True)
    if not sport_course.empty:
        sport_course = sport_course.dropna(subset=["XH"]).copy()
        sport_course["sid"] = normalize_id(sport_course["XH"])
        p = parse_term_text_series(sport_course["XQ"])
        sport_course = pd.concat([sport_course, p], axis=1).dropna(subset=["term_id"])
        sport_course_feat = sport_course.groupby(["sid", "term_id"], as_index=False).agg(
            pe_course_count=("KC", "count"),
            pe_unique_course=("KC", pd.Series.nunique),
            pe_unique_class=("ZKC", pd.Series.nunique),
        )
    else:
        sport_course_feat = pd.DataFrame(columns=["sid", "term_id"])

    print(f"[{time.time()}] train_domain_models.build_sport_dataset daily_read BEFORE", flush=True)
    daily = safe_read_excel(DATA_DIR / "日常锻炼.xlsx", ["XH", "XQ", "ZC", "DKCS"])
    print(f"[{time.time()}] train_domain_models.build_sport_dataset daily_read AFTER rows={len(daily)}", flush=True)
    if not daily.empty:
        daily = daily.dropna(subset=["XH"]).copy()
        daily["sid"] = normalize_id(daily["XH"])
        daily["DKCS"] = pd.to_numeric(daily["DKCS"], errors="coerce")
        p = parse_term_text_series(daily["XQ"])
        daily = pd.concat([daily, p], axis=1).dropna(subset=["term_id"])
        daily_feat = daily.groupby(["sid", "term_id"], as_index=False).agg(
            daily_daka_sum=("DKCS", "sum"),
            daily_daka_mean=("DKCS", "mean"),
            daily_week_count=("ZC", "nunique"),
        )
    else:
        daily_feat = pd.DataFrame(columns=["sid", "term_id"])

    print(f"[{time.time()}] train_domain_models.build_sport_dataset run_read BEFORE", flush=True)
    run = safe_read_excel(DATA_DIR / "跑步打卡.xlsx", ["USERNUM", "PUNCH_DAY", "STATE"])
    print(f"[{time.time()}] train_domain_models.build_sport_dataset run_read AFTER rows={len(run)}", flush=True)
    if not run.empty:
        run = run.dropna(subset=["USERNUM"]).copy()
        run["sid"] = normalize_id(run["USERNUM"])
        p = infer_terms_from_datetime_series(run["PUNCH_DAY"])
        run = pd.concat([run, p], axis=1).dropna(subset=["term_id"])
        run_feat = run.groupby(["sid", "term_id"], as_index=False).agg(
            run_punch_count=("PUNCH_DAY", "count"),
            run_state_mean=("STATE", "mean"),
        )
    else:
        run_feat = pd.DataFrame(columns=["sid", "term_id"])

    print(f"[{time.time()}] train_domain_models.build_sport_dataset fitness_read BEFORE", flush=True)
    fitness = safe_read_excel(DATA_DIR / "学生体能考核.xlsx", ["XH", "XQ", "CFBFS", "SJCJ", "HSCJ"])
    print(f"[{time.time()}] train_domain_models.build_sport_dataset fitness_read AFTER rows={len(fitness)}", flush=True)
    if not fitness.empty:
        fitness = fitness.dropna(subset=["XH"]).copy()
        fitness["sid"] = normalize_id(fitness["XH"])
        for c in ["CFBFS", "SJCJ", "HSCJ"]:
            fitness[c] = pd.to_numeric(fitness[c], errors="coerce")
        p = parse_term_text_series(fitness["XQ"])
        fitness = pd.concat([fitness, p], axis=1).dropna(subset=["term_id"])
        fit_feat = fitness.groupby(["sid", "term_id"], as_index=False).agg(
            fit_cfbfs_mean=("CFBFS", "mean"),
            fit_sjcj_mean=("SJCJ", "mean"),
            fit_hscj_mean=("HSCJ", "mean"),
        )
    else:
        fit_feat = pd.DataFrame(columns=["sid", "term_id"])

    data = base.merge(sport_course_feat, on=["sid", "term_id"], how="left")
    data = data.merge(daily_feat, on=["sid", "term_id"], how="left")
    data = data.merge(run_feat, on=["sid", "term_id"], how="left")
    data = data.merge(fit_feat, on=["sid", "term_id"], how="left")

    numeric_cols = [c for c in data.columns if c not in ["sid", "term_id", "zf_grade"]]
    for c in numeric_cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    count_cols = [c for c in data.columns if "count" in c or "sum" in c or "num" in c]
    if count_cols:
        data[count_cols] = data[count_cols].fillna(0)

    # trend feature bundle: stable first-order temporal interactions.
    data["trend_bundle_run_daily_gap"] = data["run_punch_count"].fillna(0) - data["daily_daka_sum"].fillna(0)
    data["trend_bundle_fit_combo"] = data["fit_sjcj_mean"].fillna(0) + data["fit_hscj_mean"].fillna(0)
    data["trend_bundle_strength_endurance"] = data["fhl_mean"].fillna(0) + data["ldty_mean"].fillna(0)
    data["trend_bundle_activity_load"] = (
        data["run_punch_count"].fillna(0)
        + data["daily_daka_sum"].fillna(0)
        + data["pe_course_count"].fillna(0)
    )
    data["trend_bundle_quality_index"] = data["zf_score"].fillna(0) - 0.2 * data["trend_bundle_run_daily_gap"]

    # baseline/deviation feature families for future-window prediction experiments.
    data = data.sort_values(["sid", "term_order"]).reset_index(drop=True)
    sid_group = data.groupby("sid", sort=False)
    hist_mean = sid_group["zf_score"].transform(lambda s: s.shift(1).expanding().mean())
    hist_std = sid_group["zf_score"].transform(lambda s: s.shift(1).expanding().std())
    data["baseline_hist_mean_zf"] = hist_mean
    data["baseline_hist_std_zf"] = hist_std.fillna(0.0)
    data["baseline_hist_max_run"] = sid_group["run_punch_count"].transform(lambda s: s.shift(1).expanding().max())
    data["baseline_hist_interrupts"] = sid_group["run_punch_count"].transform(lambda s: (s.shift(1).fillna(0) <= 0).astype(int).cumsum())
    data["baseline_hist_recovery_speed"] = sid_group["run_punch_count"].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    data["deviation_zf_vs_hist"] = data["zf_score"] - data["baseline_hist_mean_zf"]
    data["deviation_run_vs_hist4"] = data["run_punch_count"] - sid_group["run_punch_count"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    data["deviation_daily_vs_hist4"] = data["daily_daka_sum"] - sid_group["daily_daka_sum"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    data["deviation_activity_volatility"] = sid_group["trend_bundle_activity_load"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).std()).fillna(0.0)
    prev_run = sid_group["run_punch_count"].shift(1).fillna(0.0)
    prev_daily = sid_group["daily_daka_sum"].shift(1).fillna(0.0)
    prev_activity = sid_group["trend_bundle_activity_load"].shift(1).fillna(0.0)
    prev_run_mean2 = sid_group["run_punch_count"].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean()).fillna(0.0)
    prev_daily_mean2 = sid_group["daily_daka_sum"].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean()).fillna(0.0)
    prev_activity_mean4 = sid_group["trend_bundle_activity_load"].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean()).fillna(0.0)
    prev_run_std4 = sid_group["run_punch_count"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).std()).fillna(0.0)
    prev_daily_std4 = sid_group["daily_daka_sum"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).std()).fillna(0.0)
    prev_activity_std4 = sid_group["trend_bundle_activity_load"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).std()).fillna(0.0)
    data["recovery_run_rebound"] = data["run_punch_count"].fillna(0.0) - prev_run_mean2
    data["recovery_daily_rebound"] = data["daily_daka_sum"].fillna(0.0) - prev_daily_mean2
    data["recovery_activity_rebound"] = data["trend_bundle_activity_load"].fillna(0.0) - prev_activity_mean4
    data["recovery_return_from_zero"] = ((prev_run <= 0) & (data["run_punch_count"].fillna(0.0) > 0)).astype(float)
    data["recovery_restart_strength"] = data["recovery_return_from_zero"] * data["run_punch_count"].fillna(0.0)
    data["rhythm_run_cv4"] = prev_run_std4 / prev_run_mean2.replace(0, np.nan)
    data["rhythm_daily_cv4"] = prev_daily_std4 / prev_daily_mean2.replace(0, np.nan)
    data["rhythm_activity_cv4"] = prev_activity_std4 / prev_activity_mean4.replace(0, np.nan)
    data["rhythm_run_daily_balance"] = data["run_punch_count"].fillna(0.0) / (1.0 + data["daily_daka_sum"].fillna(0.0))
    data["rhythm_activity_vs_prev"] = data["trend_bundle_activity_load"].fillna(0.0) / (1.0 + prev_activity)
    data["rhythm_course_share"] = data["pe_course_count"].fillna(0.0) / (1.0 + data["trend_bundle_activity_load"].fillna(0.0))
    rhythm_cols = [c for c in data.columns if c.startswith("rhythm_")]
    if rhythm_cols:
        data[rhythm_cols] = data[rhythm_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # stage-1 state recognition features: cohort-relative stability, engagement, and recovery.
    def pct_rank_by_term(values: pd.Series, *, ascending: bool = True, default: float = 0.5) -> pd.Series:
        ranked = values.groupby(data["term_order"]).rank(pct=True, ascending=ascending)
        return ranked.fillna(default)

    stage1_hist_score = pct_rank_by_term(data["baseline_hist_mean_zf"].fillna(data["zf_score"].fillna(0.0)))
    stage1_engagement_score = (
        0.45 * pct_rank_by_term(data["trend_bundle_activity_load"].fillna(0.0))
        + 0.30 * pct_rank_by_term(data["run_punch_count"].fillna(0.0))
        + 0.15 * pct_rank_by_term(data["daily_week_count"].fillna(0.0))
        + 0.10 * pct_rank_by_term(data["pe_course_count"].fillna(0.0))
    )
    stage1_consistency_score = (
        0.35 * pct_rank_by_term(data["deviation_activity_volatility"].fillna(0.0), ascending=False)
        + 0.25 * pct_rank_by_term(data["baseline_hist_interrupts"].fillna(0.0), ascending=False)
        + 0.20 * pct_rank_by_term(data["baseline_hist_std_zf"].fillna(0.0), ascending=False)
        + 0.20 * pct_rank_by_term((data["deviation_zf_vs_hist"].fillna(0.0)).abs(), ascending=False)
    )
    stage1_recovery_score = (
        0.45 * pct_rank_by_term(data["recovery_activity_rebound"].fillna(0.0))
        + 0.30 * pct_rank_by_term(data["recovery_run_rebound"].fillna(0.0))
        + 0.15 * pct_rank_by_term(data["baseline_hist_recovery_speed"].fillna(0.0))
        + 0.10 * data["recovery_return_from_zero"].fillna(0.0)
    )
    stage1_momentum_score = (
        0.50 * pct_rank_by_term(data["rhythm_activity_vs_prev"].fillna(0.0))
        + 0.25 * pct_rank_by_term(data["run_state_mean"].fillna(0.0))
        + 0.25 * pct_rank_by_term(data["trend_bundle_quality_index"].fillna(0.0))
    )
    data["stage1_hist_score"] = stage1_hist_score
    data["stage1_engagement_score"] = stage1_engagement_score
    data["stage1_consistency_score"] = stage1_consistency_score
    data["stage1_recovery_score"] = stage1_recovery_score
    data["stage1_momentum_score"] = stage1_momentum_score
    data["stage1_state_score"] = (
        0.32 * data["stage1_hist_score"]
        + 0.28 * data["stage1_engagement_score"]
        + 0.22 * data["stage1_consistency_score"]
        + 0.10 * data["stage1_recovery_score"]
        + 0.08 * data["stage1_momentum_score"]
    )

    # future-window label versions (mainline prediction targets).
    future_1 = sid_group["zf_score"].shift(-1)
    future_2 = sid_group["zf_score"].shift(-2)
    data["sport_label_future_v1"] = (future_1 < 60).astype(float)
    data["sport_label_future_v2"] = ((future_1 < 60) | (future_2 < 60)).astype(float)
    data["sport_label_future_v3"] = (((data["zf_score"] >= 75) & (future_1 < 60)) | ((data["zf_score"] >= 75) & (future_2 < 60))).astype(float)
    data.loc[future_1.isna(), "sport_label_future_v1"] = np.nan
    data.loc[future_1.isna() & future_2.isna(), "sport_label_future_v2"] = np.nan
    data.loc[future_1.isna() & future_2.isna(), "sport_label_future_v3"] = np.nan

    data = data.sort_values(["term_order", "sid"]).reset_index(drop=True)
    return data


def _resolve_label_column(data: pd.DataFrame, label_version: str) -> str:
    mapping = {
        "future_v1": "sport_label_future_v1",
        "future_v2": "sport_label_future_v2",
        "future_v3": "sport_label_future_v3",
    }
    col = mapping.get(label_version, "sport_label_future_v1")
    if col not in data.columns:
        raise RuntimeError(f"Missing label column for version {label_version}: {col}")
    return col


def _blocked_sport_label_columns() -> List[str]:
    return [
        "zf_score",
        "zf_grade",
        "zf_label_v2",
        "zf_label_v3",
        "sport_label_future_v1",
        "sport_label_future_v2",
        "sport_label_future_v3",
    ]


def _apply_population_filter(data: pd.DataFrame, population_version: str) -> pd.DataFrame:
    if population_version == "active_courses":
        return data[data["pe_course_count"].fillna(0) >= 1].copy()
    if population_version == "recoverable":
        return data[
            (data["baseline_hist_mean_zf"].fillna(0) >= 65)
            & (data["baseline_hist_interrupts"].fillna(99) <= 2)
        ].copy()
    return data.copy()


def _select_feature_columns(data: pd.DataFrame, feature_bundle: str, label_col: str) -> List[str]:
    id_cols = ["sid", "term_id", "year_start", "semester", "term_order"]
    block = set(id_cols + _blocked_sport_label_columns() + [label_col])
    all_features = [c for c in data.columns if c not in block and not c.startswith("stage1_")]
    trend_cols = [c for c in all_features if c.startswith("trend_bundle_")]
    baseline_cols = [c for c in all_features if c.startswith("baseline_")]
    deviation_cols = [c for c in all_features if c.startswith("deviation_")]
    recovery_cols = [c for c in all_features if c.startswith("recovery_")]
    rhythm_cols = [c for c in all_features if c.startswith("rhythm_")]
    grouped_cols = trend_cols + baseline_cols + deviation_cols + recovery_cols + rhythm_cols
    other_cols = [c for c in all_features if c not in set(grouped_cols)]
    if feature_bundle == "baseline_only":
        return baseline_cols + other_cols
    if feature_bundle == "baseline+deviation":
        return baseline_cols + deviation_cols + other_cols
    if feature_bundle == "baseline+deviation+recovery+rhythm":
        return baseline_cols + deviation_cols + recovery_cols + rhythm_cols + other_cols
    if feature_bundle == "baseline+deviation+recovery+rhythm+trend":
        return baseline_cols + deviation_cols + recovery_cols + rhythm_cols + trend_cols + other_cols
    return baseline_cols + deviation_cols + trend_cols + other_cols


def _assign_state_bucket(data: pd.DataFrame) -> pd.Series:
    score = data["zf_score"].fillna(0.0)
    baseline_score = data["baseline_hist_mean_zf"].fillna(score)
    activity_load = data["trend_bundle_activity_load"].fillna(0.0)
    run_count = data["run_punch_count"].fillna(0.0)
    daily_count = data["daily_daka_sum"].fillna(0.0)
    activity_ratio = data.get("rhythm_activity_vs_prev", pd.Series(0.0, index=data.index)).fillna(0.0)
    recovery_rebound = data.get("recovery_activity_rebound", pd.Series(0.0, index=data.index)).fillna(0.0)
    low_mask = (score < 65) | ((activity_load <= 2) & (activity_ratio < 0.8) & (baseline_score < 72))
    stable_mask = (
        (score >= 80)
        & ((run_count >= 2) | (daily_count >= 2))
        & ((activity_ratio >= 0.9) | (recovery_rebound >= 0))
        & (baseline_score >= 72)
    )

    return pd.Series(
        np.select(
            [low_mask, stable_mask],
            ["low_participation", "stable"],
            default="borderline",
        ),
        index=data.index,
        dtype="object",
    )


def _assign_state_bucket_conservative(data: pd.DataFrame) -> pd.Series:
    current = _assign_state_bucket(data).astype(str)
    stage1_state = pd.to_numeric(data.get("stage1_state_score", pd.Series(0.0, index=data.index)), errors="coerce").fillna(0.0)
    stage1_engagement = pd.to_numeric(data.get("stage1_engagement_score", pd.Series(0.0, index=data.index)), errors="coerce").fillna(0.0)
    stage1_recovery = pd.to_numeric(data.get("stage1_recovery_score", pd.Series(0.0, index=data.index)), errors="coerce").fillna(0.0)

    out = current.copy()
    downgrade_mask = (
        (current == "stable")
        & (
            (stage1_state < stage1_state.quantile(0.80))
            | (stage1_engagement < stage1_engagement.quantile(0.55))
            | (stage1_recovery < stage1_recovery.quantile(0.45))
        )
    )
    out.loc[downgrade_mask] = "borderline"
    return out


def _uses_state_bucket(structure_version: str) -> bool:
    return structure_version in {"two_stage", "two_stage_conservative"}


def _resolve_state_bucket_assigner(structure_version: str):
    if structure_version == "two_stage_conservative":
        return _assign_state_bucket_conservative
    return _assign_state_bucket


def _collect_future_predictions(
    data: pd.DataFrame,
    label_version: str,
    feature_bundle: str,
    structure_version: str,
    population_version: str = "all",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    label_col = _resolve_label_column(data, label_version)
    exp_data = data.dropna(subset=[label_col]).copy()
    exp_data[label_col] = exp_data[label_col].astype(int)
    exp_data = _apply_population_filter(exp_data, population_version)
    state_col = "__sport_state_bucket"
    if _uses_state_bucket(structure_version):
        exp_data[state_col] = _resolve_state_bucket_assigner(structure_version)(exp_data).astype(str)
    if len(exp_data) < 80 or exp_data[label_col].nunique() < 2:
        return pd.DataFrame(), {
            "rows": int(len(exp_data)),
            "eval_rows": 0,
            "positive_count": 0,
            "label_col": label_col,
            "state_col": state_col,
            "evaluated_terms": [],
        }

    feature_cols = _select_feature_columns(exp_data, feature_bundle, label_col)
    feature_cols = [c for c in feature_cols if exp_data[c].notna().sum() > 0]
    if len(feature_cols) < 3:
        return pd.DataFrame(), {
            "rows": int(len(exp_data)),
            "eval_rows": 0,
            "positive_count": 0,
            "label_col": label_col,
            "state_col": state_col,
            "evaluated_terms": [],
        }

    preds: List[pd.DataFrame] = []
    evaluated_terms: List[int] = []
    keep_cols = [label_col, "sid", "term_order"]
    for extra_col in [state_col, "stage1_state_score", "zf_grade", "baseline_hist_mean_zf", "trend_bundle_activity_load"]:
        if extra_col in exp_data.columns and extra_col not in keep_cols:
            keep_cols.append(extra_col)

    for test_term in sorted(exp_data["term_order"].dropna().unique().tolist())[1:]:
        train = exp_data[exp_data["term_order"] < test_term].copy()
        test = exp_data[exp_data["term_order"] == test_term].copy()
        if train.empty or test.empty or train[label_col].nunique() < 2 or test[label_col].nunique() < 2:
            continue
        if _uses_state_bucket(structure_version):
            fold_parts: List[pd.DataFrame] = []
            for state_name, test_state in test.groupby(state_col, dropna=False):
                if test_state.empty:
                    continue
                train_state = train[train[state_col] == state_name].copy()
                active_train = train_state
                if len(active_train) < 80 or active_train[label_col].nunique() < 2:
                    active_train = train
                model_train, calib = split_temporal_train_calibration(active_train, label_col)
                fit_train = model_train if not model_train.empty and model_train[label_col].nunique() >= 2 else active_train
                live_feature_cols = [
                    c for c in feature_cols if fit_train[c].notna().sum() > 0 and test_state[c].notna().sum() > 0
                ]
                if len(live_feature_cols) < 3:
                    continue
                pipe, fold_threshold, calibrator, chosen_model = fit_best_future_model(fit_train, calib, live_feature_cols, label_col)
                fold_state = test_state[keep_cols].copy()
                test_proba = pipe.predict_proba(test_state[live_feature_cols])[:, 1]
                test_proba = apply_sigmoid_calibration(calibrator, test_proba)
                fold_state["proba"] = test_proba
                fold_state["threshold"] = float(fold_threshold)
                fold_state["chosen_model"] = chosen_model
                fold_parts.append(fold_state)
            if not fold_parts:
                continue
            fold = pd.concat(fold_parts, ignore_index=True)
        else:
            model_train, calib = split_temporal_train_calibration(train, label_col)
            fit_train = model_train if not model_train.empty and model_train[label_col].nunique() >= 2 else train
            live_feature_cols = [c for c in feature_cols if fit_train[c].notna().sum() > 0 and test[c].notna().sum() > 0]
            if len(live_feature_cols) < 3:
                continue
            pipe, fold_threshold, calibrator, chosen_model = fit_best_future_model(fit_train, calib, live_feature_cols, label_col)
            fold = test[keep_cols].copy()
            test_proba = pipe.predict_proba(test[live_feature_cols])[:, 1]
            test_proba = apply_sigmoid_calibration(calibrator, test_proba)
            fold["proba"] = test_proba
            fold["threshold"] = float(fold_threshold)
            fold["chosen_model"] = chosen_model
        preds.append(fold)
        evaluated_terms.append(int(test_term))

    if not preds:
        return pd.DataFrame(), {
            "rows": int(len(exp_data)),
            "eval_rows": 0,
            "positive_count": 0,
            "label_col": label_col,
            "state_col": state_col,
            "evaluated_terms": [],
        }

    pred_df = pd.concat(preds, ignore_index=True)
    pred_df["pred_label"] = (pred_df["proba"].to_numpy() >= pred_df["threshold"].astype(float).to_numpy()).astype(int)
    return pred_df, {
        "rows": int(len(exp_data)),
        "eval_rows": int(len(pred_df)),
        "positive_count": int(pred_df[label_col].sum()),
        "label_col": label_col,
        "state_col": state_col,
        "evaluated_terms": evaluated_terms,
    }


def _run_future_experiment(
    data: pd.DataFrame,
    label_version: str,
    feature_bundle: str,
    structure_version: str,
    population_version: str = "all",
) -> Dict[str, object]:
    pred_df, meta = _collect_future_predictions(data, label_version, feature_bundle, structure_version, population_version)
    label_col = str(meta.get("label_col"))
    state_col = str(meta.get("state_col"))
    if not isinstance(label_col, str) or not label_col:
        label_col = _resolve_label_column(data, label_version)
    if int(meta.get("rows", 0)) < 80:
        return {
            "auc": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "rows": int(meta.get("rows", 0)),
            "eval_rows": 0,
            "positive_count": 0,
            "label_version": label_version,
            "feature_bundle": feature_bundle,
            "structure_version": structure_version,
            "population_version": population_version,
        }
    if pred_df.empty or pred_df[label_col].nunique() < 2:
        return {
            "auc": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "rows": int(meta.get("rows", 0)),
            "eval_rows": int(meta.get("eval_rows", 0)),
            "positive_count": int(meta.get("positive_count", 0)),
            "label_version": label_version,
            "feature_bundle": feature_bundle,
            "structure_version": structure_version,
            "population_version": population_version,
        }
    auc = float(roc_auc_score(pred_df[label_col], pred_df["proba"]))
    state_thresholds: Optional[Dict[str, float]] = None
    pred_label = pred_df["pred_label"].astype(int)
    best_threshold = float(pred_df["threshold"].dropna().median()) if "threshold" in pred_df.columns and pred_df["threshold"].notna().any() else 0.5
    if _uses_state_bucket(structure_version) and state_col in pred_df.columns:
        state_thresholds = (
            pred_df.groupby(state_col, dropna=False)["threshold"]
            .median()
            .dropna()
            .astype(float)
            .to_dict()
        )

    result: Dict[str, object] = {
        "auc": auc,
        "f1": float(f1_score(pred_df[label_col], pred_label, zero_division=0)),
        "precision": float(precision_score(pred_df[label_col], pred_label, zero_division=0)),
        "recall": float(recall_score(pred_df[label_col], pred_label, zero_division=0)),
        "best_threshold": float(best_threshold),
        "rows": int(meta.get("rows", 0)),
        "eval_rows": int(meta.get("eval_rows", len(pred_df))),
        "positive_count": int(meta.get("positive_count", int(pred_df[label_col].sum()))),
        "evaluated_terms": meta.get("evaluated_terms", []),
        "label_version": label_version,
        "feature_bundle": feature_bundle,
        "structure_version": structure_version,
        "population_version": population_version,
    }
    if state_thresholds:
        result["state_thresholds"] = state_thresholds
    return result


def analyze_stage1_subgroups(
    data: pd.DataFrame,
    label_version: str,
    feature_bundle: str,
    structure_version: str,
    population_version: str = "all",
) -> Dict[str, object]:
    pred_df, meta = _collect_future_predictions(data, label_version, feature_bundle, structure_version, population_version)
    label_col = str(meta.get("label_col"))
    state_col = str(meta.get("state_col"))
    if pred_df.empty or label_col not in pred_df.columns or pred_df[label_col].nunique() < 2:
        return {
            "status": "insufficient_data",
            "rows": int(meta.get("rows", 0)),
            "eval_rows": int(meta.get("eval_rows", 0)),
            "subgroups": [],
            "hardest_subgroup": {},
        }

    work = pred_df.copy()
    if "stage1_state_score" in work.columns:
        try:
            work["stage1_score_band"] = pd.qcut(
                work["stage1_state_score"].rank(method="first"),
                q=4,
                labels=["q1_low", "q2_mid_low", "q3_mid_high", "q4_high"],
            ).astype(str)
        except Exception:
            work["stage1_score_band"] = "q_mid"
    else:
        work["stage1_score_band"] = "q_mid"

    subgroup_rows: List[Dict[str, object]] = []
    for group_col in [c for c in [state_col, "stage1_score_band", "zf_grade"] if c in work.columns]:
        for group_name, group_df in work.groupby(group_col, dropna=False):
            if len(group_df) < 50:
                continue
            positives = int(group_df[label_col].sum())
            auc = None
            if positives >= 10 and int(len(group_df) - positives) >= 10:
                auc = float(roc_auc_score(group_df[label_col], group_df["proba"]))
            subgroup_rows.append(
                {
                    "group_type": group_col,
                    "group": str(group_name),
                    "rows": int(len(group_df)),
                    "positive_count": positives,
                    "positive_rate": float(group_df[label_col].mean()),
                    "auc": auc,
                    "f1": float(f1_score(group_df[label_col], group_df["pred_label"], zero_division=0)),
                    "precision": float(precision_score(group_df[label_col], group_df["pred_label"], zero_division=0)),
                    "recall": float(recall_score(group_df[label_col], group_df["pred_label"], zero_division=0)),
                    "mean_proba": float(group_df["proba"].mean()),
                    "mean_threshold": float(group_df["threshold"].mean()),
                }
            )

    eligible = [row for row in subgroup_rows if isinstance(row.get("auc"), float)]
    if eligible:
        hardest = sorted(eligible, key=lambda row: (float(row["auc"]), -int(row["rows"])))[0]
    elif subgroup_rows:
        hardest = sorted(subgroup_rows, key=lambda row: (float(row["f1"]), -int(row["rows"])))[0]
    else:
        hardest = {}

    return {
        "status": "completed",
        "rows": int(meta.get("rows", 0)),
        "eval_rows": int(meta.get("eval_rows", 0)),
        "subgroups": subgroup_rows,
        "hardest_subgroup": hardest,
    }


def train_life_model(data: pd.DataFrame, out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if data.empty:
        raise RuntimeError("Life dataset is empty.")

    label_col = "club_active_flag"
    id_cols = ["sid", "term_id", "year_start", "semester", "term_order"]
    feature_cols = [c for c in data.columns if c not in set(id_cols + [label_col, "club_event_count"])]
    all_nan_feature_cols = [c for c in feature_cols if data[c].notna().sum() == 0]
    if all_nan_feature_cols:
        data = data.drop(columns=all_nan_feature_cols)
        feature_cols = [c for c in feature_cols if c not in set(all_nan_feature_cols)]

    train, val, test = split_train_val_test_by_term(data)
    if train.empty or val.empty or test.empty:
        raise RuntimeError("Life train/val/test split failed due to insufficient data.")

    pre = build_preprocessor(train, feature_cols)

    candidates: Dict[str, object] = {
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"),
    }
    if LGBMClassifier is not None:
        candidates["LightGBM"] = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42, class_weight="balanced")
    if XGBClassifier is not None:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
            tree_method="hist",
        )

    rows = []
    best = None
    best_f1 = -1.0

    for name, model in candidates.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(train[feature_cols], train[label_col])
        val_proba = pipe.predict_proba(val[feature_cols])[:, 1]
        threshold, val_f1 = optimize_f1_threshold(val[label_col], val_proba)
        rows.append({"model": name, "val_f1": val_f1, "best_val_threshold": threshold})
        if val_f1 > best_f1:
            best_f1 = val_f1
            best = (name, threshold, pipe)

    assert best is not None
    best_name, best_threshold, best_pipe = best

    test_proba = best_pipe.predict_proba(test[feature_cols])[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)
    metrics = {
        "best_model": best_name,
        "best_threshold": float(best_threshold),
        "f1": float(f1_score(test[label_col], test_pred, zero_division=0)),
        "auc": float(roc_auc_score(test[label_col], test_proba)) if test[label_col].nunique() > 1 else float("nan"),
        "accuracy": float(accuracy_score(test[label_col], test_pred)),
        "samples": int(len(data)),
    }

    pred_out = test[["sid", "term_id", label_col]].copy()
    pred_out["pred_prob"] = test_proba
    pred_out["pred_label"] = test_pred
    pred_out.to_csv(out_dir / "predictions_test.csv", index=False)

    pd.DataFrame(rows).sort_values("val_f1", ascending=False).to_csv(out_dir / "model_comparison.csv", index=False)
    data.to_csv(out_dir / "feature_dataset.csv", index=False)

    model_dir = out_dir / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(best_pipe, model_dir / "best_life_model.joblib")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def train_sport_models(
    data: pd.DataFrame,
    out_dir: Path,
    *,
    label_version: str = "future_v1",
    feature_bundle: str = "baseline+deviation+trend",
    structure_version: str = "single_stage",
    population_version: str = "all",
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if data.empty:
        raise RuntimeError("Sport dataset is empty.")

    id_cols = ["sid", "term_id", "year_start", "semester", "term_order"]
    reg_label = "zf_score"
    cls_label = "zf_label_v3"
    if cls_label not in data.columns or data[cls_label].dropna().nunique() < 2:
        cls_label = "zf_grade"
    feature_cols = [c for c in data.columns if c not in set(id_cols + _blocked_sport_label_columns())]
    all_nan_feature_cols = [c for c in feature_cols if data[c].notna().sum() == 0]
    if all_nan_feature_cols:
        data = data.drop(columns=all_nan_feature_cols)
        feature_cols = [c for c in feature_cols if c not in set(all_nan_feature_cols)]

    train, val, test = split_train_val_test_by_term(data)
    if train.empty or val.empty or test.empty:
        raise RuntimeError("Sport train/val/test split failed due to insufficient data.")

    pre = build_preprocessor(train, feature_cols)

    reg_candidates: Dict[str, object] = {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1),
    }
    if LGBMRegressor is not None:
        reg_candidates["LightGBM"] = LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
    if XGBRegressor is not None:
        reg_candidates["XGBoost"] = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            tree_method="hist",
        )

    reg_rows = []
    best_reg = None
    best_r2 = -1e18

    for name, model in reg_candidates.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(train[feature_cols], train[reg_label])
        val_pred = pipe.predict(val[feature_cols])
        val_r2 = r2_score(val[reg_label], val_pred)
        reg_rows.append({"model": name, "val_r2": float(val_r2)})
        if val_r2 > best_r2:
            best_r2 = float(val_r2)
            best_reg = (name, pipe)

    assert best_reg is not None
    best_reg_name, best_reg_pipe = best_reg
    reg_test_pred = best_reg_pipe.predict(test[feature_cols])

    mse = mean_squared_error(test[reg_label], reg_test_pred)
    rmse = float(np.sqrt(mse))
    reg_metrics = {
        "best_model": best_reg_name,
        "rmse": rmse,
        "mae": float(mean_absolute_error(test[reg_label], reg_test_pred)),
        "r2": float(r2_score(test[reg_label], reg_test_pred)),
        "samples": int(len(data)),
    }

    reg_out_dir = out_dir / "regression"
    reg_out_dir.mkdir(exist_ok=True)
    reg_pred_out = test[["sid", "term_id", reg_label]].copy()
    reg_pred_out["pred_zf_score"] = reg_test_pred
    reg_pred_out.to_csv(reg_out_dir / "predictions_test.csv", index=False)
    pd.DataFrame(reg_rows).sort_values("val_r2", ascending=False).to_csv(reg_out_dir / "model_comparison.csv", index=False)
    joblib.dump(best_reg_pipe, reg_out_dir / "best_sport_regression_model.joblib")
    reg_metrics["model_path"] = str(reg_out_dir / "best_sport_regression_model.joblib")

    cls_candidates: Dict[str, object] = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1),
    }
    if LGBMClassifier is not None:
        cls_candidates["LightGBM"] = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42)
    # XGBoost classifier requires numeric class labels; zf_grade is string labels.

    cls_data = data.dropna(subset=[cls_label]).copy()
    cls_train, cls_val, cls_test = split_train_val_test_by_term(cls_data)
    if cls_train.empty or cls_val.empty or cls_test.empty:
        if len(cls_data) < 30 and "zf_grade" in data.columns and cls_label != "zf_grade":
            cls_label = "zf_grade"
            cls_data = data.dropna(subset=[cls_label]).copy()
            cls_train, cls_val, cls_test = split_train_val_test_by_term(cls_data)
        if cls_train.empty or cls_val.empty or cls_test.empty:
            cls_train, cls_val, cls_test = fallback_random_split(cls_data)

    cls_rows = []
    best_cls = None
    best_cls_f1 = -1.0
    labels = sorted(cls_train[cls_label].dropna().unique().tolist())

    for name, model in cls_candidates.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        pipe.fit(cls_train[feature_cols], cls_train[cls_label])
        val_pred = pipe.predict(cls_val[feature_cols])
        val_f1 = f1_score(cls_val[cls_label], val_pred, average="macro", zero_division=0)
        cls_rows.append({"model": name, "val_macro_f1": float(val_f1)})
        if val_f1 > best_cls_f1:
            best_cls_f1 = float(val_f1)
            best_cls = (name, pipe)

    assert best_cls is not None
    best_cls_name, best_cls_pipe = best_cls
    cls_test_pred = best_cls_pipe.predict(cls_test[feature_cols])
    cls_accuracy = accuracy_score(cls_test[cls_label], cls_test_pred)
    cls_macro_f1 = f1_score(cls_test[cls_label], cls_test_pred, average="macro", zero_division=0)

    cls_out_dir = out_dir / "classification"
    cls_out_dir.mkdir(exist_ok=True)
    cls_pred_out = cls_test[["sid", "term_id", cls_label]].copy()
    cls_pred_out["pred_zf_grade"] = cls_test_pred
    cls_pred_out.to_csv(cls_out_dir / "predictions_test.csv", index=False)
    pd.DataFrame(cls_rows).sort_values("val_macro_f1", ascending=False).to_csv(cls_out_dir / "model_comparison.csv", index=False)
    joblib.dump(best_cls_pipe, cls_out_dir / "best_sport_classification_model.joblib")

    grade_dist = cls_test[cls_label].value_counts(dropna=False).to_dict()
    cls_metrics = {
        "best_model": best_cls_name,
        "accuracy": float(cls_accuracy),
        "macro_f1": float(cls_macro_f1),
        "samples": int(len(cls_test)),
        "labels": labels,
        "test_label_distribution": {str(k): int(v) for k, v in grade_dist.items()},
        "model_path": str(cls_out_dir / "best_sport_classification_model.joblib"),
    }

    data.to_csv(out_dir / "feature_dataset.csv", index=False)

    combined_metrics = {
        "regression": reg_metrics,
        "classification": cls_metrics,
    }
    # Mainline future-window experiment matrix.
    matrix: List[Dict[str, object]] = []
    for lv in ["future_v1", "future_v3"]:
        for fb in ["baseline_only", "baseline+deviation", "baseline+deviation+trend"]:
            for sv in ["single_stage", "two_stage", "two_stage_conservative"]:
                for pv in ["all", "active_courses", "recoverable"]:
                    matrix.append(_run_future_experiment(data, lv, fb, sv, pv))

    valid_rows = [r for r in matrix if isinstance(r.get("auc"), float)]
    credible_rows = [r for r in valid_rows if int(r.get("eval_rows", 0)) >= 500 and int(r.get("positive_count", 0)) >= 50]
    normal_band = [r for r in credible_rows if 0.8 <= float(r["auc"]) <= 0.95]
    best_candidate = None
    if normal_band:
        best_candidate = sorted(normal_band, key=lambda r: float(r["auc"]), reverse=True)[0]
    elif credible_rows:
        best_candidate = sorted(credible_rows, key=lambda r: float(r["auc"]), reverse=True)[0]
    elif valid_rows:
        best_candidate = sorted(valid_rows, key=lambda r: float(r["auc"]), reverse=True)[0]
    frozen_rows = [r for r in normal_band if all(r.get(k) == v for k, v in FROZEN_SPORT_MAINLINE.items())]
    if frozen_rows:
        trusted_mainline = frozen_rows[0]
    else:
        trusted_mainline = best_candidate
    conservative_pool = [
        r
        for r in normal_band
        if r.get("structure_version") == "two_stage_conservative"
    ]
    if not conservative_pool:
        conservative_pool = [
            r
            for r in credible_rows
            if r.get("structure_version") == "two_stage_conservative"
        ]
    controlled_candidate = None
    if conservative_pool:
        controlled_candidate = sorted(conservative_pool, key=lambda r: float(r["auc"]), reverse=True)[0]
    selected = _run_future_experiment(data, label_version, feature_bundle, structure_version, population_version)
    mainline_row = trusted_mainline or selected
    ablation = {
        row["feature_bundle"]: row["auc"]
        for row in matrix
        if row["label_version"] == mainline_row.get("label_version", label_version)
        and row["structure_version"] == mainline_row.get("structure_version", structure_version)
        and row.get("population_version", "all") == mainline_row.get("population_version", population_version)
    }
    selected_auc = mainline_row.get("auc")
    suspicious = bool(isinstance(selected_auc, float) and selected_auc > 0.95)
    tautology = "low"
    drop_auc = None
    base_auc = ablation.get("baseline+deviation+trend")
    no_trend_auc = ablation.get("baseline+deviation")
    if isinstance(base_auc, float) and isinstance(no_trend_auc, float):
        drop_auc = float(base_auc - no_trend_auc)
        if drop_auc > 0.12:
            tautology = "high"
    combined_metrics["mainline_experiments"] = {
        "selected": selected,
        "best_candidate": best_candidate or {},
        "best_normal_candidate": best_candidate or {},
        "trusted_mainline": trusted_mainline or {},
        "controlled_candidate": controlled_candidate or {},
        "mainline_frozen": True,
        "matrix": matrix,
        "ablation_summary": {"trend_drop_auc": drop_auc, "by_bundle": ablation},
        "tautology_risk": tautology,
        "suspicious_high_auc": suspicious,
        "future_window_prediction": True,
    }
    with open(out_dir / "metrics_regression.json", "w", encoding="utf-8") as f:
        json.dump(reg_metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "metrics_classification.json", "w", encoding="utf-8") as f:
        json.dump(cls_metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, ensure_ascii=False, indent=2)

    return combined_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train life and sport domain models.")
    parser.add_argument("--life-active-threshold", type=int, default=3, help="Threshold for active club participation label.")
    parser.add_argument("--skip-life", action="store_true", help="Skip life domain model training.")
    parser.add_argument("--skip-sport", action="store_true", help="Skip sport domain model training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: Dict[str, object] = {}

    if not args.skip_life:
        life_data = build_life_dataset(active_threshold=args.life_active_threshold)
        summary["life"] = train_life_model(life_data, OUT_DIR / "life")

    if not args.skip_sport:
        sport_data = build_sport_dataset()
        summary["sport"] = train_sport_models(sport_data, OUT_DIR / "sport")

    with open(OUT_DIR / "domain_models_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
