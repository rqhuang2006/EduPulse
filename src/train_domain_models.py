from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
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


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据集及类型"
OUT_DIR = ROOT / "outputs"


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
    best_f1 = -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (proba >= t).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t, float(best_f1)


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
    target = safe_read_excel(DATA_DIR / "体测数据.xlsx")
    if target.empty:
        return pd.DataFrame()

    target = target.dropna(subset=["XH"]).copy()
    target["sid"] = normalize_id(target["XH"])

    parsed = target["TCNF"].apply(parse_year_to_term)
    p = pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])
    target = pd.concat([target, p], axis=1).dropna(subset=["term_id"])

    target["ZF"] = pd.to_numeric(target.get("ZF"), errors="coerce")
    target = target.dropna(subset=["ZF"])
    # Derive a stable grade label from numeric score to avoid noisy text labels.
    target["zf_grade"] = pd.cut(
        target["ZF"],
        bins=[-np.inf, 60, 70, 80, 90, np.inf],
        labels=["E", "D", "C", "B", "A"],
    ).astype(str)

    base = target.groupby(["sid", "term_id", "year_start", "semester", "term_order"], as_index=False).agg(
        zf_score=("ZF", "mean"),
        zf_grade=("zf_grade", lambda s: s.mode().iloc[0] if not s.mode().empty else "C"),
        bmi_count=("BMI", "count"),
        fhl_mean=("FHL", "mean"),
        ws_mean=("WS", "mean"),
        ldty_mean=("LDTY", "mean"),
        zwtqq_mean=("ZWTQQ", "mean"),
        bb_mean=("BB", "mean"),
        ywqz_mean=("YWQZ", "mean"),
    )

    sport_course = safe_read_excel(DATA_DIR / "体育课.xlsx", ["XH", "XQ", "KC", "ZKC"])
    if not sport_course.empty:
        sport_course = sport_course.dropna(subset=["XH"]).copy()
        sport_course["sid"] = normalize_id(sport_course["XH"])
        parsed = sport_course["XQ"].apply(parse_term_text)
        p = pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])
        sport_course = pd.concat([sport_course, p], axis=1).dropna(subset=["term_id"])
        sport_course_feat = sport_course.groupby(["sid", "term_id"], as_index=False).agg(
            pe_course_count=("KC", "count"),
            pe_unique_course=("KC", pd.Series.nunique),
            pe_unique_class=("ZKC", pd.Series.nunique),
        )
    else:
        sport_course_feat = pd.DataFrame(columns=["sid", "term_id"])

    daily = safe_read_excel(DATA_DIR / "日常锻炼.xlsx", ["XH", "XQ", "ZC", "DKCS"])
    if not daily.empty:
        daily = daily.dropna(subset=["XH"]).copy()
        daily["sid"] = normalize_id(daily["XH"])
        daily["DKCS"] = pd.to_numeric(daily["DKCS"], errors="coerce")
        parsed = daily["XQ"].apply(parse_term_text)
        p = pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])
        daily = pd.concat([daily, p], axis=1).dropna(subset=["term_id"])
        daily_feat = daily.groupby(["sid", "term_id"], as_index=False).agg(
            daily_daka_sum=("DKCS", "sum"),
            daily_daka_mean=("DKCS", "mean"),
            daily_week_count=("ZC", "nunique"),
        )
    else:
        daily_feat = pd.DataFrame(columns=["sid", "term_id"])

    run = safe_read_excel(DATA_DIR / "跑步打卡.xlsx", ["USERNUM", "PUNCH_DAY", "STATE"])
    if not run.empty:
        run = run.dropna(subset=["USERNUM"]).copy()
        run["sid"] = normalize_id(run["USERNUM"])
        parsed = run["PUNCH_DAY"].apply(infer_term_from_datetime)
        p = pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])
        run = pd.concat([run, p], axis=1).dropna(subset=["term_id"])
        run_feat = run.groupby(["sid", "term_id"], as_index=False).agg(
            run_punch_count=("PUNCH_DAY", "count"),
            run_state_mean=("STATE", "mean"),
        )
    else:
        run_feat = pd.DataFrame(columns=["sid", "term_id"])

    fitness = safe_read_excel(DATA_DIR / "学生体能考核.xlsx", ["XH", "XQ", "CFBFS", "SJCJ", "HSCJ"])
    if not fitness.empty:
        fitness = fitness.dropna(subset=["XH"]).copy()
        fitness["sid"] = normalize_id(fitness["XH"])
        for c in ["CFBFS", "SJCJ", "HSCJ"]:
            fitness[c] = pd.to_numeric(fitness[c], errors="coerce")
        parsed = fitness["XQ"].apply(parse_term_text)
        p = pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])
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

    data = data.sort_values(["term_order", "sid"]).reset_index(drop=True)
    return data


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


def train_sport_models(data: pd.DataFrame, out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if data.empty:
        raise RuntimeError("Sport dataset is empty.")

    id_cols = ["sid", "term_id", "year_start", "semester", "term_order"]
    reg_label = "zf_score"
    cls_label = "zf_grade"
    feature_cols = [c for c in data.columns if c not in set(id_cols + [reg_label, cls_label])]
    all_nan_feature_cols = [c for c in feature_cols if data[c].notna().sum() == 0]
    if all_nan_feature_cols:
        data = data.drop(columns=all_nan_feature_cols)
        feature_cols = [c for c in feature_cols if c not in set(all_nan_feature_cols)]

    train, val, test = split_train_val_test_by_term(data)
    if train.empty or val.empty or test.empty:
        raise RuntimeError("Sport train/val/test split failed due to insufficient data.")

    pre = build_preprocessor(train, feature_cols)

    reg_candidates: Dict[str, object] = {
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1),
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

    cls_candidates: Dict[str, object] = {
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
    }
    if LGBMClassifier is not None:
        cls_candidates["LightGBM"] = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=42)
    if XGBClassifier is not None:
        cls_candidates["XGBoost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric="mlogloss",
            tree_method="hist",
        )

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
    }

    data.to_csv(out_dir / "feature_dataset.csv", index=False)

    combined_metrics = {
        "regression": reg_metrics,
        "classification": cls_metrics,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    life_data = build_life_dataset(active_threshold=args.life_active_threshold)
    sport_data = build_sport_dataset()

    life_metrics = train_life_model(life_data, OUT_DIR / "life")
    sport_metrics = train_sport_models(sport_data, OUT_DIR / "sport")

    summary = {
        "life": life_metrics,
        "sport": sport_metrics,
    }
    with open(OUT_DIR / "domain_models_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
