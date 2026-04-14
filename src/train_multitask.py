from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
    import xgboost as xgb
except Exception:
    XGBClassifier = None
    XGBRegressor = None
    xgb = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据集及类型"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR = OUT_DIR / "models"
GPU_COMPAT_MODELS = {"LightGBM", "XGBoost"}
LAST_RUN_USED_GPU = False


def has_nvidia_gpu() -> bool:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        return completed.returncode == 0 and "GPU" in completed.stdout
    except Exception:
        return False


@dataclass
class ModelResult:
    model_name: str
    params: Dict[str, object]
    score: float
    threshold: Optional[float] = None


def safe_read_excel(path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if usecols is None:
        return pd.read_excel(path)
    return pd.read_excel(path, usecols=lambda c: c in usecols)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def pick_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_term_from_datetime(value: object) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[int]]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None, None, None, None

    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt) and text.isdigit() and len(text) >= 8:
        dt = pd.to_datetime(text[:8], format="%Y%m%d", errors="coerce")
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


def parse_term_from_year_sem(year_series: pd.Series, sem_series: pd.Series) -> pd.DataFrame:
    year_text = year_series.astype(str).str.strip()
    sem_num = pd.to_numeric(sem_series, errors="coerce").fillna(1).astype(int)
    term_id = year_text + "-T" + sem_num.astype(str)
    start_year = pd.to_numeric(year_text.str.split("-").str[0], errors="coerce")
    order_key = (start_year * 10 + sem_num).astype("Int64")
    return pd.DataFrame(
        {
            "term_id": term_id,
            "academic_year": year_text,
            "semester": sem_num,
            "term_order": order_key,
        }
    )


def resolve_term_info(df: pd.DataFrame) -> pd.DataFrame:
    year_col = pick_existing_column(df, ["XN", "KKXN", "学年", "XNMC"])
    sem_col = pick_existing_column(df, ["XQ", "KKXQ", "学期", "XQMC"])
    if year_col and sem_col:
        return parse_term_from_year_sem(df[year_col], df[sem_col])

    time_col = pick_existing_column(
        df,
        [
            "KSSJ",
            "TSTAMP",
            "ATTEND_TIME",
            "CREATE_TIME",
            "CREATED_TIME",
            "CREATED_AT",
            "CREATE_DATE",
            "SUBMIT_TIME",
            "UPDATE_TIME",
            "PUBLISH_TIME",
            "ADD_TIME",
            "CTIME",
            "DATE",
            "TIME",
        ],
    )
    if time_col:
        parsed = df[time_col].apply(infer_term_from_datetime)
        return pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])

    return pd.DataFrame(
        {
            "term_id": [None] * len(df),
            "year_start": [None] * len(df),
            "semester": [None] * len(df),
            "term_order": [None] * len(df),
        }
    )


def build_targets_from_grades() -> pd.DataFrame:
    grades = safe_read_excel(DATA_DIR / "学生成绩.xlsx", ["XH", "KSSJ", "KCCJ", "JDCJ"])
    grades = grades.dropna(subset=["XH"]).copy()
    grades["sid"] = normalize_id(grades["XH"])

    parsed = grades["KSSJ"].apply(infer_term_from_datetime)
    parsed_df = pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])
    grades = pd.concat([grades, parsed_df], axis=1)

    grades["KCCJ"] = pd.to_numeric(grades["KCCJ"], errors="coerce")
    grades["JDCJ"] = pd.to_numeric(grades["JDCJ"], errors="coerce")
    grades = grades.dropna(subset=["term_id", "KCCJ"])

    agg = (
        grades.groupby(["sid", "term_id", "year_start", "semester", "term_order"], as_index=False)
        .agg(
            avg_score=("KCCJ", "mean"),
            avg_gpa=("JDCJ", "mean"),
            course_count=("KCCJ", "size"),
            fail_count=("KCCJ", lambda s: int((s < 60).sum())),
        )
    )
    agg["fail_flag"] = (agg["fail_count"] > 0).astype(int)
    return agg


def build_selection_features() -> pd.DataFrame:
    df = safe_read_excel(DATA_DIR / "学生选课信息.xlsx", ["XH", "KKXN", "KKXQ", "KCH", "KXH"])
    if df.empty:
        return pd.DataFrame(columns=["sid", "term_id"])
    df = df.dropna(subset=["XH", "KKXN"]).copy()
    df["sid"] = normalize_id(df["XH"])

    term_df = parse_term_from_year_sem(df["KKXN"], df["KKXQ"])
    df = pd.concat([df, term_df], axis=1)

    agg = (
        df.groupby(["sid", "term_id"], as_index=False)
        .agg(
            select_course_count=("KCH", "count"),
            select_unique_course=("KCH", pd.Series.nunique),
            select_unique_class=("KXH", pd.Series.nunique),
        )
    )
    return agg


def build_attendance_features() -> pd.DataFrame:
    df = safe_read_excel(DATA_DIR / "考勤汇总.xlsx", ["XH", "XN", "XQ", "ZT"])
    if df.empty:
        return pd.DataFrame(columns=["sid", "term_id"])
    df = df.dropna(subset=["XH", "XN"]).copy()
    df["sid"] = normalize_id(df["XH"])

    term_df = parse_term_from_year_sem(df["XN"], df["XQ"])
    df = pd.concat([df, term_df], axis=1)

    df["is_absent"] = df["ZT"].astype(str).str.contains("缺勤|旷课|请假", regex=True).astype(int)
    agg = (
        df.groupby(["sid", "term_id"], as_index=False)
        .agg(attend_record_count=("ZT", "count"), absent_count=("is_absent", "sum"))
    )
    agg["absent_rate"] = np.where(agg["attend_record_count"] > 0, agg["absent_count"] / agg["attend_record_count"], 0.0)
    return agg


def build_sign_features() -> pd.DataFrame:
    df = safe_read_excel(DATA_DIR / "学生签到记录.xlsx", ["LOGIN_NAME", "TSTAMP", "ATTEND_TIME"])
    if df.empty:
        return pd.DataFrame(columns=["sid", "term_id"])

    df = df.dropna(subset=["LOGIN_NAME"]).copy()
    df["sid"] = normalize_id(df["LOGIN_NAME"])

    parsed = df["TSTAMP"].apply(infer_term_from_datetime)
    term_df = pd.DataFrame(parsed.tolist(), columns=["term_id", "year_start", "semester", "term_order"])
    df = pd.concat([df, term_df], axis=1).dropna(subset=["term_id"])

    return df.groupby(["sid", "term_id"], as_index=False).agg(sign_count=("ATTEND_TIME", "count"))


def _fallback_attach_terms_by_sid(base: pd.DataFrame, sid_term: pd.DataFrame) -> pd.DataFrame:
    # When behavior logs miss explicit time columns, duplicate sid-level stats to each observed term for that student.
    return sid_term.merge(base, on="sid", how="left")


def build_time_window_learning_features(target: pd.DataFrame) -> pd.DataFrame:
    sid_term = target[["sid", "term_id"]].drop_duplicates().copy()
    features: List[pd.DataFrame] = []

    hw = safe_read_excel(DATA_DIR / "学生作业提交记录.xlsx")
    if not hw.empty:
        sid_col = pick_existing_column(hw, ["CREATER_LOGIN_NAME", "LOGIN_NAME", "XH"])
        score_col = pick_existing_column(hw, ["SCORE", "成绩", "得分"])
        if sid_col and score_col:
            hw = hw.dropna(subset=[sid_col]).copy()
            hw["sid"] = normalize_id(hw[sid_col])
            hw[score_col] = pd.to_numeric(hw[score_col], errors="coerce")
            term_df = resolve_term_info(hw)
            hw = pd.concat([hw, term_df[["term_id"]]], axis=1)
            by_term = hw.dropna(subset=["term_id"]).groupby(["sid", "term_id"], as_index=False).agg(
                hw_submit_count=(score_col, "count"),
                hw_score_mean=(score_col, "mean"),
                hw_score_std=(score_col, "std"),
            )
            if by_term.empty:
                sid_static = hw.groupby("sid", as_index=False).agg(
                    hw_submit_count=(score_col, "count"), hw_score_mean=(score_col, "mean"), hw_score_std=(score_col, "std")
                )
                by_term = _fallback_attach_terms_by_sid(sid_static, sid_term)
            features.append(by_term)

    exam = safe_read_excel(DATA_DIR / "考试提交记录.xlsx")
    if not exam.empty:
        sid_col = pick_existing_column(exam, ["CREATER_LOGIN_NAME", "LOGIN_NAME", "XH"])
        score_col = pick_existing_column(exam, ["SCORE", "成绩", "得分"])
        if sid_col and score_col:
            exam = exam.dropna(subset=[sid_col]).copy()
            exam["sid"] = normalize_id(exam[sid_col])
            exam[score_col] = pd.to_numeric(exam[score_col], errors="coerce")
            term_df = resolve_term_info(exam)
            exam = pd.concat([exam, term_df[["term_id"]]], axis=1)
            by_term = exam.dropna(subset=["term_id"]).groupby(["sid", "term_id"], as_index=False).agg(
                exam_submit_count=(score_col, "count"),
                exam_score_mean=(score_col, "mean"),
                exam_score_std=(score_col, "std"),
            )
            if by_term.empty:
                sid_static = exam.groupby("sid", as_index=False).agg(
                    exam_submit_count=(score_col, "count"),
                    exam_score_mean=(score_col, "mean"),
                    exam_score_std=(score_col, "std"),
                )
                by_term = _fallback_attach_terms_by_sid(sid_static, sid_term)
            features.append(by_term)

    discuss = safe_read_excel(DATA_DIR / "讨论记录.xlsx")
    if not discuss.empty:
        term_df = resolve_term_info(discuss)
        discuss = pd.concat([discuss, term_df[["term_id"]]], axis=1)

        topic = pd.DataFrame(columns=["sid", "term_id", "topic_count"])
        reply = pd.DataFrame(columns=["sid", "term_id", "reply_count"])

        if "CREATER_LOGIN_NAME" in discuss.columns:
            c = discuss.dropna(subset=["CREATER_LOGIN_NAME"]).copy()
            c["sid"] = normalize_id(c["CREATER_LOGIN_NAME"])
            if c["term_id"].notna().any():
                topic = c.dropna(subset=["term_id"]).groupby(["sid", "term_id"], as_index=False).agg(
                    topic_count=(pick_existing_column(c, ["TOPIC_ID", "ID"]) or "sid", "count")
                )
            else:
                topic_sid = c.groupby("sid", as_index=False).agg(topic_count=("sid", "count"))
                topic = _fallback_attach_terms_by_sid(topic_sid, sid_term)

        if "REPLY_LOGIN_NAME" in discuss.columns:
            r = discuss.dropna(subset=["REPLY_LOGIN_NAME"]).copy()
            r["sid"] = normalize_id(r["REPLY_LOGIN_NAME"])
            if r["term_id"].notna().any():
                reply = r.dropna(subset=["term_id"]).groupby(["sid", "term_id"], as_index=False).agg(
                    reply_count=(pick_existing_column(r, ["TOPIC_ID", "ID"]) or "sid", "count")
                )
            else:
                reply_sid = r.groupby("sid", as_index=False).agg(reply_count=("sid", "count"))
                reply = _fallback_attach_terms_by_sid(reply_sid, sid_term)

        by_term = topic.merge(reply, on=["sid", "term_id"], how="outer")
        if not by_term.empty:
            features.append(by_term)

    online = safe_read_excel(DATA_DIR / "线上学习（综合表现）.xlsx", ["LOGIN_NAME", "BFB"])
    if not online.empty:
        online = online.dropna(subset=["LOGIN_NAME"]).copy()
        online["sid"] = normalize_id(online["LOGIN_NAME"])
        online["BFB"] = pd.to_numeric(online["BFB"], errors="coerce")
        o_sid = online.groupby("sid", as_index=False).agg(online_bfb=("BFB", "mean"))
        features.append(_fallback_attach_terms_by_sid(o_sid, sid_term))

    task = safe_read_excel(
        DATA_DIR / "课堂任务参与.xlsx",
        ["LOGIN_NAME", "JOB_NUM", "JOB_RATE", "TEST_AVGSCORE", "WORK_AVGSCORE", "EXAM_AVGSCORE", "BBS_NUM", "REPLY_NUM"],
    )
    if not task.empty:
        task = task.dropna(subset=["LOGIN_NAME"]).copy()
        task["sid"] = normalize_id(task["LOGIN_NAME"])
        num_cols = ["JOB_NUM", "JOB_RATE", "TEST_AVGSCORE", "WORK_AVGSCORE", "EXAM_AVGSCORE", "BBS_NUM", "REPLY_NUM"]
        for c in num_cols:
            if c in task.columns:
                task[c] = pd.to_numeric(task[c], errors="coerce")
        t_sid = task.groupby("sid", as_index=False).agg(
            task_job_num=("JOB_NUM", "mean"),
            task_job_rate=("JOB_RATE", "mean"),
            task_test_avg=("TEST_AVGSCORE", "mean"),
            task_work_avg=("WORK_AVGSCORE", "mean"),
            task_exam_avg=("EXAM_AVGSCORE", "mean"),
            task_bbs_num=("BBS_NUM", "mean"),
            task_reply_num=("REPLY_NUM", "mean"),
        )
        features.append(_fallback_attach_terms_by_sid(t_sid, sid_term))

    if not features:
        return pd.DataFrame(columns=["sid", "term_id"])

    out = features[0]
    for feat in features[1:]:
        out = out.merge(feat, on=["sid", "term_id"], how="outer")
    return out


def build_student_profile() -> pd.DataFrame:
    profile = safe_read_excel(DATA_DIR / "学生基本信息.xlsx", ["XH", "XB", "MZMC", "ZZMMMC", "JG", "ZYM"])
    if profile.empty:
        return pd.DataFrame(columns=["sid"])
    profile = profile.dropna(subset=["XH"]).copy()
    profile["sid"] = normalize_id(profile["XH"])
    for c in ["XB", "MZMC", "ZZMMMC", "JG", "ZYM"]:
        if c in profile.columns:
            profile[c] = profile[c].astype(str)
    return profile[["sid", "XB", "MZMC", "ZZMMMC", "JG", "ZYM"]].drop_duplicates("sid")


def build_training_table() -> pd.DataFrame:
    target = build_targets_from_grades()
    selection = build_selection_features()
    attendance = build_attendance_features()
    sign = build_sign_features()
    dynamic_learning = build_time_window_learning_features(target)
    profile = build_student_profile()

    data = target.merge(selection, on=["sid", "term_id"], how="left")
    data = data.merge(attendance, on=["sid", "term_id"], how="left")
    data = data.merge(sign, on=["sid", "term_id"], how="left")
    data = data.merge(dynamic_learning, on=["sid", "term_id"], how="left")
    data = data.merge(profile, on="sid", how="left")

    for col in data.columns:
        if any(k in col for k in ["count", "num", "rate", "mean", "std", "avg"]):
            data[col] = pd.to_numeric(data[col], errors="coerce")

    count_cols = [c for c in data.columns if any(k in c for k in ["count", "num"])]
    if count_cols:
        data[count_cols] = data[count_cols].fillna(0)

    data = data.sort_values(["term_order", "sid"]).reset_index(drop=True)
    return data


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
    if len(val) < 10:
        val = train_all.sample(frac=0.2, random_state=42)
        train = train_all.drop(val.index)
    return train, val, test


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


def get_feature_names(pipe: Pipeline) -> List[str]:
    names = pipe.named_steps["pre"].get_feature_names_out()
    return [str(n) for n in names]


def extract_feature_importance(pipe: Pipeline) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    importance = np.asarray(model.feature_importances_, dtype=float)
    feat_names = get_feature_names(pipe)
    if len(feat_names) != len(importance):
        return pd.DataFrame(columns=["feature", "importance"])
    out = pd.DataFrame({"feature": feat_names, "importance": importance})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multitask models with optional GPU acceleration.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Training device mode. auto will prefer GPU for LightGBM/XGBoost when available.",
    )
    return parser.parse_args()


def resolve_gpu_usage(device: str) -> bool:
    has_gpu_capable_lib = (LGBMRegressor is not None and LGBMClassifier is not None) or (
        XGBRegressor is not None and XGBClassifier is not None
    )
    if device == "cpu":
        return False
    if device == "gpu" and not has_gpu_capable_lib:
        warnings.warn("GPU mode requested, but LightGBM/XGBoost is unavailable. Falling back to CPU.")
        return False
    return has_gpu_capable_lib


def model_supports_gpu(model_name: str) -> bool:
    return model_name in GPU_COMPAT_MODELS


def augment_params_for_device(model_name: str, params: Dict[str, object], use_gpu: bool) -> Dict[str, object]:
    merged = dict(params)
    if not use_gpu or not model_supports_gpu(model_name):
        return merged
    if model_name == "LightGBM":
        merged.setdefault("device_type", "gpu")
        # Prefer NVIDIA OpenCL platform when available; default auto often picks Intel iGPU first.
        if has_nvidia_gpu():
            merged.setdefault("gpu_platform_id", 1)
            merged.setdefault("gpu_device_id", 0)
    elif model_name == "XGBoost":
        if os.environ.get("XGB_USE_CUDA", "0") != "1":
            return merged
        version = str(getattr(xgb, "__version__", "")) if xgb is not None else ""
        major = int(version.split(".")[0]) if version and version.split(".")[0].isdigit() else 0
        if major >= 2:
            # XGBoost 2.x prefers `device=cuda` + `tree_method=hist`.
            merged.setdefault("device", "cuda")
            merged.setdefault("tree_method", "hist")
        else:
            merged.setdefault("tree_method", "gpu_hist")
            merged.setdefault("predictor", "gpu_predictor")
    return merged


def save_artifacts(
    reg_pipe: Pipeline,
    cls_pipe: Pipeline,
    cluster_artifact: Dict[str, object],
    reg_metrics: Dict[str, float],
    cls_metrics: Dict[str, float],
    device: str,
    use_gpu: bool,
    fallback_logs: List[str],
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    reg_model_path = MODELS_DIR / "best_regression_model.joblib"
    cls_model_path = MODELS_DIR / "best_classification_model.joblib"
    cluster_model_path = MODELS_DIR / "cluster_model.joblib"

    joblib.dump(reg_pipe, reg_model_path)
    joblib.dump(cls_pipe, cls_model_path)
    joblib.dump(cluster_artifact, cluster_model_path)

    metadata = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "device_request": device,
        "gpu_enabled": bool(use_gpu),
        "regression": reg_metrics,
        "classification": cls_metrics,
        "cluster": {"n_clusters": int(cluster_artifact.get("n_clusters", 4))},
        "gpu_fallback_logs": fallback_logs,
    }
    with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def regression_candidates() -> Dict[str, List[Dict[str, object]]]:
    cands: Dict[str, List[Dict[str, object]]] = {
        "RandomForest": [
            {"n_estimators": 300, "max_depth": None},
            {"n_estimators": 500, "max_depth": 12},
        ]
    }
    if LGBMRegressor is not None:
        cands["LightGBM"] = [
            {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31},
            {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63},
        ]
    if XGBRegressor is not None:
        cands["XGBoost"] = [
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6},
            {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 8},
        ]
    return cands


def classification_candidates() -> Dict[str, List[Dict[str, object]]]:
    cands: Dict[str, List[Dict[str, object]]] = {
        "RandomForest": [
            {"n_estimators": 400, "max_depth": None, "class_weight": "balanced"},
            {"n_estimators": 600, "max_depth": 16, "class_weight": "balanced_subsample"},
        ]
    }
    if LGBMClassifier is not None:
        cands["LightGBM"] = [
            {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "class_weight": "balanced"},
            {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63, "class_weight": "balanced"},
        ]
    if XGBClassifier is not None:
        cands["XGBoost"] = [
            {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6, "scale_pos_weight": 1.0},
            {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 8, "scale_pos_weight": 1.0},
        ]
    return cands


def make_regressor(name: str, params: Dict[str, object], use_gpu: bool = False):
    params = augment_params_for_device(name, params, use_gpu)
    if name == "RandomForest":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if name == "LightGBM" and LGBMRegressor is not None:
        return LGBMRegressor(random_state=42, n_jobs=-1, **params)
    if name == "XGBoost" and XGBRegressor is not None:
        return XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror", **params)
    raise ValueError(f"Unsupported regression model: {name}")


def make_classifier(name: str, params: Dict[str, object], use_gpu: bool = False):
    params = augment_params_for_device(name, params, use_gpu)
    if name == "RandomForest":
        return RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    if name == "LightGBM" and LGBMClassifier is not None:
        return LGBMClassifier(random_state=42, n_jobs=-1, **params)
    if name == "XGBoost" and XGBClassifier is not None:
        return XGBClassifier(
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            **params,
        )
    raise ValueError(f"Unsupported classification model: {name}")


def train_and_select_regression(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    use_gpu: bool,
    gpu_fallback_logs: List[str],
) -> Tuple[Dict[str, float], pd.Series, pd.DataFrame, pd.DataFrame, str, Pipeline]:
    y_train = train["avg_score"]
    y_val = val["avg_score"]
    y_test = test["avg_score"]

    comp_rows: List[Dict[str, object]] = []
    best: Optional[ModelResult] = None

    for model_name, grid in regression_candidates().items():
        for params in grid:
            pre = build_preprocessor(train, feature_cols)
            try:
                model = make_regressor(model_name, params, use_gpu=use_gpu)
                pipe = Pipeline(steps=[("pre", pre), ("model", model)])
                pipe.fit(train[feature_cols], y_train)
            except Exception as exc:
                if use_gpu and model_supports_gpu(model_name):
                    gpu_fallback_logs.append(f"regression/{model_name}: {type(exc).__name__}")
                    warnings.warn(f"{model_name} GPU training failed, retrying on CPU: {exc}")
                    try:
                        model = make_regressor(model_name, params, use_gpu=False)
                        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
                        pipe.fit(train[feature_cols], y_train)
                    except Exception as retry_exc:
                        warnings.warn(f"Skipping regression candidate {model_name} due to error: {retry_exc}")
                        continue
                else:
                    warnings.warn(f"Skipping regression candidate {model_name} due to error: {exc}")
                    continue
            val_pred = pipe.predict(val[feature_cols])
            score = float(r2_score(y_val, val_pred))
            comp_rows.append({"model": model_name, "params": json.dumps(params, ensure_ascii=False), "val_r2": score})
            if best is None or score > best.score:
                best = ModelResult(model_name=model_name, params=params, score=score)

    if best is None:
        raise RuntimeError("No regression model candidates were available.")

    full_train = pd.concat([train, val], axis=0)
    pre = build_preprocessor(full_train, feature_cols)
    try:
        model = make_regressor(best.model_name, best.params, use_gpu=use_gpu)
        best_pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        best_pipe.fit(full_train[feature_cols], full_train["avg_score"])
    except Exception as exc:
        if use_gpu and model_supports_gpu(best.model_name):
            gpu_fallback_logs.append(f"regression/final/{best.model_name}: {type(exc).__name__}")
            warnings.warn(f"Best regression model GPU training failed, retrying on CPU: {exc}")
            model = make_regressor(best.model_name, best.params, use_gpu=False)
            best_pipe = Pipeline(steps=[("pre", pre), ("model", model)])
            best_pipe.fit(full_train[feature_cols], full_train["avg_score"])
        else:
            raise
    test_pred = best_pipe.predict(test[feature_cols])

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "mae": float(mean_absolute_error(y_test, test_pred)),
        "r2": float(r2_score(y_test, test_pred)),
        "best_model": best.model_name,
        "best_val_r2": float(best.score),
    }

    importance = extract_feature_importance(best_pipe)
    comp_df = pd.DataFrame(comp_rows).sort_values("val_r2", ascending=False).reset_index(drop=True)
    return metrics, pd.Series(test_pred, index=test.index), comp_df, importance, best.model_name, best_pipe


def train_and_select_classification(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: List[str],
    use_gpu: bool,
    gpu_fallback_logs: List[str],
) -> Tuple[Dict[str, float], pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, str, Pipeline]:
    y_train = train["fail_flag"]
    y_val = val["fail_flag"]
    y_test = test["fail_flag"]

    comp_rows: List[Dict[str, object]] = []
    best: Optional[ModelResult] = None

    for model_name, grid in classification_candidates().items():
        for params in grid:
            pre = build_preprocessor(train, feature_cols)
            try:
                model = make_classifier(model_name, params, use_gpu=use_gpu)
                pipe = Pipeline(steps=[("pre", pre), ("model", model)])
                pipe.fit(train[feature_cols], y_train)
            except Exception as exc:
                if use_gpu and model_supports_gpu(model_name):
                    gpu_fallback_logs.append(f"classification/{model_name}: {type(exc).__name__}")
                    warnings.warn(f"{model_name} GPU training failed, retrying on CPU: {exc}")
                    try:
                        model = make_classifier(model_name, params, use_gpu=False)
                        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
                        pipe.fit(train[feature_cols], y_train)
                    except Exception as retry_exc:
                        warnings.warn(f"Skipping classification candidate {model_name} due to error: {retry_exc}")
                        continue
                else:
                    warnings.warn(f"Skipping classification candidate {model_name} due to error: {exc}")
                    continue
            val_proba = pipe.predict_proba(val[feature_cols])[:, 1]
            threshold, val_f1 = optimize_f1_threshold(y_val, val_proba)
            try:
                val_auc = float(roc_auc_score(y_val, val_proba))
            except ValueError:
                val_auc = float("nan")
            comp_rows.append(
                {
                    "model": model_name,
                    "params": json.dumps(params, ensure_ascii=False),
                    "best_val_threshold": threshold,
                    "val_f1": val_f1,
                    "val_auc": val_auc,
                }
            )
            if best is None or val_f1 > best.score:
                best = ModelResult(model_name=model_name, params=params, score=val_f1, threshold=threshold)

    if best is None:
        raise RuntimeError("No classification model candidates were available.")

    full_train = pd.concat([train, val], axis=0)
    pre = build_preprocessor(full_train, feature_cols)
    try:
        model = make_classifier(best.model_name, best.params, use_gpu=use_gpu)
        best_pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        best_pipe.fit(full_train[feature_cols], full_train["fail_flag"])
    except Exception as exc:
        if use_gpu and model_supports_gpu(best.model_name):
            gpu_fallback_logs.append(f"classification/final/{best.model_name}: {type(exc).__name__}")
            warnings.warn(f"Best classification model GPU training failed, retrying on CPU: {exc}")
            model = make_classifier(best.model_name, best.params, use_gpu=False)
            best_pipe = Pipeline(steps=[("pre", pre), ("model", model)])
            best_pipe.fit(full_train[feature_cols], full_train["fail_flag"])
        else:
            raise

    test_proba = best_pipe.predict_proba(test[feature_cols])[:, 1]
    test_pred = (test_proba >= (best.threshold or 0.5)).astype(int)

    try:
        test_auc = float(roc_auc_score(y_test, test_proba))
    except ValueError:
        test_auc = float("nan")

    metrics = {
        "auc": test_auc,
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "positive_rate_test": float(np.mean(y_test)),
        "best_model": best.model_name,
        "best_val_f1": float(best.score),
        "best_threshold": float(best.threshold or 0.5),
    }

    importance = extract_feature_importance(best_pipe)
    comp_df = pd.DataFrame(comp_rows).sort_values("val_f1", ascending=False).reset_index(drop=True)
    return (
        metrics,
        pd.Series(test_pred, index=test.index),
        pd.Series(test_proba, index=test.index),
        comp_df,
        importance,
        best.model_name,
        best_pipe,
    )


def clustering_task(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.Series, pd.DataFrame, Dict[str, object]]:
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        numeric_cols = ["avg_score", "avg_gpa", "fail_flag"]
    x = df[numeric_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True))
    x = x.fillna(0.0)
    x = x.clip(lower=-1e6, upper=1e6)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, n_init=10, batch_size=2048)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        labels = kmeans.fit_predict(x_scaled)

    cluster_df = df[["sid", "term_id", "avg_score", "fail_flag"]].copy()
    cluster_df["cluster"] = labels
    summary = cluster_df.groupby("cluster", as_index=False).agg(
        sample_count=("sid", "count"),
        mean_score=("avg_score", "mean"),
        fail_rate=("fail_flag", "mean"),
    )
    cluster_artifact = {
        "scaler": scaler,
        "kmeans": kmeans,
        "feature_cols": numeric_cols,
        "n_clusters": 4,
    }
    return pd.Series(labels, index=df.index), summary, cluster_artifact


def score_task(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["sid", "term_id", "avg_score", "avg_gpa", "fail_flag"]].copy()
    source = pd.DataFrame(
        {
            "avg_score": pd.to_numeric(df.get("avg_score"), errors="coerce"),
            "avg_gpa": pd.to_numeric(df.get("avg_gpa"), errors="coerce"),
            "absent_rate": pd.to_numeric(df.get("absent_rate"), errors="coerce"),
            "select_course_count": pd.to_numeric(df.get("select_course_count"), errors="coerce"),
            "hw_score_mean": pd.to_numeric(df.get("hw_score_mean"), errors="coerce"),
            "exam_score_mean": pd.to_numeric(df.get("exam_score_mean"), errors="coerce"),
            "online_bfb": pd.to_numeric(df.get("online_bfb"), errors="coerce"),
            "sign_count": pd.to_numeric(df.get("sign_count"), errors="coerce"),
        }
    )

    weights = {
        "avg_score": 0.25,
        "avg_gpa": 0.20,
        "absent_rate": -0.15,
        "select_course_count": 0.10,
        "hw_score_mean": 0.10,
        "exam_score_mean": 0.10,
        "online_bfb": 0.05,
        "sign_count": 0.05,
    }

    z = (source - source.mean()) / source.std(ddof=0)
    z = z.fillna(0.0)

    composite = sum(weights[c] * z[c] for c in weights)
    c_min = composite.min()
    c_max = composite.max()
    if c_max - c_min < 1e-12:
        score_100 = pd.Series(np.full(len(composite), 50.0), index=composite.index)
    else:
        score_100 = 100 * (composite - c_min) / (c_max - c_min)

    out["learning_score"] = score_100
    out["learning_score_rank"] = out["learning_score"].rank(ascending=False, method="dense").astype(int)
    return out


def main() -> None:
    global LAST_RUN_USED_GPU
    args = parse_args()
    use_gpu = resolve_gpu_usage(args.device)
    LAST_RUN_USED_GPU = bool(use_gpu)
    gpu_fallback_logs: List[str] = []

    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBM.* was fitted with feature names",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Falling back to prediction using DMatrix due to mismatched devices.*",
        category=UserWarning,
    )

    print(f"Requested device mode: {args.device}")
    print(f"GPU path enabled for supported models: {use_gpu}")

    data = build_training_table()
    if data.empty:
        raise RuntimeError("No training data was built. Please check input files.")

    label_cols = ["avg_score", "avg_gpa", "fail_count", "fail_flag"]
    identity_cols = ["sid", "term_id", "year_start", "semester", "term_order"]
    raw_feature_cols = [c for c in data.columns if c not in set(label_cols + identity_cols)]
    all_nan_feature_cols = [c for c in raw_feature_cols if data[c].notna().sum() == 0]
    if all_nan_feature_cols:
        data = data.drop(columns=all_nan_feature_cols)

    data.to_csv(OUT_DIR / "feature_dataset.csv", index=False)
    feature_cols = [c for c in data.columns if c not in set(label_cols + identity_cols)]

    train, val, test = split_train_val_test_by_term(data)
    if train.empty or val.empty or test.empty:
        raise RuntimeError("Train/val/test split failed due to insufficient data.")

    reg_metrics, reg_pred, reg_cmp, reg_importance, best_reg, reg_pipe = train_and_select_regression(
        train,
        val,
        test,
        feature_cols,
        use_gpu=use_gpu,
        gpu_fallback_logs=gpu_fallback_logs,
    )
    cls_metrics, cls_pred, cls_proba, cls_cmp, cls_importance, best_cls, cls_pipe = train_and_select_classification(
        train,
        val,
        test,
        feature_cols,
        use_gpu=use_gpu,
        gpu_fallback_logs=gpu_fallback_logs,
    )

    clusters, cluster_summary, cluster_artifact = clustering_task(data, feature_cols)
    score_df = score_task(data)

    save_artifacts(
        reg_pipe=reg_pipe,
        cls_pipe=cls_pipe,
        cluster_artifact=cluster_artifact,
        reg_metrics=reg_metrics,
        cls_metrics=cls_metrics,
        device=args.device,
        use_gpu=use_gpu,
        fallback_logs=gpu_fallback_logs,
    )

    pred_df = test[["sid", "term_id", "avg_score", "fail_flag"]].copy()
    pred_df["pred_avg_score"] = reg_pred
    pred_df["pred_fail_flag"] = cls_pred
    pred_df["pred_fail_proba"] = cls_proba
    pred_df["best_reg_model"] = best_reg
    pred_df["best_cls_model"] = best_cls
    pred_df.to_csv(OUT_DIR / "predictions_test.csv", index=False)

    cluster_out = data[["sid", "term_id"]].copy()
    cluster_out["cluster"] = clusters
    cluster_out.to_csv(OUT_DIR / "cluster_labels.csv", index=False)
    cluster_summary.to_csv(OUT_DIR / "cluster_summary.csv", index=False)

    score_df.to_csv(OUT_DIR / "learning_score.csv", index=False)

    reg_cmp.to_csv(OUT_DIR / "model_comparison_regression.csv", index=False)
    cls_cmp.to_csv(OUT_DIR / "model_comparison_classification.csv", index=False)
    reg_importance.to_csv(OUT_DIR / "feature_importance_regression.csv", index=False)
    cls_importance.to_csv(OUT_DIR / "feature_importance_classification.csv", index=False)

    metrics = {
        "dataset": {
            "total_samples": int(len(data)),
            "train_samples": int(len(train)),
            "val_samples": int(len(val)),
            "test_samples": int(len(test)),
            "num_features": int(len(feature_cols)),
            "val_terms": sorted(val["term_id"].dropna().unique().tolist()),
            "test_terms": sorted(test["term_id"].dropna().unique().tolist()),
            "device_request": args.device,
            "gpu_enabled": bool(use_gpu),
        },
        "regression": reg_metrics,
        "classification": cls_metrics,
        "clustering": {
            "n_clusters": 4,
            "cluster_sizes": cluster_summary.set_index("cluster")["sample_count"].to_dict(),
        },
        "artifacts": {
            "model_dir": str(MODELS_DIR),
            "gpu_fallback_logs": gpu_fallback_logs,
        },
    }

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Done. Outputs generated in:", OUT_DIR)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    succeeded = False
    try:
        main()
        succeeded = True
    finally:
        # Some GPU backends may crash during Python teardown on Windows after successful training.
        if succeeded and LAST_RUN_USED_GPU and os.environ.get("FAST_EXIT_ON_GPU", "1") == "1":
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
