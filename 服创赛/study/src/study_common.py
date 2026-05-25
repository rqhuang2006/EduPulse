from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ODS_DIR = DATA_DIR / "ods"
DWD_DIR = DATA_DIR / "dwd"
DM_DIR = DATA_DIR / "dm"
DELIVERABLE_DIR = DATA_DIR / "deliverables"
CONF_DIR = ROOT / "conf"
DOCS_DIR = ROOT / "docs"

NULL_LIKE = {"", "null", "none", "nan", "na", "n/a", "无", "无数据", "空", "--", "-"}
ID_COLUMNS = ["XH", "TERM_ID"]
STANDARD_FIELDS = [
    "XH",
    "TERM_ID",
    "COURSE_STD",
    "COURSE_NAME",
    "CREDIT",
    "COURSE_TYPE",
    "EVENT_TIME",
    "SCORE",
    "STATUS",
    "SOURCE_FILE",
]


def ensure_dirs() -> None:
    for path in [RAW_DIR, ODS_DIR, DWD_DIR, DM_DIR, DELIVERABLE_DIR, CONF_DIR, DOCS_DIR, ROOT / "notebooks"]:
        path.mkdir(parents=True, exist_ok=True)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def file_registry() -> dict[str, Any]:
    return read_yaml(CONF_DIR / "file_registry.yaml").get("datasets", {})


def field_alias() -> dict[str, Any]:
    return read_yaml(CONF_DIR / "field_alias.yaml")


def feature_registry() -> dict[str, Any]:
    return read_yaml(CONF_DIR / "feature_registry.yaml")


def model_params() -> dict[str, Any]:
    return read_yaml(CONF_DIR / "model_params.yaml")


def clean_column_name(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    return text


def normalize_key(value: Any) -> str:
    return re.sub(r"[\s_\-（）()]+", "", str(value).strip().upper())


def normalize_empty(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in NULL_LIKE:
            return pd.NA
        return text
    return value


def normalize_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [clean_column_name(c) for c in result.columns]
    object_cols = result.select_dtypes(include=["object"]).columns
    for col in object_cols:
        result[col] = result[col].map(normalize_empty)
    return result.replace({np.nan: pd.NA})


def stringify_id(series: pd.Series) -> pd.Series:
    def _one(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        text = str(value).strip()
        if text.lower() in NULL_LIKE:
            return pd.NA
        if re.fullmatch(r"\d+\.0", text):
            text = text[:-2]
        return text

    return series.map(_one).astype("string")


def to_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    cleaned = series.astype("string").str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def to_datetime(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    text = series.astype("string").str.strip()
    ymd_mask = text.str.match(r"^\d{8}$").fillna(False)
    time_only_mask = text.str.match(r"^\d{1,2}:\d{2}:\d{2}(\.\d+)?$").fillna(False)
    parsed = pd.to_datetime(text.where(ymd_mask), format="%Y%m%d", errors="coerce", utc=True)
    fallback = pd.to_datetime(text.where(~ymd_mask & ~time_only_mask), errors="coerce", utc=True)
    return parsed.combine_first(fallback).dt.tz_convert(None)


def raw_path_for(dataset_name: str) -> Path:
    registry = file_registry()
    return RAW_DIR / registry[dataset_name]["raw_file"]


def ods_path_for(dataset_name: str) -> Path:
    registry = file_registry()
    return ODS_DIR / registry[dataset_name].get("ods_file", f"ods_{dataset_name}.parquet")


def load_ods(dataset_name: str) -> pd.DataFrame:
    path = ods_path_for(dataset_name)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == "object":
            safe[col] = safe[col].astype("string")
    safe.to_parquet(path, index=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def read_excel_all_sheets(path: Path, nrows: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    excel = pd.ExcelFile(path)
    for sheet in excel.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, nrows=nrows)
        df = normalize_string_columns(df)
        df["SOURCE_SHEET"] = sheet
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _alias_lookup(alias_conf: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for standard, aliases in alias_conf.get("standard_fields", {}).items():
        for alias in aliases:
            lookup[normalize_key(alias)] = standard
    return lookup


def apply_standard_aliases(df: pd.DataFrame, alias_conf: dict[str, Any] | None = None) -> pd.DataFrame:
    alias_conf = alias_conf or field_alias()
    lookup = _alias_lookup(alias_conf)
    result = df.copy()
    result.columns = [clean_column_name(c) for c in result.columns]

    grouped: dict[str, list[str]] = {}
    for col in result.columns:
        standard = lookup.get(normalize_key(col))
        if standard:
            grouped.setdefault(standard, []).append(col)

    for standard, cols in grouped.items():
        if standard in result.columns and standard not in cols:
            cols = [standard] + cols
        combined = result[cols[0]]
        for col in cols[1:]:
            combined = combined.combine_first(result[col])
        result[standard] = combined
        duplicate_cols = [c for c in cols if c != standard]
        result = result.drop(columns=duplicate_cols, errors="ignore")

    return result


def academic_year_from_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"(20\d{2})\D+(20\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    match = re.search(r"(20\d{2})", text)
    if match:
        year = int(match.group(1))
        return f"{year}-{year + 1}"
    return None


def semester_from_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in {"1", "1.0", "一", "第一学期", "上", "上学期", "秋", "秋季", "9"}:
        return "1"
    if text in {"2", "2.0", "二", "第二学期", "下", "下学期", "春", "春季", "3"}:
        return "2"
    match = re.search(r"([12])", text)
    return match.group(1) if match else None


def term_from_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    match = re.search(r"(20\d{2})\D+(20\d{2})\D+([12])", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


def term_from_date(value: Any) -> str | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    year = ts.year if ts.month >= 9 else ts.year - 1
    semester = "1" if ts.month >= 9 or ts.month <= 1 else "2"
    return f"{year}-{year + 1}-{semester}"


def infer_term_id(df: pd.DataFrame, alias_conf: dict[str, Any] | None = None) -> pd.Series:
    alias_conf = alias_conf or field_alias()
    if "TERM_ID" in df.columns:
        current = df["TERM_ID"].astype("string")
    else:
        current = pd.Series(pd.NA, index=df.index, dtype="string")

    term_conf = alias_conf.get("term_fields", {})
    year_cols = [c for c in term_conf.get("academic_year", []) if c in df.columns]
    sem_cols = [c for c in term_conf.get("semester", []) if c in df.columns]
    date_cols = [c for c in term_conf.get("date", []) if c in df.columns]

    for y_col in year_cols:
        exact_term = df[y_col].map(term_from_value).astype("string")
        current = current.combine_first(exact_term)
        years = df[y_col].map(academic_year_from_value)
        for s_col in sem_cols:
            semesters = df[s_col].map(semester_from_value)
            candidate = years.astype("string") + "-" + semesters.astype("string")
            candidate = candidate.where(years.notna() & semesters.notna(), pd.NA)
            current = current.combine_first(candidate.astype("string"))
        if not sem_cols:
            candidate = years.astype("string") + "-1"
            candidate = candidate.where(years.notna(), pd.NA)
            current = current.combine_first(candidate.astype("string"))

    for d_col in date_cols:
        candidate = df[d_col].map(term_from_date).astype("string")
        current = current.combine_first(candidate)

    return current.astype("string")


def load_excel_standardized(dataset_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = file_registry()
    if dataset_name not in registry:
        raise KeyError(f"Unknown dataset: {dataset_name}")

    path = RAW_DIR / registry[dataset_name]["raw_file"]
    started = datetime.now()
    raw = read_excel_all_sheets(path)
    before_rows = len(raw)
    df = apply_standard_aliases(raw)
    df["SOURCE_FILE"] = path.name
    df["SOURCE_DATASET"] = dataset_name
    if "XH" in df.columns:
        df["XH"] = stringify_id(df["XH"])
    if "COURSE_STD" in df.columns:
        df["COURSE_STD"] = stringify_id(df["COURSE_STD"])
    if "EVENT_TIME" in df.columns:
        df["EVENT_TIME"] = to_datetime(df["EVENT_TIME"])
    df["TERM_ID"] = infer_term_id(df)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    log = {
        "dataset": dataset_name,
        "source_file": path.name,
        "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_rows": before_rows,
        "clean_rows": len(df),
        "columns": list(df.columns),
        "standard_fields_present": [c for c in STANDARD_FIELDS if c in df.columns],
        "term_id_missing_rate": float(df["TERM_ID"].isna().mean()) if "TERM_ID" in df else 1.0,
        "xh_missing_rate": float(df["XH"].isna().mean()) if "XH" in df else 1.0,
    }
    return df, log


def require_id_term(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "XH" not in df.columns or "TERM_ID" not in df.columns:
        return pd.DataFrame(columns=["XH", "TERM_ID"])
    result = df[df["XH"].notna() & df["TERM_ID"].notna()].copy()
    result["XH"] = stringify_id(result["XH"])
    result["TERM_ID"] = result["TERM_ID"].astype("string")
    return result


def status_abnormal(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="int64")
    text = series.astype("string").fillna("").str.lower()
    ok_words = "正常|已提交|完成|签到|出勤|在校|通过|success|submit|finish|done"
    bad_words = "缺勤|迟到|早退|旷课|未|异常|失败|退课|重修|补考|缓考|缺考|fail|late|absent|missing"
    bad = text.str.contains(bad_words, regex=True)
    ok = text.str.contains(ok_words, regex=True)
    return (bad & ~ok).astype(int)


def grouped_base(df: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    data = require_id_term(df)
    return data.groupby(["XH", "TERM_ID"], dropna=False)


def safe_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if left.empty:
        return right
    if right.empty:
        return left
    return left.merge(right, on=["XH", "TERM_ID"], how="left")


def term_sort_key(term_id: Any) -> tuple[int, int]:
    text = str(term_id)
    match = re.search(r"(20\d{2})\D+(?:20\d{2})\D+([12])", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"(20\d{2}).*?([12])?$", text)
    if match:
        return int(match.group(1)), int(match.group(2) or 1)
    return 0, 0


def latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    temp = df.copy()
    temp["_TERM_SORT"] = temp["TERM_ID"].map(term_sort_key)
    temp = temp.sort_values(["XH", "_TERM_SORT"])
    return temp.groupby("XH", as_index=False).tail(1).drop(columns=["_TERM_SORT"])


def collect_feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("FEATURE_")])


def ensure_feature_columns(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for col in features:
        if col not in result.columns:
            result[col] = np.nan
    return result


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True
