from __future__ import annotations

import numpy as np
import pandas as pd

from study_common import DWD_DIR, ensure_dirs, load_ods, require_id_term, status_abnormal, term_sort_key, to_numeric, write_parquet


def _apply_term_mapping(df: pd.DataFrame, mapping: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    if df.empty or mapping.empty or not set(keys).issubset(df.columns) or not set(keys).issubset(mapping.columns):
        return df
    prepared = mapping[keys + ["TERM_ID"]].dropna().drop_duplicates()
    if prepared.empty:
        return df
    prepared["_SORT"] = prepared["TERM_ID"].map(term_sort_key)
    prepared = prepared.sort_values("_SORT").drop_duplicates(keys, keep="last")
    result = df.merge(prepared.drop(columns=["_SORT"]).rename(columns={"TERM_ID": name}), on=keys, how="left")
    result["TERM_ID"] = result["TERM_ID"].combine_first(result[name])
    return result.drop(columns=[name], errors="ignore")


def enrich_term(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "TERM_ID" not in df.columns or df["TERM_ID"].notna().all():
        return df
    result = df.copy()
    class_task = require_id_term(load_ods("class_task"))
    result = _apply_term_mapping(result, class_task, ["XH", "COURSE_STD"], "TERM_FROM_CLASS_TASK")
    result = _apply_term_mapping(result, class_task, ["XH", "CLAZZ_ID"], "TERM_FROM_CLASS")
    selection = require_id_term(load_ods("course_selection"))
    result = _apply_term_mapping(result, selection, ["XH", "COURSE_STD"], "TERM_FROM_SELECTION")
    return result


def main() -> None:
    ensure_dirs()
    event = require_id_term(enrich_term(load_ods("assignment")))
    if event.empty:
        term = pd.DataFrame(columns=["XH", "TERM_ID"])
    else:
        event["SCORE_NUM"] = to_numeric(event.get("SCORE"))
        event["MISSING_FLAG"] = status_abnormal(event.get("STATUS"))
        term = event.groupby(["XH", "TERM_ID"], as_index=False).agg(
            FEATURE_ASSIGNMENT_COUNT=("XH", "count"),
            FEATURE_ASSIGNMENT_SCORE_AVG=("SCORE_NUM", "mean"),
            FEATURE_ASSIGNMENT_MISSING_COUNT=("MISSING_FLAG", "sum"),
        )
        term["FEATURE_ASSIGNMENT_SUBMIT_RATE"] = np.where(
            term["FEATURE_ASSIGNMENT_COUNT"] > 0,
            1 - term["FEATURE_ASSIGNMENT_MISSING_COUNT"] / term["FEATURE_ASSIGNMENT_COUNT"],
            np.nan,
        )

    write_parquet(event, DWD_DIR / "study_l5_assignment_event.parquet")
    write_parquet(term, DWD_DIR / "study_l5_assignment_term.parquet")
    print(f"l5 assignment written: event={len(event)} term={len(term)}")


if __name__ == "__main__":
    main()
