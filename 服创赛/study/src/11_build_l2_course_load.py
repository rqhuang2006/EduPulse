from __future__ import annotations

import pandas as pd

from study_common import DWD_DIR, ensure_dirs, load_ods, require_id_term, status_abnormal, to_numeric, write_parquet


def main() -> None:
    ensure_dirs()
    selection = require_id_term(load_ods("course_selection"))
    course_info = load_ods("course_info")
    if selection.empty:
        course = pd.DataFrame(columns=["XH", "TERM_ID"])
        term = pd.DataFrame(columns=["XH", "TERM_ID"])
    else:
        course = selection.copy()
        if "CREDIT" not in course.columns and not course_info.empty and {"COURSE_STD", "CREDIT"}.issubset(course_info.columns):
            info = course_info[["COURSE_STD", "CREDIT"]].drop_duplicates("COURSE_STD")
            course = course.merge(info, on="COURSE_STD", how="left")
        course["CREDIT_NUM"] = to_numeric(course.get("CREDIT"))
        course["RETAKE_FLAG"] = status_abnormal(course.get("STATUS"))
        term = course.groupby(["XH", "TERM_ID"], as_index=False).agg(
            FEATURE_COURSE_SELECTED_COUNT=("COURSE_STD", "nunique"),
            FEATURE_COURSE_CREDIT_SUM=("CREDIT_NUM", "sum"),
            FEATURE_COURSE_RETAKE_COUNT=("RETAKE_FLAG", "sum"),
        )

    write_parquet(course, DWD_DIR / "study_l2_course_load_course.parquet")
    write_parquet(term, DWD_DIR / "study_l2_course_load_term.parquet")
    print(f"l2 course load written: course={len(course)} term={len(term)}")


if __name__ == "__main__":
    main()

