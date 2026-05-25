from __future__ import annotations

import pandas as pd

from study_common import DWD_DIR, ensure_dirs, load_ods, require_id_term, to_numeric, write_parquet


def main() -> None:
    ensure_dirs()
    grade = require_id_term(load_ods("grade"))
    if grade.empty:
        course = pd.DataFrame(columns=["XH", "TERM_ID"])
        term = pd.DataFrame(columns=["XH", "TERM_ID"])
    else:
        grade["SCORE_NUM"] = to_numeric(grade.get("SCORE"))
        grade["CREDIT_NUM"] = to_numeric(grade.get("CREDIT"))
        course_cols = [c for c in ["XH", "TERM_ID", "COURSE_STD", "COURSE_NAME", "CREDIT_NUM", "SCORE_NUM", "STATUS"] if c in grade.columns]
        course = grade[course_cols].copy()
        term = grade.groupby(["XH", "TERM_ID"], as_index=False).agg(
            FEATURE_GRADE_COURSE_COUNT=("SCORE_NUM", "count"),
            FEATURE_GRADE_AVG_SCORE=("SCORE_NUM", "mean"),
            FEATURE_GRADE_MIN_SCORE=("SCORE_NUM", "min"),
            FEATURE_GRADE_FAIL_COUNT=("SCORE_NUM", lambda s: int((s < 60).sum())),
            FEATURE_GRADE_CREDIT_SUM=("CREDIT_NUM", "sum"),
        )

    cet = require_id_term(load_ods("cet_score"))
    if not cet.empty:
        cet["CET_SCORE_NUM"] = to_numeric(cet.get("SCORE"))
        cet_term = cet.groupby(["XH", "TERM_ID"], as_index=False).agg(FEATURE_CET_SCORE_MAX=("CET_SCORE_NUM", "max"))
        term = term.merge(cet_term, on=["XH", "TERM_ID"], how="outer") if not term.empty else cet_term
    elif "FEATURE_CET_SCORE_MAX" not in term.columns:
        term["FEATURE_CET_SCORE_MAX"] = pd.NA

    write_parquet(course, DWD_DIR / "study_l1_grade_course.parquet")
    write_parquet(term, DWD_DIR / "study_l1_grade_term.parquet")
    print(f"l1 grade written: course={len(course)} term={len(term)}")


if __name__ == "__main__":
    main()

