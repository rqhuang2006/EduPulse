from __future__ import annotations

import numpy as np
import pandas as pd

from study_common import DWD_DIR, ensure_dirs, term_sort_key, write_parquet


def main() -> None:
    ensure_dirs()
    grade_path = DWD_DIR / "study_l1_grade_term.parquet"
    if not grade_path.exists():
        label = pd.DataFrame(columns=["XH", "TERM_ID", "NEXT_TERM_ID", "LABEL", "LABEL_SUBTYPE", "LABEL_REASON"])
    else:
        grade = pd.read_parquet(grade_path)
        if grade.empty:
            label = pd.DataFrame(columns=["XH", "TERM_ID", "NEXT_TERM_ID", "LABEL", "LABEL_SUBTYPE", "LABEL_REASON"])
        else:
            future = grade[["XH", "TERM_ID", "FEATURE_GRADE_AVG_SCORE", "FEATURE_GRADE_FAIL_COUNT"]].copy()
            future["_SORT"] = future["TERM_ID"].map(term_sort_key)
            future = future.sort_values(["XH", "_SORT"])
            future["PREV_TERM_ID"] = future.groupby("XH")["TERM_ID"].shift(1)
            future["LABEL"] = (
                (future["FEATURE_GRADE_AVG_SCORE"] < 60) | (future["FEATURE_GRADE_FAIL_COUNT"].fillna(0) > 0)
            ).astype(int)
            fail_count = future["FEATURE_GRADE_FAIL_COUNT"].fillna(0)
            avg_score = future["FEATURE_GRADE_AVG_SCORE"]
            future["LABEL_SUBTYPE"] = np.where(
                future["LABEL"] == 0,
                "normal",
                np.where(avg_score < 60, "overall_low", "single_fail"),
            )
            future["LABEL_REASON"] = np.where(
                future["LABEL_SUBTYPE"] == "overall_low",
                "next_term_overall_low",
                np.where(
                    future["LABEL_SUBTYPE"] == "single_fail",
                    "next_term_single_fail",
                    "next_term_normal",
                ),
            )
            label = future.rename(columns={"PREV_TERM_ID": "TERM_ID", "TERM_ID": "NEXT_TERM_ID"})
            label = label.dropna(subset=["TERM_ID"])[["XH", "TERM_ID", "NEXT_TERM_ID", "LABEL", "LABEL_SUBTYPE", "LABEL_REASON"]]

    output = DWD_DIR / "study_label_table.parquet"
    write_parquet(label, output)
    print(f"label table written: {output} rows={len(label)}")


if __name__ == "__main__":
    main()

