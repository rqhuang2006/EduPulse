from __future__ import annotations

import pandas as pd

from study_common import DWD_DIR, ensure_dirs, latest_rows, load_ods, require_id_term, stringify_id, write_parquet


def main() -> None:
    ensure_dirs()
    frames = []
    for dataset in ["course_selection", "grade", "attendance_summary", "signin", "assignment", "exam_quiz", "class_task", "online_activity"]:
        df = require_id_term(load_ods(dataset))
        if not df.empty:
            frames.append(df[["XH", "TERM_ID"]])

    if frames:
        base = pd.concat(frames, ignore_index=True).drop_duplicates()
    else:
        base = pd.DataFrame(columns=["XH", "TERM_ID"])

    student = load_ods("student_info")
    if not student.empty and "XH" in student.columns:
        keep = [c for c in ["XH", "XB", "MZMC", "ZZMMMC", "CSRQ", "JG", "XSM", "ZYM"] if c in student.columns]
        student = student[keep].drop_duplicates("XH")
        student["XH"] = stringify_id(student["XH"])
        if base.empty:
            base = student[["XH"]].dropna().drop_duplicates()
            base["TERM_ID"] = pd.NA
        base = base.merge(student, on="XH", how="left")

    status = load_ods("status_change")
    invalid = require_id_term(status)
    if not invalid.empty:
        invalid["STATUS_CHANGE_FLAG"] = 1
        invalid["INVALID_TERM_FLAG"] = invalid.get("STATUS", "").astype("string").str.contains("退|休|停|异动|离", regex=True, na=False).astype(int)
        invalid = invalid[["XH", "TERM_ID", "STATUS_CHANGE_FLAG", "INVALID_TERM_FLAG"]].drop_duplicates(["XH", "TERM_ID"])
        base = base.merge(invalid, on=["XH", "TERM_ID"], how="left")

    for col in ["STATUS_CHANGE_FLAG", "INVALID_TERM_FLAG"]:
        if col not in base.columns:
            base[col] = 0
        base[col] = base[col].fillna(0).astype(int)

    base = base.dropna(subset=["XH", "TERM_ID"]).drop_duplicates(["XH", "TERM_ID"])
    output = DWD_DIR / "study_l0_student_term_base.parquet"
    write_parquet(base, output)
    print(f"student-term base written: {output} rows={len(base)}")


if __name__ == "__main__":
    main()

