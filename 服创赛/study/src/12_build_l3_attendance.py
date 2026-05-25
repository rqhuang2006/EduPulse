from __future__ import annotations

import numpy as np
import pandas as pd

from study_common import DWD_DIR, ensure_dirs, load_ods, require_id_term, status_abnormal, to_numeric, write_parquet


def main() -> None:
    ensure_dirs()
    frames = []
    for dataset in ["attendance_summary", "signin"]:
        df = require_id_term(load_ods(dataset))
        if not df.empty:
            df["ATTENDANCE_ABNORMAL_FLAG"] = status_abnormal(df.get("STATUS"))
            frames.append(df)
    event = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=["XH", "TERM_ID"])

    if event.empty:
        term = pd.DataFrame(columns=["XH", "TERM_ID"])
    else:
        term = event.groupby(["XH", "TERM_ID"], as_index=False).agg(
            FEATURE_ATTENDANCE_EVENT_COUNT=("XH", "count"),
            FEATURE_ATTENDANCE_ABNORMAL_COUNT=("ATTENDANCE_ABNORMAL_FLAG", "sum"),
        )
        term["FEATURE_ATTENDANCE_ABNORMAL_RATE"] = np.where(
            term["FEATURE_ATTENDANCE_EVENT_COUNT"] > 0,
            term["FEATURE_ATTENDANCE_ABNORMAL_COUNT"] / term["FEATURE_ATTENDANCE_EVENT_COUNT"],
            np.nan,
        )

    session = require_id_term(load_ods("class_session"))
    if not session.empty:
        session["HEAD_UP_NUM"] = to_numeric(session.get("HEAD_UP_RATE"))
        session["FRONT_ROW_NUM"] = to_numeric(session.get("FRONT_ROW_RATE"))
        session_term = session.groupby(["XH", "TERM_ID"], as_index=False).agg(
            FEATURE_CLASS_HEAD_UP_RATE_AVG=("HEAD_UP_NUM", "mean"),
            FEATURE_CLASS_FRONT_ROW_RATE_AVG=("FRONT_ROW_NUM", "mean"),
        )
        term = term.merge(session_term, on=["XH", "TERM_ID"], how="outer") if not term.empty else session_term
        event = pd.concat([event, session], ignore_index=True, sort=False)

    write_parquet(event, DWD_DIR / "study_l3_attendance_event.parquet")
    write_parquet(term, DWD_DIR / "study_l3_attendance_term.parquet")
    print(f"l3 attendance written: event={len(event)} term={len(term)}")


if __name__ == "__main__":
    main()

