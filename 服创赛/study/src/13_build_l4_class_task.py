from __future__ import annotations

import pandas as pd

from study_common import DWD_DIR, ensure_dirs, load_ods, require_id_term, to_numeric, write_parquet


def first_existing(df: pd.DataFrame, names: list[str]) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return df[name]
    return None


def main() -> None:
    ensure_dirs()
    event = require_id_term(load_ods("class_task"))
    if event.empty:
        term = pd.DataFrame(columns=["XH", "TERM_ID"])
    else:
        event["TASK_RATE_NUM"] = to_numeric(first_existing(event, ["JOB_RATE", "TASK_RATE", "INTERACTION_RATE"]))
        event["VIDEO_RATE_NUM"] = to_numeric(first_existing(event, ["VIDEOJOB_RATE", "VIDEO_RATE"]))
        term = event.groupby(["XH", "TERM_ID"], as_index=False).agg(
            FEATURE_CLASS_TASK_COUNT=("XH", "count"),
            FEATURE_CLASS_TASK_RATE_AVG=("TASK_RATE_NUM", "mean"),
            FEATURE_CLASS_VIDEO_RATE_AVG=("VIDEO_RATE_NUM", "mean"),
        )

    write_parquet(event, DWD_DIR / "study_l4_class_task_event.parquet")
    write_parquet(term, DWD_DIR / "study_l4_class_task_term.parquet")
    print(f"l4 class task written: event={len(event)} term={len(term)}")


if __name__ == "__main__":
    main()

