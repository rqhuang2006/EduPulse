from __future__ import annotations

import pandas as pd

from study_common import DWD_DIR, ensure_dirs, load_ods, require_id_term, to_numeric, write_parquet


def main() -> None:
    ensure_dirs()
    online = require_id_term(load_ods("online_activity"))
    if not online.empty:
        online["ONLINE_SCORE_NUM"] = to_numeric(online.get("SCORE"))

    library = require_id_term(load_ods("library_visit"))
    if not library.empty:
        library["LIBRARY_VISIT_FLAG"] = 1

    event = pd.concat([online, library], ignore_index=True, sort=False) if not online.empty or not library.empty else pd.DataFrame(columns=["XH", "TERM_ID"])

    term_parts = []
    if not online.empty:
        term_parts.append(online.groupby(["XH", "TERM_ID"], as_index=False).agg(FEATURE_ONLINE_ACTIVITY_SCORE_AVG=("ONLINE_SCORE_NUM", "mean")))
    if not library.empty:
        term_parts.append(library.groupby(["XH", "TERM_ID"], as_index=False).agg(FEATURE_LIBRARY_VISIT_COUNT=("LIBRARY_VISIT_FLAG", "sum")))

    if term_parts:
        term = term_parts[0]
        for part in term_parts[1:]:
            term = term.merge(part, on=["XH", "TERM_ID"], how="outer")
    else:
        term = pd.DataFrame(columns=["XH", "TERM_ID"])

    write_parquet(event, DWD_DIR / "study_l7_online_activity_event.parquet")
    write_parquet(term, DWD_DIR / "study_l7_online_activity_term.parquet")
    print(f"l7 online activity written: event={len(event)} term={len(term)}")


if __name__ == "__main__":
    main()

