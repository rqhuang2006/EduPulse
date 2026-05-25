from __future__ import annotations

import pandas as pd

from study_common import DOCS_DIR, ensure_dirs, file_registry, load_excel_standardized, ods_path_for, write_json, write_parquet


def main() -> None:
    ensure_dirs()
    logs = []
    for dataset_name in file_registry():
        df, log = load_excel_standardized(dataset_name)
        output = ods_path_for(dataset_name)
        write_parquet(df, output)
        logs.append(log)
        print(f"{dataset_name}: {len(df)} rows -> {output.name}")

    log_df = pd.DataFrame(logs)
    log_df.to_csv(DOCS_DIR / "ods_cleaning_log.csv", index=False, encoding="utf-8-sig")
    write_json({"datasets": logs}, DOCS_DIR / "ods_cleaning_log.json")
    print(f"cleaning logs written: {DOCS_DIR}")


if __name__ == "__main__":
    main()

