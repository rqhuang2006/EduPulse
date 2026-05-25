from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from study_common import CONF_DIR, DOCS_DIR, RAW_DIR, ensure_dirs, field_alias, normalize_key, read_yaml, write_yaml


def guess_standard_field(raw_field: str, aliases: dict[str, list[str]]) -> tuple[str, float]:
    key = normalize_key(raw_field)
    best = ("", 0.0)
    for standard, names in aliases.items():
        for name in names:
            alias_key = normalize_key(name)
            if key == alias_key:
                return standard, 1.0
            if alias_key and (alias_key in key or key in alias_key):
                score = min(len(alias_key), len(key)) / max(len(alias_key), len(key))
                if score > best[1]:
                    best = (standard, score)
    rules = [
        (r"学号|XH|LOGIN|SID|XSBH|USERID|CARD", "XH", 0.75),
        (r"KCH|COURSE.*(ID|CODE)|课程.*(号|码)", "COURSE_STD", 0.75),
        (r"KCM|COURSE.*NAME|课程.*名", "COURSE_NAME", 0.75),
        (r"XF|CREDIT|学分", "CREDIT", 0.75),
        (r"SCORE|成绩|总分|KCCJ|BFB|CJ", "SCORE", 0.7),
        (r"TIME|DATE|RQ|SJ|时间|日期", "EVENT_TIME", 0.55),
        (r"STATUS|STATE|状态|标志", "STATUS", 0.55),
    ]
    for pattern, standard, score in rules:
        if re.search(pattern, str(raw_field), flags=re.I):
            return standard, score
    return best


def collect_fields() -> pd.DataFrame:
    inventory = DOCS_DIR / "raw_inventory.xlsx"
    if inventory.exists():
        return pd.read_excel(inventory, sheet_name="fields")

    rows: list[dict] = []
    for path in sorted(RAW_DIR.glob("*.xls*")):
        excel = pd.ExcelFile(path)
        for sheet in excel.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet, nrows=5)
            for col in df.columns:
                rows.append({"SOURCE_FILE": path.name, "SHEET_NAME": sheet, "RAW_FIELD": str(col)})
        excel.close()
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    conf = field_alias()
    aliases = conf.get("standard_fields", {})
    fields = collect_fields()
    rows = []
    for _, row in fields.iterrows():
        guess, confidence = guess_standard_field(str(row["RAW_FIELD"]), aliases)
        rows.append({**row.to_dict(), "STANDARD_GUESS": guess, "CONFIDENCE": confidence})

    mapping = pd.DataFrame(rows)
    output = DOCS_DIR / "raw_field_mapping.xlsx"
    with pd.ExcelWriter(output) as writer:
        mapping.to_excel(writer, sheet_name="mapping_draft", index=False)

    for _, row in mapping[mapping["CONFIDENCE"] >= 0.75].iterrows():
        standard = row["STANDARD_GUESS"]
        raw_field = str(row["RAW_FIELD"])
        if standard and raw_field not in aliases.setdefault(standard, []):
            aliases[standard].append(raw_field)

    conf["standard_fields"] = aliases
    write_yaml(conf, CONF_DIR / "field_alias.yaml")
    print(f"field mapping draft written: {output}")
    print(f"field alias updated: {CONF_DIR / 'field_alias.yaml'}")


if __name__ == "__main__":
    main()

