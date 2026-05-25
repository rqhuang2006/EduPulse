from __future__ import annotations

from typing import Any


def validate_eval_report_schema(
    *,
    report_name: str,
    payload: dict[str, Any],
    required_fields: list[str],
) -> dict[str, Any]:
    missing_fields = [field for field in required_fields if field not in payload]
    return {
        "report_name": report_name,
        "required_fields": required_fields,
        "missing_fields": missing_fields,
        "schema_ok": not missing_fields,
    }
