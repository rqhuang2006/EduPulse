from __future__ import annotations

from typing import Any


def build_agent_protocol(
    *,
    domain: str,
    diagnosis: dict[str, Any],
    proposal: dict[str, Any],
    comparison: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "domain": domain,
        "protocol_version": "agent_protocol_v1",
        "diagnosis": diagnosis,
        "proposal": proposal,
        "comparison": comparison,
        "recommendation": recommendation,
    }
