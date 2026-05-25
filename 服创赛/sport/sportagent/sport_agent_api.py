from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .sport_agent import SportAgent


def run_with_request_file(request_path: Path) -> Dict[str, Any]:
    agent = SportAgent.from_request_file(request_path)
    return agent.run()
