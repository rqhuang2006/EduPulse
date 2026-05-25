"""Sport domain agent entry point.

Re-exports the real SportAgent class from sport.sportagent.sport_agent.
This file is the formal import target: ``from sport.src.sport_agent import SportAgent``.
"""
from __future__ import annotations

from sport.sportagent.sport_agent import SportAgent, SportAgentResult  # noqa: F401

__all__ = ["SportAgent", "SportAgentResult"]
