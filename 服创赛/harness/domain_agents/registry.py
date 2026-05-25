from __future__ import annotations

from typing import Dict

from harness.domain_agents.base.base_domain_agent import BaseDomainAgent


class DomainAgentRegistry:
    def __init__(self) -> None:
        self._agents: Dict[str, BaseDomainAgent] = {}

    def register(self, agent: BaseDomainAgent) -> None:
        self._agents[agent.domain_name] = agent

    def get(self, domain_name: str) -> BaseDomainAgent:
        if domain_name not in self._agents:
            raise KeyError(f"Domain agent not registered: {domain_name}")
        return self._agents[domain_name]

    def has(self, domain_name: str) -> bool:
        return domain_name in self._agents

    def all_agents(self) -> Dict[str, BaseDomainAgent]:
        return dict(self._agents)