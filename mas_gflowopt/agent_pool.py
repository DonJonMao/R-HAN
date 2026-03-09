from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .types import AgentProfile
from .vectorizer import HashingVectorizer


@dataclass
class AgentPool:
    agents: List[AgentProfile]

    def vectorize(self, vectorizer: HashingVectorizer) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for agent in self.agents:
            vec = vectorizer.vectorize_agent(agent)
            agent.vector = vec
            out[agent.agent_id] = vec
        return out

    def as_nodes(self) -> List[str]:
        return [agent.agent_id for agent in self.agents]

    def as_node_records(self) -> Iterable[Dict[str, str]]:
        for agent in self.agents:
            yield {
                "id": agent.agent_id,
                "role": agent.role,
                "profile": agent.profile,
            }


def default_agent_pool() -> AgentPool:
    return AgentPool(
        agents=[
            AgentProfile(
                agent_id="algo_designer",
                role="算法设计师",
                profile="擅长算法架构、复杂度分析、搜索策略设计。",
                system_prompt="你负责提出可实现且可验证的算法流程与结构。",
                metadata={"domain": "algorithm"},
            ),
            AgentProfile(
                agent_id="doctor",
                role="医生",
                profile="具备临床知识、诊断流程与医疗风险评估能力。",
                system_prompt="你负责医疗语境下的决策可靠性与安全性判断。",
                metadata={"domain": "medicine"},
            ),
            AgentProfile(
                agent_id="programmer",
                role="程序员",
                profile="擅长工程实现、代码重构、系统集成。",
                system_prompt="你负责将方案拆解为可执行的软件模块与接口。",
                metadata={"domain": "software"},
            ),
            AgentProfile(
                agent_id="mathematician",
                role="数学家",
                profile="擅长建模、证明、优化理论与数值分析。",
                system_prompt="你负责形式化建模与优化路径推导。",
                metadata={"domain": "math"},
            ),
            AgentProfile(
                agent_id="finance_expert",
                role="金融学家",
                profile="擅长风险收益分析、决策组合与市场逻辑。",
                system_prompt="你负责金融场景中的策略评估和风险约束。",
                metadata={"domain": "finance"},
            ),
        ]
    )
