from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    role: str
    profile: str
    system_prompt: str
    capabilities: Sequence[str] = field(default_factory=tuple)
    metadata: Dict[str, str] = field(default_factory=dict)

    def as_text(self) -> str:
        caps = ",".join(self.capabilities)
        meta = ",".join(f"{k}:{v}" for k, v in sorted(self.metadata.items()))
        return " | ".join(
            [
                self.agent_id,
                self.role,
                self.profile,
                self.system_prompt,
                caps,
                meta,
            ]
        )


@dataclass
class AgentPool:
    agents: List[AgentSpec]

    def ids(self) -> List[str]:
        return [a.agent_id for a in self.agents]

    def by_id(self) -> Dict[str, AgentSpec]:
        return {a.agent_id: a for a in self.agents}

    def get(self, agent_id: str) -> AgentSpec:
        agents = self.by_id()
        if agent_id not in agents:
            raise KeyError(f"Unknown agent: {agent_id}")
        return agents[agent_id]

    def texts(self) -> Iterable[tuple[str, str]]:
        for agent in self.agents:
            yield agent.agent_id, agent.as_text()


def default_agent_pool() -> AgentPool:
    return AgentPool(
        agents=[
            AgentSpec(
                agent_id="planner",
                role="任务规划者",
                profile="拆解问题，决定解题步骤和信息流。",
                system_prompt="你负责先理解问题，再给出简洁可靠的分析计划。",
                capabilities=("planning", "routing", "synthesis"),
                metadata={"family": "coordination"},
            ),
            AgentSpec(
                agent_id="reasoner",
                role="通用推理专家",
                profile="擅长多步文本推理和中间结论组织。",
                system_prompt="你负责给出完整、可追踪的推理结论，避免跳步。",
                capabilities=("reasoning", "analysis"),
                metadata={"family": "reasoning"},
            ),
            AgentSpec(
                agent_id="math",
                role="数学求解专家",
                profile="擅长算术、代数和符号推理。",
                system_prompt="你负责精确计算和数学推理，优先确保结果正确。",
                capabilities=("math", "reasoning"),
                metadata={"family": "reasoning"},
            ),
            AgentSpec(
                agent_id="coder",
                role="代码与算法专家",
                profile="擅长程序思维、伪代码和执行步骤设计。",
                system_prompt="你负责把问题转成可执行的算法步骤或程序化思路。",
                capabilities=("code", "algorithm", "reasoning"),
                metadata={"family": "implementation"},
            ),
            AgentSpec(
                agent_id="researcher",
                role="知识整合专家",
                profile="擅长整合事实、背景知识和候选证据。",
                system_prompt="你负责基于已有信息整理事实，不编造不存在的依据。",
                capabilities=("knowledge", "retrieval", "analysis"),
                metadata={"family": "knowledge"},
            ),
            AgentSpec(
                agent_id="skeptic",
                role="反驳与质检专家",
                profile="擅长找漏洞、找矛盾、指出不充分论证。",
                system_prompt="你负责批判性检查已有答案，指出最可能的错误。",
                capabilities=("critique", "verification"),
                metadata={"family": "verification"},
            ),
            AgentSpec(
                agent_id="verifier",
                role="验证专家",
                profile="擅长核对答案是否满足题意、约束和格式要求。",
                system_prompt="你负责对候选答案做约束检查，并给出是否通过的明确判断。",
                capabilities=("verification", "format_check"),
                metadata={"family": "verification"},
            ),
            AgentSpec(
                agent_id="summarizer",
                role="汇总专家",
                profile="擅长从多个候选中提炼一致结论并压缩输出。",
                system_prompt="你负责综合多路结果，输出简洁一致的最终答案。",
                capabilities=("synthesis", "aggregation"),
                metadata={"family": "coordination"},
            ),
            AgentSpec(
                agent_id="debater_a",
                role="正方辩手",
                profile="擅长提出一个强而明确的候选答案。",
                system_prompt="你负责提出一个强候选解，并说明你最核心的依据。",
                capabilities=("reasoning", "debate"),
                metadata={"family": "debate"},
            ),
            AgentSpec(
                agent_id="debater_b",
                role="反方辩手",
                profile="擅长从不同角度提出替代答案或反例。",
                system_prompt="你负责提出与现有思路不同的候选解，并强调分歧点。",
                capabilities=("reasoning", "debate"),
                metadata={"family": "debate"},
            ),
        ]
    )
