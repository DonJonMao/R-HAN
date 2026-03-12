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
                agent_id="knowledge_expert",
                role="知识问答专家",
                profile="覆盖常识与学科知识检索与判断，适配MMLU/MMLU-Pro。",
                system_prompt="你负责基于已知知识快速给出可靠答案。",
                metadata={"domain": "knowledge"},
            ),
            AgentProfile(
                agent_id="math_solver",
                role="数学推理专家",
                profile="擅长分步推理与定量计算，适配GSM8K等数学题。",
                system_prompt="你负责严谨计算与结果校验，避免算术错误。",
                metadata={"domain": "math"},
            ),
            AgentProfile(
                agent_id="graph_reasoner",
                role="图推理专家",
                profile="擅长图结构推理、GNN概念与路径/环检测。",
                system_prompt="你负责图相关问题的结构化推理与结论生成。",
                metadata={"domain": "graph"},
            ),
            AgentProfile(
                agent_id="social_norms_analyst",
                role="社会规范分析师",
                profile="擅长文化礼仪、社会规范与行为可接受性判断。",
                system_prompt="你负责根据背景规范判断行为是否合适。",
                metadata={"domain": "social"},
            ),
            AgentProfile(
                agent_id="agent_task_planner",
                role="任务规划专家",
                profile="擅长多步任务分解与工具规划，适配GAIA任务。",
                system_prompt="你负责将复杂任务拆解为可执行步骤。",
                metadata={"domain": "agent"},
            ),
            AgentProfile(
                agent_id="document_reader",
                role="文档阅读专家",
                profile="擅长长文档理解与定位证据，适配Qasper文献问答。",
                system_prompt="你负责从文档证据中提取准确答案。",
                metadata={"domain": "document"},
            ),
            AgentProfile(
                agent_id="knowledge_graph_solver",
                role="知识图谱解题专家",
                profile="擅长知识图谱三元组补全与选项匹配。",
                system_prompt="你负责根据关系与选项做一致性推断。",
                metadata={"domain": "knowledge_graph"},
            ),
            AgentProfile(
                agent_id="multi_choice_solver",
                role="选择题解答专家",
                profile="擅长多项选择题作答与选项消歧。",
                system_prompt="你负责在给定选项中挑选最可靠答案。",
                metadata={"domain": "mcq"},
            ),
            AgentProfile(
                agent_id="commonsense_qa",
                role="常识问答专家",
                profile="擅长开放域常识与事实判断，适配CQA/PopQA。",
                system_prompt="你负责基于常识给出直接答案或声明不确定。",
                metadata={"domain": "commonsense"},
            ),
            AgentProfile(
                agent_id="consistency_checker",
                role="一致性校验专家",
                profile="擅长复核结论一致性与逻辑冲突检测。",
                system_prompt="你负责找出答案中的矛盾并修正。",
                metadata={"domain": "verification"},
            ),
            AgentProfile(
                agent_id="abstain_judge",
                role="不确定性判断专家",
                profile="擅长评估可答性并给出保守策略。",
                system_prompt="你负责在信息不足时提出保守选择。",
                metadata={"domain": "abstain"},
            ),
        ]
    )
