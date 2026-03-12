from __future__ import annotations

import re
from typing import Dict, Optional

from .agents import AgentSpec
from .types import PromptSlots


def prompt_slot_instructions(slots: PromptSlots) -> str:
    parts: list[str] = []
    reasoning = {
        "direct": "直接给出结论，不要展开冗长中间过程。",
        "stepwise": "请分步骤推理，但保持步骤精炼。",
        "critique_then_answer": "先指出潜在风险或歧义，再给出最终答案。",
    }
    upstream = {
        "summary": "优先总结上游信息后再推理。",
        "quote_then_reason": "先引用关键上游片段，再据此推理。",
    }
    output = {
        "raw": "输出自然语言答案。",
        "bullet": "输出使用简洁项目符号。",
        "json": "输出为紧凑JSON，字段必须稳定。",
    }
    verify = {
        "off": "不额外做显式验证。",
        "light": "在结尾快速自检一次。",
        "strict": "在结尾逐条检查约束是否满足。",
    }
    finalization = {
        "answer_only": "最后只保留结论。",
        "answer_with_rationale": "最后先给结论，再给一句简短理由。",
    }
    parts.append(reasoning.get(slots.reasoning_mode, ""))
    parts.append(upstream.get(slots.upstream_usage, ""))
    parts.append(output.get(slots.output_style, ""))
    parts.append(verify.get(slots.verification_mode, ""))
    parts.append(finalization.get(slots.finalization, ""))
    return "\n".join(p for p in parts if p)


def build_system_prompt(agent: AgentSpec, slots: PromptSlots, extra_role_hint: str = "") -> str:
    extra = f"\n你的当前子任务角色：{extra_role_hint}" if extra_role_hint else ""
    return f"{agent.system_prompt}{extra}\n{prompt_slot_instructions(slots)}".strip()


def _normalize_inline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def render_question_text(question_text: str, metadata: Optional[dict] = None) -> str:
    rendered = question_text.strip()
    if not metadata or not isinstance(metadata.get("options"), list):
        return rendered

    options = [str(item).strip() for item in metadata["options"] if str(item).strip()]
    if not options:
        return rendered

    normalized_question = _normalize_inline(rendered)
    matched = sum(1 for option in options if _normalize_inline(option) in normalized_question)
    if matched >= max(2, len(options) // 2):
        return rendered

    option_lines = [f"{idx}) {option}" for idx, option in enumerate(options, start=1)]
    return f"{rendered}\n\nOptions:\n" + "\n".join(option_lines)


def build_user_prompt(
    question_text: str,
    upstream_outputs: Dict[str, str],
    task_instruction: str,
    metadata: Optional[dict] = None,
    answer_contract: str = "",
) -> str:
    upstream = "\n".join(f"[{name}]\n{text}" for name, text in upstream_outputs.items() if text.strip())
    if not upstream:
        upstream = "无上游信息。"
    rendered_question = render_question_text(question_text, metadata=metadata)
    contract_block = f"\n输出约束：\n{answer_contract}\n" if answer_contract.strip() else ""
    return (
        f"问题：\n{rendered_question}\n\n"
        f"上游信息：\n{upstream}\n\n"
        f"当前任务：\n{task_instruction}\n"
        f"{contract_block}"
    )
