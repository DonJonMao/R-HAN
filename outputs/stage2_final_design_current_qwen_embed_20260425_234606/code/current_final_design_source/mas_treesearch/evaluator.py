from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from math import sqrt
from typing import Dict, List, Optional, Tuple

from .agents import AgentPool
from .cache import DictCache
from .clients import OpenAICompatClient
from .config import ChatConfig, TierRuntimeConfig, TieredEvalConfig
from .profiles import DEFAULT_PROFILE, DatasetProfile
from .prompting import build_system_prompt, build_user_prompt, render_question_text
from .reward import reward_from_evaluation, reward_weights_for_profile
from .types import ArchitectureState, CompiledArchitecture, EvalSummary, PromptSlots, TaskEvaluation


@dataclass
class ExecutionContext:
    role_outputs: Dict[str, str]
    trace: List[Dict[str, str]]


class MultiFidelityEvaluator:
    """Runs tier-1 / tier-2 evaluation using the deployed local LLM."""

    def __init__(self, config: TieredEvalConfig, agent_pool: AgentPool):
        self.config = config
        self.agent_pool = agent_pool
        self._by_id = agent_pool.by_id()
        self._chat = OpenAICompatClient(config.chat)
        judge_chat_config = ChatConfig(
            api_base=config.judge.api_base or config.chat.api_base,
            model=config.judge.model or config.chat.model,
            api_key=config.judge.api_key or config.chat.api_key,
            timeout_s=config.judge.timeout_s,
            temperature=config.judge.temperature,
            max_tokens=config.judge.max_tokens,
            max_retries=config.chat.max_retries,
        )
        self._judge = OpenAICompatClient(judge_chat_config)
        self._chat_cache: DictCache[str] = DictCache()
        self._eval_cache: DictCache[EvalSummary] = DictCache()

    @staticmethod
    def _safe_json(text: str) -> Optional[dict]:
        try:
            return json.loads(text)
        except Exception:
            pass
        if "{" in text and "}" in text:
            snippet = text[text.find("{") : text.rfind("}") + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return None
        return None

    def _cached_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        runtime: TierRuntimeConfig,
        model: Optional[str] = None,
        cacheable: bool = True,
        client: str = "chat",
    ) -> str:
        key = json.dumps(
            {
                "client": client,
                "messages": messages,
                "temperature": runtime.temperature,
                "max_tokens": runtime.max_tokens,
                "model": model or (self.config.judge.model if client == "judge" else self.config.chat.model),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        if cacheable and self.config.enable_chat_cache:
            cached = self._chat_cache.get(key)
            if cached is not None:
                return cached
        active_client = self._judge if client == "judge" else self._chat
        value = active_client.chat(
            messages,
            temperature=runtime.temperature,
            max_tokens=runtime.max_tokens,
            model=model,
        )
        if cacheable and self.config.enable_chat_cache:
            self._chat_cache.put(key, value)
        return value

    def _resolve_runtime(self, tier: str, dataset_profile: DatasetProfile) -> TierRuntimeConfig:
        runtime = self.config.tier1 if tier == "tier1" else self.config.tier2
        overrides = dataset_profile.runtime_overrides
        if tier == "tier1":
            return replace(
                runtime,
                repeats=overrides.tier1_repeats or runtime.repeats,
                max_tokens=overrides.tier1_max_tokens or runtime.max_tokens,
                judge_max_tokens=overrides.tier1_judge_max_tokens or runtime.judge_max_tokens,
            )
        return replace(
            runtime,
            repeats=overrides.tier2_repeats or runtime.repeats,
            max_tokens=overrides.tier2_max_tokens or runtime.max_tokens,
            judge_max_tokens=overrides.tier2_judge_max_tokens or runtime.judge_max_tokens,
        )

    def _role_task_instruction(
        self,
        compiled: CompiledArchitecture,
        role: str,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
        metadata: Optional[dict] = None,
    ) -> str:
        template = compiled.state.template.value
        mapping = {
            "solver": "给出一个尽可能正确的候选答案。",
            "solver_a": "独立给出一个候选答案。",
            "solver_b": "从不同角度独立给出另一个候选答案。",
            "aggregator": "综合多路候选，输出最终答案。若题目给了固定输出格式，必须严格只输出该格式。",
            "generator": "先生成初稿答案。",
            "critic": "指出初稿最可能存在的错误或遗漏。",
            "reviser": "根据初稿和批评意见产出修订版最终答案。若题目要求固定格式，严格只输出该格式。",
            "verifier": "检查候选答案是否满足题意、约束和格式。若通过，只输出清洗后的最终答案，不要附加检查说明。",
            "judge": "比较两位辩手的观点，给出更可靠的最终结论。若题目要求固定格式，严格只输出该格式。",
            "router": "分析问题重点，决定求解方向。",
        }
        extra: List[str] = []
        if dataset_profile.task_type == "code_generation":
            entry_point = str((metadata or {}).get("entry_point") or "").strip()
            extra.append("这是 Python 代码生成任务，优先给出完整、可执行、对边界条件稳健的实现。")
            if entry_point:
                extra.append(f"必须实现且保留函数名 `{entry_point}`。")
            if role in {"critic", "verifier", "judge"}:
                extra.append("重点检查函数签名、返回值、异常边界、空输入和隐藏测试失败风险。")
            else:
                extra.append("不要输出解释性文字、Markdown 代码块或伪代码。")
        elif dataset_profile.task_type == "mcq":
            extra.append("这是多项选择题。重点比较候选选项差异，避免输出题外解释。")
            if role in {"verifier", "judge"}:
                extra.append("优先检查选项编号是否合法、是否存在格式污染、以及是否误选了保守/弃答选项。")
        elif dataset_profile.task_type in {"numeric", "math_expression"}:
            extra.append("优先确保结论正确，再考虑表达简洁。")
            if role in {"verifier", "critic"}:
                extra.append("重点检查数字、符号、边界条件和最后化简形式。")
        elif dataset_profile.task_type == "graph_reasoning":
            extra.append("这是图推理任务。优先保证结构约束满足，再考虑语言自然度。")
            if role in {"verifier", "judge"}:
                extra.append("重点检查 JSON 结构、节点顺序、边约束和数值一致性。")
        elif dataset_profile.task_type == "structured_list":
            extra.append("这是结构化填空任务。必须保持元素顺序稳定，避免多余文本。")
        elif dataset_profile.task_type == "boolean":
            extra.append("这是是非判断任务。最终只能给出 yes 或 no，不要模糊表达。")
        elif dataset_profile.answer_format in {"short_span", "question_defined"}:
            extra.append("保持答案短且精确，优先满足题目显式格式要求。")
        suffix = " ".join(extra).strip()
        if suffix:
            suffix = " " + suffix
        return f"当前架构模板：{template}。{mapping.get(role, '完成你在当前架构中的子任务。')}{suffix}"

    @staticmethod
    def _extract_assert_examples(text: str, *, limit: int = 2) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        lines = [line.strip() for line in text.splitlines() if "assert" in line]
        return lines[:limit]

    def _task_context(
        self,
        question_text: str,
        *,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
        metadata: Optional[dict] = None,
        role: str = "",
    ) -> str:
        metadata = metadata or {}
        lines: List[str] = []
        if dataset_profile.task_type == "code_generation":
            entry_point = str(metadata.get("entry_point") or "").strip()
            if entry_point:
                lines.append(f"Required function: `{entry_point}`.")
            visible_tests: List[str] = []
            if isinstance(metadata.get("test_list"), list):
                test_list = [str(item).strip() for item in metadata["test_list"] if str(item).strip()]
                if test_list:
                    lines.append(f"Visible tests: {len(test_list)}.")
                    visible_tests = test_list[:2]
            elif isinstance(metadata.get("test"), str):
                visible_tests = self._extract_assert_examples(str(metadata["test"]), limit=2)
                if visible_tests:
                    lines.append(f"Visible assertions: {len(self._extract_assert_examples(str(metadata['test']), limit=50))}.")
            if visible_tests and role in {"solver", "solver_a", "solver_b", "generator", "reviser", "verifier", "critic"}:
                lines.append("Example visible checks:")
                lines.extend(f"- {item}" for item in visible_tests)
            if metadata.get("test_setup_code"):
                lines.append("There is helper setup code available in the tests; do not redefine conflicting names unless needed.")
        elif dataset_profile.task_type == "mcq":
            option_count = self._extract_mcq_option_range(question_text)
            if option_count <= 0 and isinstance(metadata.get("options"), list):
                option_count = len(metadata["options"])
            if option_count > 0:
                lines.append(f"Valid option range: 1 to {option_count}.")
            if "I Don't Know/ None of the above" in question_text:
                lines.append("Only choose the abstain option if other options are unsupported.")
        elif dataset_profile.task_type in {"numeric", "math_expression"}:
            lines.append("Keep track of the final scalar/expression exactly; intermediate reasoning may be discarded later.")
        elif dataset_profile.task_type == "graph_reasoning":
            task_name = str(metadata.get("task") or "").strip()
            if task_name:
                lines.append(f"Graph subtask: {task_name}.")
            lines.append("The final answer must preserve canonical JSON structure and respect graph constraints exactly.")
        elif dataset_profile.task_type == "structured_list":
            blanks = metadata.get("blanks")
            if isinstance(blanks, list) and blanks:
                lines.append(f"Blank count: {len(blanks)}. Preserve blank order exactly.")
        elif dataset_profile.task_type == "boolean":
            lines.append("Map the final judgement to a single lowercase token: yes or no.")
        elif dataset_profile.answer_format == "short_span":
            lines.append("The answer should be a short span copied or normalized from the most relevant evidence.")
        elif dataset_profile.answer_format == "question_defined":
            lines.append("Respect the explicit format required by the question; if unspecified, keep the answer minimal.")
        return "\n".join(lines)

    def _code_precheck_role(self, compiled: CompiledArchitecture) -> Optional[str]:
        preferred = ("reviser", "aggregator", "solver", "generator", "solver_a", "solver_b", "verifier", "judge")
        roles = set(compiled.execution_roles)
        for role in preferred:
            if role in roles:
                agent_id = compiled.state.role_to_agent.get(role, "")
                agent = self._by_id.get(agent_id)
                if agent and any(cap in agent.capabilities for cap in ("code", "algorithm", "reasoning")):
                    return role
        for role in compiled.execution_roles:
            if role in roles:
                return role
        return None

    def fast_code_precheck(
        self,
        compiled: CompiledArchitecture,
        question_text: str,
        *,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
    ) -> Optional[EvalSummary]:
        if dataset_profile.task_type != "code_generation":
            return None
        role = self._code_precheck_role(compiled)
        if not role:
            return None
        agent_id = compiled.state.role_to_agent.get(role, "")
        agent = self._by_id.get(agent_id)
        if agent is None:
            return None
        slots = compiled.state.role_to_prompt.get(role, PromptSlots())
        runtime = TierRuntimeConfig(
            max_tokens=min(160, max(96, self.config.tier1.max_tokens)),
            judge_max_tokens=min(96, self.config.judge.max_tokens),
            repeats=1,
            temperature=min(0.1, self.config.tier1.temperature),
        )
        answer_contract = self._output_contract(
            question_text,
            reference_answer=reference_answer,
            metadata=metadata,
        )
        task_context = self._task_context(
            question_text,
            dataset_profile=dataset_profile,
            metadata=metadata,
            role=role,
        )
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(agent, slots, extra_role_hint=f"{role}_fast_precheck"),
            },
            {
                "role": "user",
                "content": build_user_prompt(
                    question_text=question_text,
                    upstream_outputs={},
                    task_instruction=(
                        "快速给出一个可执行的候选 Python 实现，用于预检。"
                        "优先保证函数签名、基础正确性和可见测试通过率；不要输出解释。"
                    ),
                    metadata=metadata,
                    task_context=task_context,
                    answer_contract=answer_contract,
                ),
            },
        ]
        start = time.perf_counter()
        output = self._cached_chat(messages, runtime=runtime)
        latency = time.perf_counter() - start
        token_cost = sum(len(message["content"].split()) for message in messages) * self.config.token_cost_per_word
        return self.evaluate_output(
            question_text,
            output,
            tier="precheck",
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=dataset_profile,
            trace=[{"role": role, "agent_id": agent_id, "content": output}],
            custom_metrics={"precheck": 1.0},
            latency=latency,
            token_cost=token_cost,
        )

    @staticmethod
    def _task_type(question_text: str, reference_answer: Optional[str] = None, metadata: Optional[dict] = None) -> str:
        if metadata and isinstance(metadata.get("mas_task_type"), str) and metadata.get("mas_task_type", "").strip():
            return str(metadata["mas_task_type"]).strip()
        text = question_text.lower()
        if metadata and isinstance(metadata.get("options"), list):
            return "mcq"
        if reference_answer:
            ref = reference_answer.strip()
            if re.fullmatch(r"[A-Za-z]", ref):
                return "mcq"
        if "multiple choice question" in text or "\n options:" in text or "option -" in text:
            return "mcq"
        if "####" in (reference_answer or ""):
            return "numeric"
        return "generic"

    @staticmethod
    def _answer_format(metadata: Optional[dict]) -> str:
        if metadata and isinstance(metadata.get("mas_answer_format"), str):
            return str(metadata["mas_answer_format"]).strip()
        return ""

    @staticmethod
    def _extract_mcq_option_range(question_text: str) -> int:
        options = []
        for line in question_text.splitlines():
            line = line.strip()
            if not line:
                continue
            prefix = line.split(")", 1)[0].strip()
            if prefix.isdigit():
                options.append(int(prefix))
        return max(options) if options else 0

    @staticmethod
    def _strip_hidden_reasoning(text: str) -> str:
        cleaned = re.sub(r"(?is)<think>.*?</think>", "", text)
        cleaned = re.sub(r"(?im)^\s*</?think>\s*$", "", cleaned)
        return cleaned.strip()

    @staticmethod
    def _token_to_option(token: str, max_option: int) -> Optional[int]:
        raw = token.strip().upper().rstrip(".")
        if not raw:
            return None
        if raw.isdigit():
            value = int(raw)
        elif len(raw) == 1 and raw.isalpha():
            value = ord(raw) - ord("A") + 1
        else:
            return None
        if max_option > 0 and not (1 <= value <= max_option):
            return None
        return value

    def _normalize_mcq_output(self, question_text: str, final_output: str, metadata: Optional[dict] = None) -> str:
        cleaned = self._strip_hidden_reasoning(final_output)
        max_option = self._extract_mcq_option_range(question_text)
        if max_option <= 0 and metadata and isinstance(metadata.get("options"), list):
            max_option = len(metadata["options"])
        option_line = ""
        confidence_line = ""
        for raw in cleaned.splitlines():
            line = raw.strip()
            upper = line.upper()
            if upper.startswith("OPTION"):
                option_line = line
            elif upper.startswith("CONFIDENCE"):
                confidence_line = line
        if not option_line:
            text_candidates = [cleaned]
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            if lines:
                text_candidates.append(lines[-1])
            patterns = [
                r"OPTION\s*[:：-]?\s*([A-Z]|\d+)",
                r"(?:FINAL ANSWER|ANSWER|答案|最终答案)\s*[:：-]?\s*([A-Z]|\d+)",
                r"\b([A-Z])\b",
                r"\b(\d+)\b",
            ]
            parsed_option: Optional[int] = None
            for candidate in text_candidates:
                for pattern in patterns:
                    match = re.search(pattern, candidate, flags=re.IGNORECASE)
                    if not match:
                        continue
                    parsed_option = self._token_to_option(match.group(1), max_option)
                    if parsed_option is not None:
                        option_line = f"OPTION - {parsed_option}"
                        break
                if option_line:
                    break
        if not confidence_line:
            match = re.search(r"CONFIDENCE\s*[:：-]?\s*(\d+)", cleaned, flags=re.IGNORECASE)
            if match:
                confidence_line = f"CONFIDENCE - {match.group(1)}"
        if option_line and confidence_line:
            return f"{option_line}\n{confidence_line}"
        if option_line:
            return option_line
        return cleaned

    def _judge_prompt(
        self,
        question_text: str,
        final_output: str,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> tuple[str, str]:
        task_type = self._task_type(question_text, reference_answer=reference_answer, metadata=metadata)
        if task_type == "mcq":
            max_option = self._extract_mcq_option_range(question_text)
            if max_option <= 0 and metadata and isinstance(metadata.get("options"), list):
                max_option = len(metadata["options"])
            normalized = self._normalize_mcq_output(question_text, final_output, metadata=metadata)
            judge_prompt = (
                "你是一个严格评审。当前任务是高风险多项选择题。"
                "请重点检查：1. 答案是否只包含要求的格式；2. 选项编号是否在给定范围内；"
                "3. 是否存在多余解释；4. 该选项是否在医学/常识/学科知识层面看起来正确；"
                "5. 若选择“不知道/以上皆非”，只有在确实无法支持其他选项时才允许高分。"
                "返回 JSON："
                '{"task_score":0到1之间的小数,"success":0或1,"safety_penalty":0到1之间的小数,"brief_reason":"简短说明"}'
            )
            judge_input = (
                f"题目：\n{question_text}\n\n"
                f"候选答案（已规范化）：\n{normalized}\n\n"
                f"给定选项范围：1 到 {max_option}。"
            )
            if reference_answer:
                judge_input += f"\n\n标准答案参考：{reference_answer}"
            return judge_prompt, judge_input
        judge_prompt = (
            "你是一个严格评审。请根据问题与候选答案给出 JSON："
            '{"task_score":0到1之间的小数,"success":0或1,"safety_penalty":0到1之间的小数,"brief_reason":"简短说明"}'
        )
        judge_input = f"问题：\n{question_text}\n\n候选答案：\n{final_output}"
        if reference_answer:
            judge_input += f"\n\n参考答案：\n{reference_answer}"
        return judge_prompt, judge_input

    @staticmethod
    def _parse_mcq_answer(final_output: str) -> Tuple[Optional[int], Optional[int]]:
        option = None
        confidence = None
        for raw in final_output.splitlines():
            line = raw.strip()
            upper = line.upper()
            if upper.startswith("OPTION"):
                try:
                    option = int(line.split("-", 1)[1].strip())
                except Exception:
                    option = None
            elif upper.startswith("CONFIDENCE"):
                try:
                    confidence = int(line.split("-", 1)[1].strip())
                except Exception:
                    confidence = None
        return option, confidence

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.strip().replace("\r", "\n")
        text = re.sub(r"\s+", " ", text)
        return text.upper()

    @staticmethod
    def _normalize_yes_no_output(text: str) -> str:
        cleaned = MultiFidelityEvaluator._strip_hidden_reasoning(text).strip()
        match = re.search(r"\b(yes|no)\b", cleaned.lower())
        if match:
            return match.group(1)
        return cleaned

    @staticmethod
    def _extract_json_list(text: str) -> Optional[str]:
        cleaned = MultiFidelityEvaluator._strip_hidden_reasoning(text).strip()
        candidates = [cleaned]
        if "[" in cleaned and "]" in cleaned:
            candidates.append(cleaned[cleaned.find("[") : cleaned.rfind("]") + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, list):
                return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return None

    @staticmethod
    def _extract_first_number(text: str) -> Optional[float]:
        match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        if not match:
            return None
        try:
            return float(match.group(0))
        except Exception:
            return None

    @staticmethod
    def _extract_last_number(text: str) -> Optional[float]:
        matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        if not matches:
            return None
        try:
            return float(matches[-1])
        except Exception:
            return None

    @staticmethod
    def _gold_mcq_option(reference_answer: Optional[str], metadata: Optional[dict]) -> Optional[int]:
        if metadata:
            answer_index = metadata.get("answer_index")
            if isinstance(answer_index, int):
                return int(answer_index) + 1
        if not reference_answer:
            return None
        text = reference_answer.strip()
        match = re.search(r"OPTION\s*-\s*(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        if len(text) == 1 and text.isalpha():
            return ord(text.upper()) - ord("A") + 1
        if text.isdigit():
            return int(text)
        return None

    @staticmethod
    def _extract_gsm8k_target(reference_answer: str) -> Optional[float]:
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", reference_answer.replace(",", ""))
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        return MultiFidelityEvaluator._extract_last_number(reference_answer)

    @staticmethod
    def _extract_prediction_target(final_output: str) -> Optional[float]:
        cleaned = final_output.strip()
        match = re.search(r"(?:answer|答案)\s*[:：-]?\s*(-?\d+(?:\.\d+)?)", cleaned, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        return MultiFidelityEvaluator._extract_last_number(cleaned)

    @staticmethod
    def _dataset_name(metadata: Optional[dict]) -> str:
        if metadata and isinstance(metadata.get("mas_dataset_name"), str):
            return str(metadata["mas_dataset_name"]).strip()
        return ""

    @staticmethod
    def _extract_python_code(text: str) -> str:
        cleaned = MultiFidelityEvaluator._strip_hidden_reasoning(text).strip()
        fenced = re.findall(r"```(?:python)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            cleaned = max(fenced, key=len).strip()
        return cleaned.strip()

    @staticmethod
    def _extract_sequence_numbers(text: str) -> List[int]:
        return [int(token) for token in re.findall(r"-?\d+", text)]

    @staticmethod
    def _extract_boxed_expression(text: str) -> str:
        marker = r"\boxed{"
        idx = text.rfind(marker)
        if idx != -1:
            start = idx + len(marker)
            depth = 1
            out: List[str] = []
            for ch in text[start:]:
                if ch == "{":
                    depth += 1
                    out.append(ch)
                    continue
                if ch == "}":
                    depth -= 1
                    if depth == 0:
                        return "".join(out).strip()
                    out.append(ch)
                    continue
                out.append(ch)
        cleaned = MultiFidelityEvaluator._strip_hidden_reasoning(text)
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if lines:
            return lines[-1]
        return cleaned.strip()

    @staticmethod
    def _normalize_math_expression(text: str) -> str:
        expr = MultiFidelityEvaluator._extract_boxed_expression(text)
        expr = expr.replace("$", "").replace("\\left", "").replace("\\right", "")
        expr = expr.replace("\\dfrac", "\\frac")
        expr = expr.replace("\\text{ degrees}", "^\\circ").replace("\\text{degrees}", "^\\circ")
        expr = expr.replace("\\!", "").replace(",", "")
        expr = re.sub(r"\s+", "", expr)
        return expr.strip(".;")

    @staticmethod
    def _parse_nlgraph_edges(question_text: str) -> Dict[Tuple[int, int], int]:
        weighted_edges: Dict[Tuple[int, int], int] = {}
        for src, dst, weight in re.findall(
            r"edge (?:between|from node)\s+node\s+(\d+)\s+(?:and|to)\s+node\s+(\d+)\s+with weight\s+(-?\d+)",
            question_text,
            flags=re.IGNORECASE,
        ):
            weighted_edges[(int(src), int(dst))] = int(weight)
            weighted_edges[(int(dst), int(src))] = int(weight)
        for src, dst in re.findall(r"\((\d+),(\d+)\)", question_text):
            weighted_edges.setdefault((int(src), int(dst)), 1)
            weighted_edges.setdefault((int(dst), int(src)), 1)
        return weighted_edges

    @staticmethod
    def _path_weight(path: List[int], edges: Dict[Tuple[int, int], int]) -> Optional[int]:
        if len(path) < 2:
            return None
        total = 0
        for src, dst in zip(path, path[1:]):
            if (src, dst) not in edges:
                return None
            total += edges[(src, dst)]
        return total

    @staticmethod
    def _score_knowledge_crosswords(
        final_output: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
    ) -> Optional[Tuple[float, float, float, Dict[str, object]]]:
        gold = None
        if metadata and isinstance(metadata.get("answer_all"), list):
            gold = [str(item) for item in metadata["answer_all"]]
        elif reference_answer:
            try:
                parsed = json.loads(reference_answer)
                if isinstance(parsed, list):
                    gold = [str(item) for item in parsed]
            except Exception:
                gold = None
        if not gold:
            return None
        parsed_pred = MultiFidelityEvaluator._extract_json_list(final_output)
        if parsed_pred is None:
            return 0.0, 0.0, 0.8, {"used_dataset_specific": True, "format_ok": False}
        pred = [str(item) for item in json.loads(parsed_pred)]
        matches = sum(1 for p, g in zip(pred, gold) if p == g)
        accuracy = matches / max(1, len(gold))
        format_ok = len(pred) == len(gold)
        task_score = accuracy if format_ok else max(0.0, accuracy - 0.2)
        success = 1.0 if format_ok and accuracy >= 0.999 else 0.0
        safety_penalty = max(0.0, 0.6 * (1.0 - accuracy) + (0.15 if not format_ok else 0.0))
        return task_score, success, safety_penalty, {
            "used_dataset_specific": True,
            "format_ok": format_ok,
            "matches": matches,
            "total": len(gold),
        }

    @staticmethod
    def _score_code_generation(
        final_output: str,
        metadata: Optional[dict],
    ) -> Optional[Tuple[float, float, float, Dict[str, object]]]:
        if not metadata:
            return None
        candidate_code = MultiFidelityEvaluator._extract_python_code(final_output)
        if not candidate_code:
            return 0.0, 0.0, 0.9, {"used_dataset_specific": True, "error": "empty_code"}
        entry_point = metadata.get("entry_point")
        humaneval_test = metadata.get("test")
        mbpp_tests = metadata.get("test_list")
        setup_code = str(metadata.get("test_setup_code") or "")
        syntax_ok = False
        entry_defined = False
        exec_error = ""
        try:
            compile(candidate_code, "<candidate>", "exec")
            syntax_ok = True
            if isinstance(entry_point, str) and entry_point:
                entry_defined = bool(re.search(rf"def\s+{re.escape(entry_point)}\s*\(", candidate_code))
            else:
                entry_defined = True
        except Exception as exc:
            exec_error = str(exc)
        script_lines = [
            "import math",
            "import itertools",
            "import functools",
            "import collections",
            "import heapq",
            "import bisect",
            candidate_code,
        ]
        if isinstance(humaneval_test, str) and isinstance(entry_point, str) and entry_point:
            script_lines.extend(
                [
                    humaneval_test,
                    f"check({entry_point})",
                    "print('PASS')",
                ]
            )
        elif isinstance(mbpp_tests, list) and mbpp_tests:
            script_lines.append(setup_code)
            script_lines.append("passed = 0")
            script_lines.append(f"total = {len(mbpp_tests)}")
            for expr in mbpp_tests:
                escaped = json.dumps(str(expr))
                script_lines.extend(
                    [
                        f"_expr = {escaped}",
                        "try:",
                        "    exec(_expr, globals(), globals())",
                        "    passed += 1",
                        "except Exception:",
                        "    pass",
                    ]
                )
            script_lines.append("print(f'PASS_COUNT={passed}/{total}')")
        else:
            return None
        try:
            proc = subprocess.run(
                [sys.executable, "-c", "\n".join(script_lines)],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except subprocess.TimeoutExpired:
            partial = 0.12 * float(syntax_ok) + 0.18 * float(entry_defined)
            return partial, 0.0, 0.85 - 0.20 * float(syntax_ok) - 0.10 * float(entry_defined), {
                "used_dataset_specific": True,
                "error": "timeout",
                "syntax_ok": syntax_ok,
                "entry_defined": entry_defined,
                "exec_error": exec_error,
            }
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if "PASS_COUNT=" in stdout:
            match = re.search(r"PASS_COUNT=(\d+)/(\d+)", stdout)
            passed = int(match.group(1)) if match else 0
            total = int(match.group(2)) if match else max(1, len(mbpp_tests))
            accuracy = passed / max(1, total)
            task_score = min(1.0, 0.72 * accuracy + 0.12 * float(syntax_ok) + 0.16 * float(entry_defined))
            safety_penalty = 0.0 if accuracy >= 0.999 else max(
                0.0,
                0.55 * (1.0 - accuracy) - 0.08 * float(syntax_ok) - 0.05 * float(entry_defined),
            )
            return task_score, 1.0 if accuracy >= 0.999 else 0.0, safety_penalty, {
                "used_dataset_specific": True,
                "passed": passed,
                "total": total,
                "stderr": stderr,
                "syntax_ok": syntax_ok,
                "entry_defined": entry_defined,
                "exec_error": exec_error,
            }
        passed = proc.returncode == 0 and "PASS" in stdout
        if passed:
            return 1.0, 1.0, 0.0, {
                "used_dataset_specific": True,
                "stderr": stderr,
                "stdout": stdout,
                "syntax_ok": syntax_ok,
                "entry_defined": entry_defined,
            }
        partial = 0.15 * float(syntax_ok) + 0.20 * float(entry_defined)
        safety_penalty = max(0.0, 0.78 - 0.18 * float(syntax_ok) - 0.10 * float(entry_defined))
        return partial, 0.0, safety_penalty, {
            "used_dataset_specific": True,
            "stderr": stderr,
            "stdout": stdout,
            "syntax_ok": syntax_ok,
            "entry_defined": entry_defined,
            "exec_error": exec_error,
        }

    def _score_nlgraph(
        self,
        question_text: str,
        final_output: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
    ) -> Optional[Tuple[float, float, float, Dict[str, object]]]:
        if not metadata:
            return None
        task = str(metadata.get("task") or "").strip()
        if not task:
            return None
        pred_text = self._strip_hidden_reasoning(final_output)
        ref_text = reference_answer or ""
        pred_json = self._safe_json(pred_text) if isinstance(pred_text, str) else None
        ref_json = self._safe_json(ref_text) if isinstance(ref_text, str) else None

        def _as_int_list(value: object) -> List[int]:
            if not isinstance(value, list):
                return []
            out: List[int] = []
            for item in value:
                try:
                    out.append(int(item))
                except Exception:
                    continue
            return out

        def _json_field(obj: Optional[dict], key: str) -> Optional[object]:
            if isinstance(obj, dict) and key in obj:
                return obj.get(key)
            return None

        if task in {"connectivity", "cycle"}:
            pred_val = _json_field(pred_json, "answer")
            ref_val = _json_field(ref_json, "answer")
            pred = self._normalize_yes_no_output(str(pred_val)) if pred_val is not None else self._normalize_yes_no_output(pred_text)
            ref = self._normalize_yes_no_output(str(ref_val)) if ref_val is not None else self._normalize_yes_no_output(ref_text)
            correct = pred == ref
            return (1.0 if correct else 0.0), (1.0 if correct else 0.0), (0.0 if correct else 0.6), {
                "used_dataset_specific": True,
                "task": task,
                "pred": pred,
                "ref": ref,
            }
        if task == "flow":
            pred_val = _json_field(pred_json, "max_flow")
            ref_val = _json_field(ref_json, "max_flow")
            pred_num = float(pred_val) if isinstance(pred_val, (int, float)) else self._extract_last_number(pred_text)
            ref_num = float(ref_val) if isinstance(ref_val, (int, float)) else self._extract_last_number(ref_text)
            correct = pred_num is not None and ref_num is not None and abs(pred_num - ref_num) <= 1e-6
            return (1.0 if correct else 0.0), (1.0 if correct else 0.0), (0.0 if correct else 0.6), {
                "used_dataset_specific": True,
                "task": task,
                "pred_num": pred_num,
                "ref_num": ref_num,
            }
        if task == "topology":
            order_val = _json_field(pred_json, "order")
            order = _as_int_list(order_val) if order_val is not None else self._extract_sequence_numbers(pred_text)
            constraints = [(int(a), int(b)) for a, b in re.findall(r"node (\d+) should be visited before node (\d+)", question_text)]
            if not order or not constraints:
                return 0.0, 0.0, 0.7, {"used_dataset_specific": True, "task": task, "format_ok": False}
            position = {node: idx for idx, node in enumerate(order)}
            satisfied = sum(1 for a, b in constraints if a in position and b in position and position[a] < position[b])
            valid = satisfied == len(constraints) and len(position) == len(order)
            score = satisfied / max(1, len(constraints))
            return score, (1.0 if valid else 0.0), (0.0 if valid else 0.4 * (1.0 - score)), {
                "used_dataset_specific": True,
                "task": task,
                "constraints_satisfied": satisfied,
                "constraints_total": len(constraints),
            }
        if task == "hamilton":
            path_val = _json_field(pred_json, "path")
            path = _as_int_list(path_val) if path_val is not None else self._extract_sequence_numbers(pred_text)
            edges = self._parse_nlgraph_edges(question_text)
            max_node = max((max(src, dst) for src, dst in edges.keys()), default=-1)
            expected_nodes = max_node + 1
            if not path:
                return 0.0, 0.0, 0.8, {"used_dataset_specific": True, "task": task, "format_ok": False}
            unique_ratio = len(set(path)) / max(1, expected_nodes)
            valid_edges = sum(1 for src, dst in zip(path, path[1:]) if (src, dst) in edges)
            edge_ratio = valid_edges / max(1, len(path) - 1)
            valid = len(path) == expected_nodes and len(set(path)) == expected_nodes and edge_ratio >= 0.999
            score = min(1.0, 0.5 * unique_ratio + 0.5 * edge_ratio)
            return score, (1.0 if valid else 0.0), (0.0 if valid else 0.5 * (1.0 - score)), {
                "used_dataset_specific": True,
                "task": task,
                "path_length": len(path),
                "expected_nodes": expected_nodes,
                "edge_ratio": edge_ratio,
            }
        if task == "shortest_path":
            path_val = _json_field(pred_json, "path")
            path = _as_int_list(path_val) if path_val is not None else self._extract_sequence_numbers(pred_text)
            edges = self._parse_nlgraph_edges(question_text)
            ref_weight_val = _json_field(ref_json, "total_weight")
            pred_weight_val = _json_field(pred_json, "total_weight")
            ref_weight = float(ref_weight_val) if isinstance(ref_weight_val, (int, float)) else self._extract_last_number(ref_text)
            pred_weight = float(pred_weight_val) if isinstance(pred_weight_val, (int, float)) else self._extract_last_number(pred_text)
            computed_weight = self._path_weight(path, edges) if path else None
            target_match = pred_weight is not None and ref_weight is not None and abs(pred_weight - ref_weight) <= 1e-6
            inferred_match = computed_weight is not None and ref_weight is not None and abs(computed_weight - ref_weight) <= 1e-6
            valid = target_match or inferred_match
            score = 1.0 if valid else (0.4 if computed_weight is not None else 0.0)
            return score, (1.0 if valid else 0.0), (0.0 if valid else 0.5), {
                "used_dataset_specific": True,
                "task": task,
                "pred_weight": pred_weight,
                "computed_weight": computed_weight,
                "ref_weight": ref_weight,
            }
        if task == "matching":
            ref_count_val = _json_field(ref_json, "count")
            pred_count_val = _json_field(pred_json, "count")
            ref_count = int(ref_count_val) if isinstance(ref_count_val, (int, float)) else None
            pred_count = int(pred_count_val) if isinstance(pred_count_val, (int, float)) else None
            if ref_count is None:
                count_match = re.search(r"(\d+)\s+applicants can find", ref_text, flags=re.IGNORECASE)
                ref_count = int(count_match.group(1)) if count_match else None
            if pred_count is None:
                pred_count_match = re.search(r"(\d+)\s+applicants can find", pred_text, flags=re.IGNORECASE)
                pred_count = int(pred_count_match.group(1)) if pred_count_match else None
            correct = pred_count is not None and ref_count is not None and pred_count == ref_count
            return (1.0 if correct else 0.0), (1.0 if correct else 0.0), (0.0 if correct else 0.6), {
                "used_dataset_specific": True,
                "task": task,
                "pred_count": pred_count,
                "ref_count": ref_count,
            }
        if task == "GNN":
            pred_emb_val = _json_field(pred_json, "node_embeddings")
            ref_emb_val = _json_field(ref_json, "node_embeddings")
            if isinstance(ref_emb_val, dict):
                ref_emb = {str(k): _as_int_list(v) for k, v in ref_emb_val.items()}
                pred_emb = {str(k): _as_int_list(v) for k, v in pred_emb_val.items()} if isinstance(pred_emb_val, dict) else {}
                total = len(ref_emb)
                matches = sum(1 for node, vec in ref_emb.items() if pred_emb.get(node) == vec)
                accuracy = matches / max(1, total)
                return accuracy, (1.0 if accuracy >= 0.999 else 0.0), (0.0 if accuracy >= 0.999 else 0.5 * (1.0 - accuracy)), {
                    "used_dataset_specific": True,
                    "task": task,
                    "format_ok": isinstance(pred_emb_val, dict),
                    "matches": matches,
                    "total": total,
                }
            pred_pairs = dict(re.findall(r"node\s+(\d+)\s*:\s*\[([^\]]+)\]", pred_text, flags=re.IGNORECASE))
            ref_pairs = dict(re.findall(r"node\s+(\d+)\s*:\s*\[([^\]]+)\]", ref_text, flags=re.IGNORECASE))
            if not ref_pairs:
                return None
            matches = sum(1 for node, vec in ref_pairs.items() if pred_pairs.get(node, "").replace(" ", "") == vec.replace(" ", ""))
            total = len(ref_pairs)
            accuracy = matches / max(1, total)
            return accuracy, (1.0 if accuracy >= 0.999 else 0.0), (0.0 if accuracy >= 0.999 else 0.5 * (1.0 - accuracy)), {
                "used_dataset_specific": True,
                "task": task,
                "format_ok": True,
                "matches": matches,
                "total": total,
            }
        return None

    def _score_with_reference(
        self,
        question_text: str,
        final_output: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
    ) -> Optional[Tuple[float, float, float, Dict[str, object]]]:
        if not reference_answer and not metadata:
            return None
        dataset_name = self._dataset_name(metadata)
        if dataset_name == "knowledge_crosswords":
            return self._score_knowledge_crosswords(final_output, reference_answer, metadata)
        if dataset_name == "nlgraph":
            return self._score_nlgraph(question_text, final_output, reference_answer, metadata)
        if dataset_name in {"humaneval", "mbpp"}:
            return self._score_code_generation(final_output, metadata)
        if dataset_name == "math" and reference_answer:
            pred_expr = self._normalize_math_expression(final_output)
            ref_expr = self._normalize_math_expression(reference_answer)
            correct = bool(pred_expr) and pred_expr == ref_expr
            return (1.0 if correct else 0.0), (1.0 if correct else 0.0), (0.0 if correct else 0.6), {
                "used_dataset_specific": True,
                "pred_expr": pred_expr,
                "ref_expr": ref_expr,
            }
        task_type = self._task_type(question_text, reference_answer=reference_answer, metadata=metadata)
        debug: Dict[str, object] = {"task_type": task_type, "reference_answer": reference_answer}
        if task_type == "mcq":
            pred_option, pred_conf = self._parse_mcq_answer(final_output)
            gold_option = self._gold_mcq_option(reference_answer, metadata)
            max_option = self._extract_mcq_option_range(question_text)
            if max_option <= 0 and metadata and isinstance(metadata.get("options"), list):
                max_option = len(metadata["options"])
            format_ok = pred_option is not None and 1 <= pred_option <= max_option
            wants_conf = "CONFIDENCE -" in question_text.upper()
            conf_ok = (pred_conf is not None and 1 <= pred_conf <= 5) if wants_conf else True
            debug.update(
                {
                    "pred_option": pred_option,
                    "pred_confidence": pred_conf,
                    "gold_option": gold_option,
                    "format_ok": format_ok,
                    "confidence_ok": conf_ok,
                }
            )
            if gold_option is None:
                return None
            correct = pred_option == gold_option
            abstain_option = max_option if "I Don't Know/ None of the above" in question_text else None
            chose_abstain = pred_option is not None and abstain_option is not None and pred_option == abstain_option
            task_score = 0.0
            if format_ok:
                task_score += 0.15
            if conf_ok:
                task_score += 0.10 if wants_conf else 0.0
            if correct:
                task_score += 0.75
                safety_penalty = 0.0
                success = 1.0
            else:
                safety_penalty = 0.45 if chose_abstain else 0.70
                success = 0.0
            task_score = max(0.0, min(1.0, task_score))
            debug["used_dataset_specific"] = True
            debug["correct"] = correct
            return task_score, success, safety_penalty, debug

        if task_type == "numeric":
            pred_num = self._extract_prediction_target(final_output)
            ref_num = self._extract_gsm8k_target(reference_answer or "")
            correct = pred_num is not None and ref_num is not None and abs(pred_num - ref_num) <= 1e-6
            debug.update(
                {
                    "pred_num": pred_num,
                    "ref_num": ref_num,
                    "used_dataset_specific": True,
                    "correct": correct,
                }
            )
            task_score = 1.0 if correct else 0.0
            success = 1.0 if correct else 0.0
            safety_penalty = 0.0 if correct else 0.60
            return task_score, success, safety_penalty, debug

        norm_pred = self._normalize_text(final_output)
        norm_ref = self._normalize_text(reference_answer)
        pred_num = self._extract_prediction_target(final_output)
        ref_num = self._extract_last_number(reference_answer)
        exact = norm_pred == norm_ref
        numeric_match = pred_num is not None and ref_num is not None and abs(pred_num - ref_num) <= 1e-6
        correct = exact or numeric_match
        debug.update(
            {
                "normalized_prediction": norm_pred,
                "normalized_reference": norm_ref,
                "pred_num": pred_num,
                "ref_num": ref_num,
                "exact_match": exact,
                "numeric_match": numeric_match,
                "used_dataset_specific": True,
            }
        )
        task_score = 1.0 if correct else 0.0
        success = 1.0 if correct else 0.0
        safety_penalty = 0.0 if correct else 0.60
        return task_score, success, safety_penalty, debug

    def _mcq_heuristic_score(self, question_text: str, final_output: str) -> Tuple[float, float, float]:
        normalized = self._normalize_mcq_output(question_text, final_output)
        option, confidence = self._parse_mcq_answer(normalized)
        max_option = self._extract_mcq_option_range(question_text)
        task_score = 0.0
        safety_penalty = 0.0
        if option is not None and 1 <= option <= max_option:
            task_score += 0.45
        else:
            safety_penalty += 0.35
        if confidence is not None and 1 <= confidence <= 5:
            task_score += 0.15
        else:
            safety_penalty += 0.15
        if normalized.count("\n") == 1 and normalized.upper().startswith("OPTION"):
            task_score += 0.15
        else:
            safety_penalty += 0.10
        abstain_option = max_option if "I Don't Know/ None of the above" in question_text else None
        if option is not None and abstain_option is not None and option == abstain_option:
            task_score -= 0.12
        task_score = max(0.0, min(1.0, task_score))
        safety_penalty = max(0.0, min(1.0, safety_penalty))
        success = 1.0 if task_score >= self.config.success_threshold else 0.0
        return task_score, success, safety_penalty

    def _output_contract(
        self,
        question_text: str,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        task_type = self._task_type(question_text, reference_answer=reference_answer, metadata=metadata)
        rules = [
            "不要输出<think>标签、分析过程、解释、'通过'、'已验证'之类元话语。",
            "严格遵守题目要求的最终答案格式。",
        ]
        answer_format = self._answer_format(metadata)
        if task_type == "mcq":
            max_option = self._extract_mcq_option_range(question_text)
            if max_option <= 0 and metadata and isinstance(metadata.get("options"), list):
                max_option = len(metadata["options"])
            if "CONFIDENCE -" in question_text.upper():
                rules.append(f"你必须且只能输出两行：`OPTION - <1到{max_option}的整数>` 和 `CONFIDENCE - <1到5的整数>`。")
            else:
                rules.append(f"你必须且只能输出一行：`OPTION - <1到{max_option}的整数>`。")
            rules.append("如果你心里想到的是字母选项 A/B/C/D，也必须先映射成 1/2/3/4 再输出。")
        elif task_type == "numeric":
            rules.append("这是数值题。最后只输出最终数值答案，必要时可保留极简单位，但不要输出推导过程。")
        elif answer_format == "yes_no":
            rules.append("这是判断题。最后只能输出 yes 或 no，且只能保留这一项。")
        elif answer_format == "json_list":
            rules.append("最后只能输出一个 JSON 数组，不要附带任何解释、项目符号或 Markdown 标记。")
            rules.append("数组长度和元素顺序必须与题目要求严格一致。")
        elif answer_format == "graph_json":
            rules.append("最后只能输出一个 JSON 对象，字段名和结构必须严格符合题目要求。")
            rules.append("不要输出自然语言解释，不要丢字段，不要改变键名。")
        elif answer_format == "python_code":
            rules.append("最后只能输出可直接执行的 Python 代码，不要加 Markdown 代码块或解释。")
            entry_point = str((metadata or {}).get("entry_point") or "").strip()
            if entry_point:
                rules.append(f"必须定义题目要求的函数 `{entry_point}`，且函数名不得改动。")
            rules.append("尽量避免额外的顶层打印、示例调用、解释注释或与题意无关的辅助代码。")
        elif answer_format == "math_expression":
            rules.append("最后只输出最终数学表达式，不要输出证明、推导或额外文字。")
            rules.append("如果有等价形式，优先输出最简洁、最标准的形式。")
        elif answer_format == "short_span":
            rules.append("最后只输出一个尽量短的答案短语，不要输出完整解释。")
        elif answer_format == "question_defined":
            rules.append("题目已经给出输出格式，必须严格照做，不要添加任何前后缀。")
        else:
            rules.append("最后只输出最终答案本身，不要附带解释。")
        return "\n".join(rules)

    def _sanitize_final_output(
        self,
        question_text: str,
        final_output: str,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        task_type = self._task_type(question_text, reference_answer=reference_answer, metadata=metadata)
        cleaned = self._strip_hidden_reasoning(final_output)
        if task_type == "mcq":
            return self._normalize_mcq_output(question_text, cleaned, metadata=metadata)
        if task_type == "numeric":
            value = self._extract_prediction_target(cleaned)
            if value is not None:
                if float(value).is_integer():
                    return str(int(value))
                return str(value)
        answer_format = self._answer_format(metadata)
        if answer_format == "yes_no":
            return self._normalize_yes_no_output(cleaned)
        if answer_format == "json_list":
            parsed = self._extract_json_list(cleaned)
            if parsed is not None:
                return parsed
        if answer_format == "python_code":
            return self._extract_python_code(cleaned)
        if answer_format == "math_expression":
            return self._extract_boxed_expression(cleaned)
        if answer_format == "short_span" and cleaned:
            return cleaned.splitlines()[0].strip()
        return cleaned

    def _evaluate_final_output(
        self,
        question_text: str,
        final_output: str,
        *,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
        trace: Optional[List[Dict[str, str]]] = None,
        latency: float = 0.0,
        token_cost: float = 0.0,
        custom_metrics: Optional[Dict[str, float]] = None,
    ) -> TaskEvaluation:
        rendered_question = render_question_text(question_text, metadata=metadata)
        final_output = self._sanitize_final_output(
            rendered_question,
            final_output,
            reference_answer=reference_answer,
            metadata=metadata,
        )
        debug_info: Dict[str, object] = {}
        dataset_specific = self._score_with_reference(rendered_question, final_output, reference_answer, metadata)
        parsed = {}
        judge_text = ""
        judge_prompt = ""
        judge_input = ""
        if dataset_specific is not None:
            task_score, success, safety_penalty, ds_debug = dataset_specific
            debug_info["dataset_specific"] = ds_debug
        else:
            judge_prompt, judge_input = self._judge_prompt(
                question_text,
                final_output,
                reference_answer=reference_answer,
                metadata=metadata,
            )
            judge_messages = [
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": judge_input},
            ]
            judge_text = self._cached_chat(
                judge_messages,
                runtime=TierRuntimeConfig(
                    max_tokens=self.config.judge.max_tokens,
                    judge_max_tokens=self.config.judge.max_tokens,
                    repeats=1,
                    temperature=self.config.judge.temperature,
                ),
                model=self.config.judge.model or self.config.chat.model,
                client="judge",
            )
            parsed = self._safe_json(judge_text) or {}
            task_score = float(parsed.get("task_score", 0.0))
            success = float(parsed.get("success", 1.0 if task_score >= self.config.success_threshold else 0.0))
            safety_penalty = float(parsed.get("safety_penalty", 0.0))
        if self._task_type(rendered_question, reference_answer=reference_answer, metadata=metadata) == "mcq":
            heuristic_task, heuristic_success, heuristic_safety = self._mcq_heuristic_score(rendered_question, final_output)
            debug_info["mcq_heuristic"] = {
                "task_score": heuristic_task,
                "success": heuristic_success,
                "safety_penalty": heuristic_safety,
            }
            if dataset_specific is not None:
                pass
            elif "task_score" not in parsed:
                task_score = heuristic_task
                success = heuristic_success
                safety_penalty = heuristic_safety
            else:
                task_score = max(0.0, min(1.0, 0.7 * task_score + 0.3 * heuristic_task))
                safety_penalty = max(0.0, min(1.0, 0.7 * safety_penalty + 0.3 * heuristic_safety))
                success = float(parsed.get("success", 1.0 if task_score >= self.config.success_threshold else 0.0))
        if self.config.debug_judge:
            debug_info["judge"] = {
                "prompt": judge_prompt,
                "input": judge_input,
                "raw_text": judge_text,
                "parsed": parsed,
                "final_task_score": task_score,
                "final_success": success,
                "final_safety_penalty": safety_penalty,
            }
        return TaskEvaluation(
            task_score=task_score,
            success=success,
            latency=latency,
            token_cost=token_cost,
            safety_penalty=safety_penalty,
            raw_output=final_output,
            trace=list(trace or []),
            custom_metrics=dict(custom_metrics or {}),
            debug_info=debug_info,
        )

    def _run_architecture_once(
        self,
        compiled: CompiledArchitecture,
        question_text: str,
        runtime: TierRuntimeConfig,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
    ) -> TaskEvaluation:
        start = time.perf_counter()
        ctx = ExecutionContext(role_outputs={}, trace=[])
        state = compiled.state
        sink_roles = set(self._sink_roles(compiled))
        answer_contract = self._output_contract(
            question_text,
            reference_answer=reference_answer,
            metadata=metadata,
        )

        for role in compiled.execution_roles:
            agent_id = state.role_to_agent[role]
            agent = self._by_id[agent_id]
            slots = state.role_to_prompt.get(role, PromptSlots())
            upstream_roles = self._upstream_roles(compiled, role)
            upstream_outputs = {r: ctx.role_outputs.get(r, "") for r in upstream_roles if ctx.role_outputs.get(r, "")}
            messages = [
                {
                    "role": "system",
                    "content": build_system_prompt(agent, slots, extra_role_hint=role),
                },
                {
                    "role": "user",
                    "content": build_user_prompt(
                        question_text=question_text,
                        upstream_outputs=upstream_outputs,
                        task_instruction=self._role_task_instruction(
                            compiled,
                            role,
                            dataset_profile=dataset_profile,
                            metadata=metadata,
                        ),
                        metadata=metadata,
                        task_context=self._task_context(
                            question_text,
                            dataset_profile=dataset_profile,
                            metadata=metadata,
                            role=role,
                        ),
                        answer_contract=answer_contract if role in sink_roles else "",
                    ),
                },
            ]
            content = self._cached_chat(messages, runtime=runtime)
            ctx.role_outputs[role] = content
            ctx.trace.append({"role": role, "agent_id": agent_id, "content": content})

        final_output = "\n\n".join(
            ctx.role_outputs.get(role, "") for role in self._sink_roles(compiled) if ctx.role_outputs.get(role, "")
        )
        latency = time.perf_counter() - start
        token_cost = sum(len(item["content"].split()) for item in ctx.trace) * self.config.token_cost_per_word
        return self._evaluate_final_output(
            question_text,
            final_output,
            reference_answer=reference_answer,
            metadata=metadata,
            trace=ctx.trace,
            latency=latency,
            token_cost=token_cost,
            custom_metrics={"num_roles": float(len(compiled.execution_roles))},
        )

    @staticmethod
    def _upstream_roles(compiled: CompiledArchitecture, role: str) -> List[str]:
        role_by_agent = {aid: r for r, aid in compiled.state.role_to_agent.items()}
        target_agent = compiled.state.role_to_agent.get(role, "")
        upstream_agents = [src for src, dst in compiled.edges if dst == target_agent]
        return [role_by_agent[src] for src in upstream_agents if src in role_by_agent]

    @staticmethod
    def _sink_roles(compiled: CompiledArchitecture) -> List[str]:
        role_by_agent = {aid: role for role, aid in compiled.state.role_to_agent.items()}
        return [role_by_agent[aid] for aid in compiled.sinks if aid in role_by_agent]

    def evaluate(
        self,
        compiled: CompiledArchitecture,
        question_text: str,
        *,
        tier: str,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
    ) -> EvalSummary:
        runtime = self._resolve_runtime(tier, dataset_profile)
        cache_key = (
            f"{tier}|{dataset_profile.name}|{compiled.signature()}|{question_text}|{reference_answer}|"
            f"{json.dumps(metadata or {}, sort_keys=True, ensure_ascii=False)}"
        )
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt_penalty = (
            sum(slot.complexity() for slot in compiled.state.role_to_prompt.values()) / max(
                1, len(compiled.state.role_to_prompt)
            )
        ) * dataset_profile.reward_prompt_penalty_scale
        size_penalty = (
            max(0.0, len(compiled.state.active_agents()) - 4) * dataset_profile.reward_size_penalty_scale
        )

        runs = [
            self._run_architecture_once(
                compiled,
                question_text,
                runtime,
                reference_answer=reference_answer,
                metadata=metadata,
                dataset_profile=dataset_profile,
            )
            for _ in range(runtime.repeats)
        ]
        rewards = [
            reward_from_evaluation(
                ev,
                size_penalty=size_penalty,
                prompt_penalty=prompt_penalty,
                weights=reward_weights_for_profile(dataset_profile, metadata),
            )
            for ev in runs
        ]
        mean_reward = sum(rewards) / max(1, len(rewards))
        variance = sum((r - mean_reward) ** 2 for r in rewards) / max(1, len(rewards))
        summary = EvalSummary(
            tier=tier,
            mean_reward=mean_reward,
            reward_std=sqrt(variance),
            mean_task_score=sum(ev.task_score for ev in runs) / len(runs),
            mean_success=sum(ev.success for ev in runs) / len(runs),
            mean_latency=sum(ev.latency for ev in runs) / len(runs),
            mean_token_cost=sum(ev.token_cost for ev in runs) / len(runs),
            mean_safety_penalty=sum(ev.safety_penalty for ev in runs) / len(runs),
            evaluations=runs,
        )
        self._eval_cache.put(cache_key, summary)
        return summary

    def evaluate_output(
        self,
        question_text: str,
        final_output: str,
        *,
        tier: str,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
        custom_metrics: Optional[Dict[str, float]] = None,
        trace: Optional[List[Dict[str, str]]] = None,
        size_penalty: float = 0.0,
        prompt_penalty: float = 0.0,
        latency: float = 0.0,
        token_cost: float = 0.0,
    ) -> EvalSummary:
        cache_key = (
            f"output|{tier}|{dataset_profile.name}|{question_text}|{reference_answer}|{final_output}|"
            f"{json.dumps(metadata or {}, sort_keys=True, ensure_ascii=False)}"
        )
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            return cached
        evaluation = self._evaluate_final_output(
            question_text,
            final_output,
            reference_answer=reference_answer,
            metadata=metadata,
            trace=trace,
            custom_metrics=custom_metrics,
            latency=latency,
            token_cost=token_cost,
        )
        reward = reward_from_evaluation(
            evaluation,
            size_penalty=size_penalty,
            prompt_penalty=prompt_penalty,
            weights=reward_weights_for_profile(dataset_profile, metadata),
        )
        summary = EvalSummary(
            tier=tier,
            mean_reward=reward,
            reward_std=0.0,
            mean_task_score=evaluation.task_score,
            mean_success=evaluation.success,
            mean_latency=evaluation.latency,
            mean_token_cost=evaluation.token_cost,
            mean_safety_penalty=evaluation.safety_penalty,
            evaluations=[evaluation],
        )
        self._eval_cache.put(cache_key, summary)
        return summary

    @staticmethod
    def feedback_signal(
        summary: EvalSummary,
        *,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
        metadata: Optional[dict] = None,
    ) -> float:
        if not summary.evaluations:
            return 0.0
        if dataset_profile.task_type == "code_generation":
            signals: List[float] = []
            for evaluation in summary.evaluations:
                ds = evaluation.debug_info.get("dataset_specific", {})
                if not isinstance(ds, dict):
                    ds = {}
                syntax_ok = 1.0 if ds.get("syntax_ok") else 0.0
                entry_defined = 1.0 if ds.get("entry_defined") else 0.0
                passed = ds.get("passed")
                total = ds.get("total")
                pass_ratio = 0.0
                if isinstance(passed, int) and isinstance(total, int) and total > 0:
                    pass_ratio = passed / total
                elif evaluation.success > 0.0:
                    pass_ratio = 1.0
                signal = 0.55 * pass_ratio + 0.20 * syntax_ok + 0.15 * entry_defined + 0.10 * evaluation.task_score
                signals.append(max(0.0, min(1.0, signal)))
            return sum(signals) / max(1, len(signals))
        if dataset_profile.task_type in {"graph_reasoning", "structured_list"}:
            return max(0.0, min(1.0, summary.mean_task_score))
        if dataset_profile.task_type == "mcq":
            return max(0.0, min(1.0, summary.mean_success * 0.7 + summary.mean_task_score * 0.3))
        del metadata
        return max(0.0, min(1.0, summary.mean_task_score))
