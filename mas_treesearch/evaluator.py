from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Optional, Tuple

from .agents import AgentPool
from .cache import DictCache
from .clients import OpenAICompatClient
from .config import TierRuntimeConfig, TieredEvalConfig
from .prompting import build_system_prompt, build_user_prompt, render_question_text
from .reward import reward_from_evaluation
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
    ) -> str:
        key = json.dumps(
            {
                "messages": messages,
                "temperature": runtime.temperature,
                "max_tokens": runtime.max_tokens,
                "model": model or self.config.chat.model,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        if cacheable and self.config.enable_chat_cache:
            cached = self._chat_cache.get(key)
            if cached is not None:
                return cached
        value = self._chat.chat(
            messages,
            temperature=runtime.temperature,
            max_tokens=runtime.max_tokens,
            model=model,
        )
        if cacheable and self.config.enable_chat_cache:
            self._chat_cache.put(key, value)
        return value

    def _role_task_instruction(self, compiled: CompiledArchitecture, role: str) -> str:
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
        return f"当前架构模板：{template}。{mapping.get(role, '完成你在当前架构中的子任务。')}"

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

    def _score_with_reference(
        self,
        question_text: str,
        final_output: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
    ) -> Optional[Tuple[float, float, float, Dict[str, object]]]:
        if not reference_answer and not metadata:
            return None
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
        if answer_format == "short_span" and cleaned:
            return cleaned.splitlines()[0].strip()
        return cleaned

    def _run_architecture_once(
        self,
        compiled: CompiledArchitecture,
        question_text: str,
        runtime: TierRuntimeConfig,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> TaskEvaluation:
        start = time.perf_counter()
        ctx = ExecutionContext(role_outputs={}, trace=[])
        state = compiled.state
        sink_roles = set(self._sink_roles(compiled))
        rendered_question = render_question_text(question_text, metadata=metadata)
        answer_contract = self._output_contract(
            rendered_question,
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
                        task_instruction=self._role_task_instruction(compiled, role),
                        metadata=metadata,
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
        final_output = self._sanitize_final_output(
            rendered_question,
            final_output,
            reference_answer=reference_answer,
            metadata=metadata,
        )
        judge_runtime = TierRuntimeConfig(
            max_tokens=runtime.max_tokens,
            judge_max_tokens=runtime.judge_max_tokens,
            repeats=runtime.repeats,
            temperature=0.0,
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
            judge_model = self.config.judge.model or self.config.chat.model
            judge_text = self._cached_chat(
                judge_messages,
                runtime=TierRuntimeConfig(
                    max_tokens=runtime.judge_max_tokens,
                    judge_max_tokens=runtime.judge_max_tokens,
                    repeats=1,
                    temperature=self.config.judge.temperature,
                ),
                model=judge_model,
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
        latency = time.perf_counter() - start
        token_cost = sum(len(item["content"].split()) for item in ctx.trace) * self.config.token_cost_per_word
        return TaskEvaluation(
            task_score=task_score,
            success=success,
            latency=latency,
            token_cost=token_cost,
            safety_penalty=safety_penalty,
            raw_output=final_output,
            trace=ctx.trace,
            custom_metrics={"num_roles": float(len(compiled.execution_roles))},
            debug_info=debug_info,
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
    ) -> EvalSummary:
        runtime = self.config.tier1 if tier == "tier1" else self.config.tier2
        cache_key = f"{tier}|{compiled.signature()}|{question_text}|{reference_answer}|{json.dumps(metadata or {}, sort_keys=True, ensure_ascii=False)}"
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            return cached

        prompt_penalty = sum(slot.complexity() for slot in compiled.state.role_to_prompt.values()) / max(
            1, len(compiled.state.role_to_prompt)
        )
        size_penalty = max(0.0, len(compiled.state.active_agents()) - 4)

        runs = [
            self._run_architecture_once(
                compiled,
                question_text,
                runtime,
                reference_answer=reference_answer,
                metadata=metadata,
            )
            for _ in range(runtime.repeats)
        ]
        rewards = [
            reward_from_evaluation(
                ev,
                size_penalty=size_penalty,
                prompt_penalty=prompt_penalty,
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
