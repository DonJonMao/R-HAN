from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from mas_treesearch import SearchConfig, TieredEvalConfig, TreeSearchMASPipeline, UnionRuntimeConfig
from mas_treesearch.agents import AgentPool
from mas_treesearch.profiles import resolve_dataset_profile
from mas_treesearch.reward import risk_adjusted_score
from mas_treesearch.types import EvalSummary, SearchResult

from .config import Stage2RuntimeConfig
from .config_v2 import Stage2V2Config
from .runtime import Stage2Runtime
from .runtime_v2 import Stage2RuntimeV2
from .structure_io import PreparedStage1Artifact, prepared_stage1_from_search_result
from .types import Stage2RunResult


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _summary_target(summary: EvalSummary) -> float:
    return _clamp01(0.55 * float(summary.mean_success) + 0.45 * float(summary.mean_task_score))


def _summary_quality_key(summary: EvalSummary) -> tuple[float, float, float]:
    return (
        float(summary.mean_success),
        float(summary.mean_task_score),
        -float(summary.mean_safety_penalty),
    )


def _selection_metadata(
    *,
    stage1_summary: EvalSummary,
    stage2_summary: EvalSummary,
    risk_std_penalty: float,
) -> dict[str, float | str]:
    stage1_reward = risk_adjusted_score(stage1_summary, risk_std_penalty)
    stage2_reward = risk_adjusted_score(stage2_summary, risk_std_penalty)
    stage1_quality = _summary_quality_key(stage1_summary)
    stage2_quality = _summary_quality_key(stage2_summary)
    if stage2_quality > stage1_quality:
        decision = "use_stage2"
        reason = "quality_better"
    elif stage2_quality < stage1_quality:
        decision = "fallback_stage1"
        reason = "quality_worse"
    else:
        decision = "use_stage2"
        reason = "quality_tie_prefer_stage2"
    return {
        "selection_decision": decision,
        "selection_reason": reason,
        "stage1_reward": float(stage1_reward),
        "stage2_reward": float(stage2_reward),
        "stage1_task_score": float(stage1_summary.mean_task_score),
        "stage2_task_score": float(stage2_summary.mean_task_score),
        "stage1_success": float(stage1_summary.mean_success),
        "stage2_success": float(stage2_summary.mean_success),
        "stage1_safety_penalty": float(stage1_summary.mean_safety_penalty),
        "stage2_safety_penalty": float(stage2_summary.mean_safety_penalty),
        "stage1_latency": float(stage1_summary.mean_latency),
        "stage2_latency": float(stage2_summary.mean_latency),
        "stage1_token_cost": float(stage1_summary.mean_token_cost),
        "stage2_token_cost": float(stage2_summary.mean_token_cost),
    }


@dataclass
class Stage2PipelineResult:
    stage1_result: Optional[SearchResult]
    stage1_artifact: PreparedStage1Artifact
    stage2_result: Stage2RunResult
    final_summary: EvalSummary
    final_signature: str
    final_output: str


@dataclass
class Stage2MASPipeline:
    search_config: SearchConfig = field(default_factory=SearchConfig)
    runtime_config: TieredEvalConfig = field(default_factory=TieredEvalConfig)
    union_config: UnionRuntimeConfig = field(default_factory=UnionRuntimeConfig)
    stage2_config: Stage2RuntimeConfig | Stage2V2Config = field(default_factory=Stage2RuntimeConfig)
    agent_pool: Optional[AgentPool] = None

    def __post_init__(self) -> None:
        self._stage1 = TreeSearchMASPipeline(
            search_config=self.search_config,
            runtime_config=self.runtime_config,
            union_config=self.union_config,
            pipeline_mode="structure_only",
            agent_pool=self.agent_pool,
        )
        self.agent_pool = self._stage1.agent_pool
        self.runtime_config = self._stage1.runtime_config
        if isinstance(self.stage2_config, Stage2V2Config):
            self._stage2 = Stage2RuntimeV2(
                self.stage2_config,
                self._stage1._evaluator,
                self._stage1.agent_pool,
                self._stage1._embedder,
            )
        else:
            self._stage2 = Stage2Runtime(
                self.stage2_config,
                self._stage1._evaluator,
                self._stage1.agent_pool,
                self._stage1._embedder,
            )

    @staticmethod
    def _resolve_dataset_name(dataset_name: Optional[str], metadata: Optional[dict]) -> Optional[str]:
        if dataset_name:
            return dataset_name
        if isinstance(metadata, dict):
            value = metadata.get("mas_dataset_name")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _run_stage1_search(
        self,
        question_text: str,
        *,
        reference_answer: Optional[str],
        metadata: Optional[dict],
        dataset_name: Optional[str],
    ) -> tuple[SearchResult, PreparedStage1Artifact, Optional[str]]:
        resolved_dataset = self._resolve_dataset_name(dataset_name, metadata)
        stage1_result = self._stage1.search(
            question_text,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_name=resolved_dataset,
            learn=False,
            pipeline_mode="structure_only",
        )
        artifact = prepared_stage1_from_search_result(stage1_result, dataset_name=resolved_dataset)
        return stage1_result, artifact, resolved_dataset

    def prepare_stage1_structure(
        self,
        question_text: str,
        *,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
        dataset_name: Optional[str] = None,
        learn: bool = False,
    ) -> PreparedStage1Artifact:
        del learn
        _, artifact, _ = self._run_stage1_search(
            question_text,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_name=dataset_name,
        )
        return artifact

    def search_prepared(
        self,
        question_text: str,
        *,
        prepared_structure: PreparedStage1Artifact,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
        dataset_name: Optional[str] = None,
        replay_dir: Optional[str] = None,
        stage1_result: Optional[SearchResult] = None,
        learn: bool = False,
        allow_stage1_fallback: bool = True,
    ) -> Stage2PipelineResult:
        if prepared_structure.question_text and prepared_structure.question_text != question_text:
            raise ValueError("Prepared stage-1 artifact does not match the requested question text.")
        resolved_dataset = dataset_name or prepared_structure.dataset_name or self._resolve_dataset_name(dataset_name, metadata)
        profile = resolve_dataset_profile(resolved_dataset)
        run_kwargs = dict(
            question_text=question_text,
            metadata=metadata,
            reference_answer=reference_answer,
            dataset_profile=profile,
            replay_dir=replay_dir,
        )
        if isinstance(self._stage2, Stage2RuntimeV2):
            run_kwargs["learn"] = learn
        stage2_result = self._stage2.run(prepared_structure.union_graph, **run_kwargs)
        stage2_summary = self._stage1._evaluator.evaluate_output(
            question_text,
            stage2_result.final_answer,
            tier="tier2",
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=profile,
            custom_metrics={
                "stage2_turns": float(stage2_result.metadata.get("turn_count", 0)),
                "stage2_task_nodes": float(stage2_result.metadata.get("task_node_count", 0)),
                "stage2_source_graphs": float(stage2_result.metadata.get("source_graph_count", 0)),
            },
            latency=float(stage2_result.metadata.get("elapsed_s", 0.0)),
            token_cost=float(stage2_result.metadata.get("approx_token_cost", 0.0)),
        )
        final_summary = stage2_summary
        final_signature = stage2_result.signature
        final_output = stage2_result.final_answer
        learning_target = _summary_target(stage2_summary)
        stage2_result.metadata["stage2_reward"] = float(risk_adjusted_score(stage2_summary, self.search_config.risk_std_penalty))
        stage2_result.metadata["stage2_task_score"] = float(stage2_summary.mean_task_score)
        stage2_result.metadata["stage2_success"] = float(stage2_summary.mean_success)
        stage2_result.metadata["stage2_safety_penalty"] = float(stage2_summary.mean_safety_penalty)
        stage2_result.metadata["stage2_latency"] = float(stage2_summary.mean_latency)
        stage2_result.metadata["stage2_token_cost"] = float(stage2_summary.mean_token_cost)
        baseline_summary = prepared_structure.stage1_summary
        if baseline_summary is not None:
            selection_meta = _selection_metadata(
                stage1_summary=baseline_summary,
                stage2_summary=stage2_summary,
                risk_std_penalty=self.search_config.risk_std_penalty,
            )
            stage2_result.metadata.update(selection_meta)
            oracle_decision = str(selection_meta["selection_decision"])
            oracle_reason = str(selection_meta["selection_reason"])
            stage2_result.metadata["oracle_selection_decision"] = oracle_decision
            stage2_result.metadata["oracle_selection_reason"] = oracle_reason
            if oracle_decision == "fallback_stage1" and allow_stage1_fallback:
                final_summary = baseline_summary
                final_signature = prepared_structure.stage1_signature
                final_output = prepared_structure.stage1_output
                learning_target = max(
                    0.0,
                    learning_target - self.stage2_config.learning.fallback_penalty - max(0.0, _summary_target(baseline_summary) - learning_target),
                )
                stage2_result.metadata["fallback_applied"] = True
                stage2_result.metadata["final_selection_decision"] = "fallback_stage1"
                stage2_result.metadata["final_selection_reason"] = oracle_reason
            else:
                stage2_result.metadata["fallback_applied"] = False
                stage2_result.metadata["final_selection_decision"] = "use_stage2"
                stage2_result.metadata["final_selection_reason"] = (
                    "oracle_fallback_disabled" if oracle_decision == "fallback_stage1" else oracle_reason
                )
        else:
            stage2_result.metadata["selection_decision"] = "use_stage2_no_stage1_baseline"
            stage2_result.metadata["selection_reason"] = "no_stage1_baseline"
            stage2_result.metadata["oracle_selection_decision"] = "use_stage2_no_stage1_baseline"
            stage2_result.metadata["oracle_selection_reason"] = "no_stage1_baseline"
            stage2_result.metadata["fallback_applied"] = False
            stage2_result.metadata["final_selection_decision"] = "use_stage2"
            stage2_result.metadata["final_selection_reason"] = "no_stage1_baseline"
        if learn:
            learning_stats = self._stage2.learn_from_run(
                prepared_structure.union_graph,
                stage2_result,
                dataset_profile=profile,
                summary=stage2_summary,
                reward_target=learning_target,
            )
            stage2_result.metadata["learning_stats"] = learning_stats
        stage2_result.metadata["structure_source"] = str(
            prepared_structure.metadata.get("source")
            or ("live_stage1_search" if stage1_result is not None else "prepared_stage1_artifact")
        )
        stage2_result.metadata["stage1_signature"] = prepared_structure.stage1_signature
        return Stage2PipelineResult(
            stage1_result=stage1_result,
            stage1_artifact=prepared_structure,
            stage2_result=stage2_result,
            final_summary=final_summary,
            final_signature=final_signature,
            final_output=final_output,
        )

    def search(
        self,
        question_text: str,
        *,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
        dataset_name: Optional[str] = None,
        learn: bool = False,
        replay_dir: Optional[str] = None,
        allow_stage1_fallback: bool = True,
    ) -> Stage2PipelineResult:
        stage1_result, artifact, resolved_dataset = self._run_stage1_search(
            question_text,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_name=dataset_name,
        )
        return self.search_prepared(
            question_text,
            prepared_structure=artifact,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_name=resolved_dataset,
            replay_dir=replay_dir,
            stage1_result=stage1_result,
            learn=learn,
            allow_stage1_fallback=allow_stage1_fallback,
        )

    def state_dict(self) -> dict:
        return {
            "search_config": asdict(self.search_config),
            "runtime_config": asdict(self.runtime_config),
            "union_config": asdict(self.union_config),
            "stage2_config": asdict(self.stage2_config),
            "stage1_pipeline": self._stage1.state_dict(),
            "stage2_runtime": self._stage2.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        stage1_state = state.get("stage1_pipeline")
        if isinstance(stage1_state, dict):
            self._stage1.load_state_dict(stage1_state)
        stage2_state = state.get("stage2_runtime")
        if isinstance(stage2_state, dict):
            self._stage2.load_state_dict(stage2_state)

    def save_checkpoint(self, path: str, *, metadata: Optional[dict] = None) -> None:
        resolved_metadata = dict(metadata or {})
        if hasattr(self._stage2, "save_binary_state"):
            binary_path = f"{path}.stage2.pt"
            self._stage2.save_binary_state(binary_path)
            resolved_metadata["stage2_binary_state_path"] = binary_path
        payload = {
            "metadata": resolved_metadata,
            "pipeline_state": self.state_dict(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def load_checkpoint(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.load_state_dict(dict(payload.get("pipeline_state", {})))
        metadata = payload.get("metadata", {})
        if hasattr(self._stage2, "load_binary_state"):
            binary_path = str((metadata or {}).get("stage2_binary_state_path", ""))
            if binary_path and os.path.exists(binary_path):
                self._stage2.load_binary_state(binary_path)
        return dict(metadata) if isinstance(metadata, dict) else {}
