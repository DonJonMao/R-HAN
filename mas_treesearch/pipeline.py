from __future__ import annotations

import json
import os
from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Optional

from .agents import AgentPool, default_agent_pool
from .cache import DictCache
from .clients import CachedEmbedder
from .config import ChatConfig, EmbeddingConfig, JudgeConfig, SearchConfig, TieredEvalConfig
from .profiles import resolve_dataset_profile
from .evaluator import MultiFidelityEvaluator
from .gating import TaskConditioner
from .learning import FeatureBuilder, LearnableEditPrior, LearnableValueModel
from .proxy import StaticProxyScorer
from .search import TreeSearchEngine
from .types import SearchResult


def _env_chat_config() -> ChatConfig:
    return ChatConfig(
        api_base=os.getenv("LLM_API_BASE", "http://127.0.0.1:8039"),
        model=os.getenv("LLM_MODEL") or None,
        api_key=os.getenv("LLM_API_KEY") or None,
        timeout_s=float(os.getenv("LLM_TIMEOUT_S", "60")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )


def _env_judge_config() -> JudgeConfig:
    return JudgeConfig(
        api_base=os.getenv("LLM_JUDGE_API_BASE") or None,
        model=os.getenv("LLM_JUDGE_MODEL") or None,
        api_key=os.getenv("LLM_JUDGE_API_KEY") or None,
        timeout_s=float(os.getenv("LLM_JUDGE_TIMEOUT_S", "60")),
        temperature=float(os.getenv("LLM_JUDGE_TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("LLM_JUDGE_MAX_TOKENS", "128")),
    )


def _env_embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        api_base=os.getenv("EMBED_API_BASE", "http://127.0.0.1:8018"),
        model=os.getenv("EMBED_MODEL", "/mnt/nvme/Qwen3-Embedding-8B"),
        api_key=os.getenv("EMBED_API_KEY") or None,
        timeout_s=float(os.getenv("EMBED_TIMEOUT_S", "30")),
        dim=int(os.getenv("EMBED_DIM", "256")),
    )


@dataclass
class TreeSearchMASPipeline:
    search_config: SearchConfig = field(default_factory=SearchConfig)
    runtime_config: TieredEvalConfig = field(default_factory=TieredEvalConfig)
    agent_pool: Optional[AgentPool] = None

    def __post_init__(self) -> None:
        if self.agent_pool is None:
            self.agent_pool = default_agent_pool()
        self.runtime_config.chat = _env_chat_config()
        self.runtime_config.judge = _env_judge_config()
        self.runtime_config.embedding = _env_embedding_config()
        self._embedder = CachedEmbedder(self.runtime_config.embedding, DictCache())
        self._conditioner = TaskConditioner(self.agent_pool, self._embedder, self.search_config)
        self._proxy = StaticProxyScorer(self.agent_pool, self._conditioner.agent_vectors)
        self._feature_builder = FeatureBuilder(self.agent_pool, self._conditioner.agent_vectors)
        self._edit_prior = (
            LearnableEditPrior(learning_rate=self.search_config.learned_prior_lr)
            if self.search_config.enable_learned_edit_prior
            else None
        )
        self._value_model = (
            LearnableValueModel(
                learning_rate=self.search_config.learned_value_lr,
                init_uncertainty=self.search_config.learned_value_uncertainty_init,
            )
            if self.search_config.enable_learned_value_model
            else None
        )
        self._evaluator = MultiFidelityEvaluator(self.runtime_config, self.agent_pool)
        self._search = TreeSearchEngine(
            config=self.search_config,
            agent_pool=self.agent_pool,
            proxy_scorer=self._proxy,
            evaluator=self._evaluator,
            feature_builder=self._feature_builder,
            edit_prior=self._edit_prior,
            value_model=self._value_model,
        )

    def search(
        self,
        question_text: str,
        *,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
        dataset_name: Optional[str] = None,
        learn: bool = True,
    ) -> SearchResult:
        resolved_dataset = dataset_name or ((metadata or {}).get("mas_dataset_name") if isinstance(metadata, dict) else None)
        profile = resolve_dataset_profile(resolved_dataset)
        selection = self._conditioner.select(question_text, profile=profile)
        return self._search.search(
            question_text=question_text,
            selection=selection,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=profile,
            learn=learn,
        )

    def state_dict(self) -> dict:
        return {
            "search_config": asdict(self.search_config),
            "runtime_config": asdict(self.runtime_config),
            "chat_config": asdict(self.runtime_config.chat),
            "judge_config": asdict(self.runtime_config.judge),
            "embedding_config": asdict(self.runtime_config.embedding),
            "edit_prior": self._edit_prior.state_dict() if self._edit_prior is not None else None,
            "value_model": self._value_model.state_dict() if self._value_model is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        edit_prior_state = state.get("edit_prior")
        if edit_prior_state is not None and self._edit_prior is not None:
            self._edit_prior.load_state_dict(dict(edit_prior_state))
        value_model_state = state.get("value_model")
        if value_model_state is not None and self._value_model is not None:
            self._value_model.load_state_dict(dict(value_model_state))

    def save_checkpoint(self, path: str, *, metadata: Optional[dict] = None) -> None:
        payload = {
            "metadata": metadata or {},
            "pipeline_state": self.state_dict(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
