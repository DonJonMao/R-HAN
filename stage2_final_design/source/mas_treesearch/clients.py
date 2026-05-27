from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .cache import DictCache
from .config import ChatConfig, EmbeddingConfig
from .types import Vector


def _normalize(vec: Sequence[float]) -> Vector:
    vals = [float(x) for x in vec]
    norm = sum(v * v for v in vals) ** 0.5
    if norm <= 1e-12:
        return [0.0 for _ in vals]
    return [v / norm for v in vals]


@dataclass
class OpenAICompatClient:
    chat_config: ChatConfig

    def resolve_model_id(self) -> str:
        if self.chat_config.model:
            return self.chat_config.model
        base = self.chat_config.api_base.rstrip("/")
        url = f"{base}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("Failed to resolve chat model id from /v1/models") from exc
        models = data.get("data", []) if isinstance(data, dict) else []
        if not models:
            raise RuntimeError(f"No models returned from {url}")
        model_id = models[0].get("id")
        if not model_id:
            raise RuntimeError(f"Malformed models response: {data}")
        self.chat_config.model = str(model_id)
        return self.chat_config.model

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        model_id = model or self.resolve_model_id()
        base = self.chat_config.api_base.rstrip("/")
        url = f"{base}/v1/chat/completions"
        payload = json.dumps(
            {
                "model": model_id,
                "messages": messages,
                "temperature": self.chat_config.temperature if temperature is None else temperature,
                "max_tokens": self.chat_config.max_tokens if max_tokens is None else max_tokens,
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.chat_config.api_key:
            req.add_header("Authorization", f"Bearer {self.chat_config.api_key}")

        last_err: Optional[Exception] = None
        for _ in range(max(1, int(self.chat_config.max_retries) + 1)):
            try:
                with urllib.request.urlopen(req, timeout=self.chat_config.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                data = json.loads(body)
                msg = data.get("choices", [{}])[0].get("message", {})
                content = msg.get("content", "")
                if content:
                    return str(content)
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"Chat request failed: {last_err}")


@dataclass
class CachedEmbedder:
    config: EmbeddingConfig
    cache: DictCache[Vector]

    def embed(self, text: str) -> Vector:
        key = text.strip()
        if not key:
            return [0.0] * self.config.dim
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        base = self.config.api_base.rstrip("/")
        url = f"{base}/v1/embeddings"
        payload = json.dumps({"model": self.config.model, "input": text}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.config.api_key:
            req.add_header("Authorization", f"Bearer {self.config.api_key}")
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
        data = json.loads(body)
        emb = data.get("data", [{}])[0].get("embedding")
        if not emb:
            raise RuntimeError(f"Embedding response missing vector: {data}")
        value = list(_normalize(emb))
        if len(value) > self.config.dim:
            value = value[: self.config.dim]
        elif len(value) < self.config.dim:
            value = value + [0.0] * (self.config.dim - len(value))
        return self.cache.put(key, value)
