from __future__ import annotations

import json
import os
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


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n\n[...truncated for OpenAI-compatible API context limit...]\n\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def _compact_messages(messages: List[Dict[str, str]], *, max_chars_per_message: int) -> List[Dict[str, str]]:
    compacted: List[Dict[str, str]] = []
    for message in messages:
        item = dict(message)
        content = str(item.get("content", ""))
        item["content"] = _truncate_middle(content, max_chars_per_message)
        compacted.append(item)
    return compacted


def _read_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    body = body.strip()
    if not body:
        return f"HTTP Error {exc.code}: {exc.reason}"
    return f"HTTP Error {exc.code}: {exc.reason}; body={body[:1000]}"


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

        resolved_max_tokens = self.chat_config.max_tokens if max_tokens is None else int(max_tokens)
        resolved_temperature = self.chat_config.temperature if temperature is None else float(temperature)

        def build_payload(request_messages: List[Dict[str, str]], request_max_tokens: int) -> bytes:
            return json.dumps(
                {
                    "model": model_id,
                    "messages": request_messages,
                    "temperature": resolved_temperature,
                    "max_tokens": request_max_tokens,
                },
                ensure_ascii=False,
            ).encode("utf-8")

        def build_request(request_messages: List[Dict[str, str]], request_max_tokens: int) -> urllib.request.Request:
            req = urllib.request.Request(url, data=build_payload(request_messages, request_max_tokens), method="POST")
            req.add_header("Content-Type", "application/json")
            if self.chat_config.api_key:
                req.add_header("Authorization", f"Bearer {self.chat_config.api_key}")
            return req

        fallback_char_limits = [
            int(value)
            for value in os.getenv("LLM_400_FALLBACK_MESSAGE_CHARS", "24000,16000,12000,8000").split(",")
            if value.strip().isdigit()
        ]
        fallback_token_limits = []
        for value in (resolved_max_tokens, min(resolved_max_tokens, 512), min(resolved_max_tokens, 256)):
            if value > 0 and value not in fallback_token_limits:
                fallback_token_limits.append(value)

        last_err: Optional[Exception] = None
        variants: List[tuple[List[Dict[str, str]], int, str]] = [(messages, resolved_max_tokens, "original")]
        for token_limit in fallback_token_limits:
            if token_limit != resolved_max_tokens:
                variants.append((messages, token_limit, f"max_tokens={token_limit}"))
        for char_limit in fallback_char_limits:
            compacted = _compact_messages(messages, max_chars_per_message=char_limit)
            for token_limit in fallback_token_limits:
                variants.append((compacted, token_limit, f"compact_chars={char_limit},max_tokens={token_limit}"))

        for request_messages, request_max_tokens, variant_name in variants:
            for _ in range(max(1, int(self.chat_config.max_retries) + 1)):
                try:
                    req = build_request(request_messages, request_max_tokens)
                    with urllib.request.urlopen(req, timeout=self.chat_config.timeout_s) as resp:
                        body = resp.read().decode("utf-8")
                    data = json.loads(body)
                    msg = data.get("choices", [{}])[0].get("message", {})
                    content = msg.get("content", "")
                    if content:
                        return str(content)
                except urllib.error.HTTPError as exc:
                    detail = _read_http_error(exc)
                    last_err = RuntimeError(f"{detail}; variant={variant_name}")
                    if int(exc.code) != 400:
                        break
                except Exception as exc:
                    last_err = exc
                    break
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
