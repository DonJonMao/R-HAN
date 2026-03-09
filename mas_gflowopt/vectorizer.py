from __future__ import annotations

import hashlib
import json
import random
import urllib.error
import urllib.request
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .math_utils import mean, normalize
from .types import AgentProfile

Vector = List[float]


def build_openai_embedding_encoder(
    api_base: str,
    model: str,
    timeout_s: float = 30.0,
    api_key: Optional[str] = None,
) -> Callable[[str], Sequence[float]]:
    base = api_base.rstrip("/")
    url = f"{base}/v1/embeddings"

    def _encode(text: str) -> Sequence[float]:
        payload = json.dumps({"model": model, "input": text}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Embedding request failed: {exc}") from exc
        data = json.loads(body)
        emb = data.get("data", [{}])[0].get("embedding")
        if not emb:
            raise RuntimeError(f"Embedding response missing vector: {data}")
        return emb

    return _encode


class HashingVectorizer:
    """Deterministic text vectorizer placeholder.

    Replace this with production embeddings (e.g., LLM encoder) later.
    """

    def __init__(
        self,
        dim: int = 32,
        text_encoder: Optional[Callable[[str], Sequence[float]]] = None,
    ):
        self.dim = dim
        self._text_encoder = text_encoder

    def set_text_encoder(self, text_encoder: Optional[Callable[[str], Sequence[float]]]) -> None:
        self._text_encoder = text_encoder

    def _coerce_external_vector(self, vec: Sequence[float]) -> Vector:
        out = [float(x) for x in vec]
        if len(out) > self.dim:
            out = out[: self.dim]
        elif len(out) < self.dim:
            out = out + [0.0] * (self.dim - len(out))
        return normalize(out)

    def vectorize_text(self, text: str) -> Vector:
        if self._text_encoder is not None:
            ext = self._text_encoder(text)
            return self._coerce_external_vector(ext)
        seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(seed_bytes[:8], byteorder="big", signed=False)
        rng = random.Random(seed)
        vec = [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]
        return normalize(vec)

    def vectorize_agent(self, agent: AgentProfile) -> Vector:
        fields = [
            agent.agent_id,
            agent.role,
            agent.profile,
            agent.system_prompt,
            " ".join(f"{k}:{v}" for k, v in sorted(agent.metadata.items())),
        ]
        return self.vectorize_text(" | ".join(fields))

    def vectorize_nodes(self, node_records: Iterable[Dict[str, str]]) -> Dict[str, Vector]:
        out: Dict[str, Vector] = {}
        for rec in node_records:
            node_id = rec["id"]
            text = f"{node_id}|{rec.get('role','')}|{rec.get('profile','')}"
            out[node_id] = self.vectorize_text(text)
        return out

    @staticmethod
    def merge(vectors: Iterable[Vector]) -> Vector:
        return normalize(mean(list(vectors)))
