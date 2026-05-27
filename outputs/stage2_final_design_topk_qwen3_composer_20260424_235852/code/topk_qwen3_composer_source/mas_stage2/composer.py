"""Memory Composer: 可学习的记忆压缩器

将原始记忆文本压缩为固定长度的 latent vector，用于注入到 LLM hidden states。
"""

from __future__ import annotations

import hashlib
import re
import torch
import torch.nn as nn
from typing import List, Optional, Sequence
from dataclasses import dataclass


@dataclass
class MemoryComposerConfig:
    """Memory Composer 配置"""
    hidden_dim: int = 4096  # LLM 隐藏维度
    latent_length: int = 8  # latent memory 序列长度
    encoder_layers: int = 2  # Encoder 层数
    dropout: float = 0.1
    max_input_length: int = 2048  # 最大输入长度


def _stable_token_id(token: str, vocab_size: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, vocab_size - 1) + 1


def tokenize_texts(
    texts: Sequence[str],
    *,
    vocab_size: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将文本稳定地哈希成 token ids。

    这里不依赖外部 tokenizer，便于在当前工程里快速验证 composer 闭环。
    """

    tokens: List[int] = []
    for text in texts:
        pieces = re.findall(r"\w+|[^\w\s]", str(text).lower())
        tokens.extend(_stable_token_id(piece, vocab_size) for piece in pieces)
        if len(tokens) >= max_length:
            break
    if not tokens:
        tokens = [_stable_token_id("<empty>", vocab_size)]
    tokens = tokens[:max_length]
    input_ids = torch.zeros(1, max_length, dtype=torch.long)
    attention_mask = torch.zeros(1, max_length, dtype=torch.bool)
    input_ids[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    attention_mask[0, : len(tokens)] = True
    return input_ids, attention_mask


class MemoryComposer(nn.Module):
    """可学习的记忆压缩器

    输入：role_profile + raw_memories (文本)
    输出：latent_memory ∈ R^(L'×D)
    """

    def __init__(self, config: MemoryComposerConfig, tokenizer, text_encoder):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder  # 用于编码文本（可以是 LLM 的 encoder 部分）

        # Compressor: 将可变长度的 hidden states 压缩为固定长度
        self.compressor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.encoder_layers
        )

        # Learnable queries for cross-attention (用于固定输出长度)
        self.latent_queries = nn.Parameter(
            torch.randn(1, config.latent_length, config.hidden_dim)
        )

        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        role_profile: str,
        raw_memories: List[str],
        return_attention: bool = False
    ) -> torch.Tensor:
        """压缩记忆

        Args:
            role_profile: 角色描述
            raw_memories: 原始记忆文本列表
            return_attention: 是否返回 attention weights

        Returns:
            latent_memory: R^(L'×D) 的 latent vector
        """
        # 1. 拼接输入文本
        input_text = self._build_input_text(role_profile, raw_memories)

        # 2. Tokenize 并编码
        with torch.no_grad():  # text_encoder 可能是冻结的 LLM encoder
            hidden_states = self._encode_text(input_text)  # (1, L, D)

        # 3. 通过 compressor 处理
        compressed = self.compressor(hidden_states)  # (1, L, D)

        # 4. Cross-attention 压缩到固定长度
        queries = self.latent_queries.expand(hidden_states.size(0), -1, -1)  # (B, L', D)
        latent_memory, attn_weights = self.cross_attn(
            query=queries,
            key=compressed,
            value=compressed
        )  # (B, L', D)

        # 5. Residual + Norm
        latent_memory = self.norm(latent_memory + queries)

        if return_attention:
            return latent_memory, attn_weights
        return latent_memory

    def _build_input_text(self, role_profile: str, raw_memories: List[str]) -> str:
        """构造输入文本"""
        parts = [f"Role: {role_profile}"]

        if raw_memories:
            parts.append("Memories:")
            for i, memory in enumerate(raw_memories[:10], 1):  # 最多 10 条
                parts.append(f"{i}. {memory}")
        else:
            parts.append("No previous memories.")

        return "\n".join(parts)

    def _encode_text(self, text: str) -> torch.Tensor:
        """编码文本为 hidden states"""
        # Tokenize
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.config.max_input_length,
            truncation=True,
            padding=True
        )

        # Encode (使用 text_encoder，可能是 LLM 的 encoder 部分)
        if hasattr(self.text_encoder, 'model'):
            # 如果是完整的 LLM，只用 encoder 部分
            hidden_states = self.text_encoder.model.embed_tokens(tokens.input_ids)
        else:
            # 如果是独立的 encoder
            hidden_states = self.text_encoder(**tokens).last_hidden_state

        return hidden_states

    def freeze(self):
        """冻结参数"""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """解冻参数"""
        for param in self.parameters():
            param.requires_grad = True


class SimpleMemoryComposer(nn.Module):
    """简化版 Memory Composer（用于快速验证）

    不依赖 LLM encoder，直接用简单的 embedding + transformer
    """

    def __init__(self, config: MemoryComposerConfig, vocab_size: int = 50000):
        super().__init__()
        self.config = config

        # Simple embedding
        self.embedding = nn.Embedding(vocab_size, config.hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_input_length, config.hidden_dim)
        )

        # Transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=8,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.encoder_layers
        )

        # Learnable queries
        self.latent_queries = nn.Parameter(
            torch.randn(1, config.latent_length, config.hidden_dim)
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(config.hidden_dim)

    def _compose_from_embeddings(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compress pre-computed semantic embeddings into latent memory.

        The Stage2-GCR+ runtime feeds this path with cached Qwen3/vLLM
        embeddings. The token-id ``forward`` method is retained for old
        checkpoints and small unit tests.
        """

        if input_embeddings.dim() != 3:
            raise ValueError("input_embeddings must have shape (batch, sequence, hidden_dim)")
        if input_embeddings.size(-1) != self.config.hidden_dim:
            raise ValueError(
                f"embedding dim {input_embeddings.size(-1)} does not match composer hidden_dim {self.config.hidden_dim}"
            )
        seq_len = min(int(input_embeddings.size(1)), int(self.config.max_input_length))
        if seq_len <= 0:
            raise ValueError("input_embeddings must contain at least one sequence item")

        x = input_embeddings[:, :seq_len, :]
        if attention_mask is None:
            attention_mask = torch.ones(x.size(0), seq_len, dtype=torch.bool, device=x.device)
        else:
            attention_mask = attention_mask[:, :seq_len].to(device=x.device, dtype=torch.bool)

        x = x + self.pos_encoding[:, :seq_len, :].to(device=x.device, dtype=x.dtype)
        encoded = self.encoder(x, src_key_padding_mask=~attention_mask)

        queries = self.latent_queries.to(device=x.device, dtype=x.dtype).expand(x.size(0), -1, -1)
        latent_memory, _ = self.cross_attn(
            query=queries,
            key=encoded,
            value=encoded,
            key_padding_mask=~attention_mask,
        )
        return self.norm(latent_memory + queries)

    def forward_embeddings(
        self,
        input_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compress dense semantic embeddings directly."""

        return self._compose_from_embeddings(input_embeddings, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) attention mask

        Returns:
            latent_memory: (B, L', D)
        """
        x = self.embedding(input_ids)  # (B, L, D)
        return self._compose_from_embeddings(x, attention_mask)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
