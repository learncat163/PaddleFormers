# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# Adapted for PaddlePaddle / paddleformers from the original Apple OpenELM implementation.
# Original code: based on transformers/PyTorch. Migrated to PaddlePaddle.

from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

# 原始代码: from transformers.utils import logging
from ...utils.log import logger

# 原始代码: from transformers.cache_utils import Cache, DynamicCache, StaticCache
from ..cache_utils import Cache, DynamicCache

# 原始代码: from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ..model_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# 原始代码: from transformers import PreTrainedModel
from ..model_utils import PretrainedModel, register_base_model
from .configuration import OpenELMConfig, make_divisible


class OpenELMRMSNorm(nn.Layer):
    """RMS Normalization layer.

    原始代码: class OpenELMRMSNorm(nn.Module)
    """

    def __init__(self, num_features: int, eps: float = 1e-6):
        """
        Initialize the OpenELMRMSNorm normalization layer.

        Args:
            num_features (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (paddle.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        # 原始代码: self.weight = nn.Parameter(torch.ones(num_features))
        self.weight = paddle.create_parameter(
            shape=[num_features],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.num_features = num_features

    def _norm(self, x: Tensor) -> Tensor:
        """
        Apply the OpenELMRMSNorm normalization to the input tensor.

        原始代码: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        Args:
            x (paddle.Tensor): The input tensor.

        Returns:
            paddle.Tensor: The normalized tensor.

        """
        # 原始代码: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * paddle.rsqrt(paddle.mean(paddle.pow(x, 2), axis=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the OpenELMRMSNorm layer.

        原始代码: output = self._norm(x.float()).type_as(x); return output * self.weight

        Args:
            x (paddle.Tensor): The input tensor.

        Returns:
            paddle.Tensor: The output tensor after applying OpenELMRMSNorm.

        """
        # 原始代码: output = self._norm(x.float()).type_as(x)
        output = self._norm(x.astype(paddle.float32)).astype(x.dtype)
        return output * self.weight

    def extra_repr(self) -> str:
        return super().extra_repr() + f"num_features={self.num_features}, eps={self.eps}"


class OpenELMPreTrainedModel(PretrainedModel):
    """
    原始代码: class OpenELMPreTrainedModel(PreTrainedModel)
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = OpenELMConfig
    base_model_prefix = "transformer"
    _no_split_modules = ["OpenELMDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs) -> None:
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module: nn.Layer) -> None:
        """Initialize the weights.

        原始代码: _init_weights 使用 torch.nn.init.normal_ / zero_
        """
        if isinstance(module, nn.Linear):
            # 原始代码: module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=self.config.initializer_range,
                    shape=module.weight.shape,
                ).cast(module.weight.dtype)
            )
            if module.bias is not None:
                # 原始代码: module.bias.data.zero_()
                module.bias.set_value(paddle.zeros(module.bias.shape, dtype=module.bias.dtype))
        elif isinstance(module, nn.Embedding):
            # 原始代码: module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.set_value(
                paddle.normal(
                    mean=0.0,
                    std=self.config.initializer_range,
                    shape=module.weight.shape,
                ).cast(module.weight.dtype)
            )
        elif isinstance(module, OpenELMRMSNorm):
            # 原始代码: module.weight.data.fill_(1.0)
            module.weight.set_value(paddle.ones(module.weight.shape, dtype=module.weight.dtype))


def _rotate_half(x: Tensor) -> Tensor:
    """
    原始代码:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    """
    # 原始代码: x1, x2 = x.chunk(2, dim=-1)
    x1, x2 = paddle.chunk(x, chunks=2, axis=-1)
    # 原始代码: torch.cat((-x2, x1), dim=-1)
    return paddle.concat([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(x: Tensor, pos_sin: Tensor, pos_cos: Tensor) -> Tensor:
    """
    原始代码: return (x * pos_cos) + (_rotate_half(x) * pos_sin)
    """
    return (x * pos_cos) + (_rotate_half(x) * pos_sin)


class OpenELMRotaryEmbedding(nn.Layer):
    """
    The rotary position embeddings (aka RoPE) from `RoFormer <https://arxiv.org/abs/2104.09864>`_.

    RoPE encodes the position information of tokens using a rotation matrix, and is able to capture
    explicit relative positional dependencies.

    原始代码: class OpenELMRotaryEmbedding(torch.nn.Module)

    Args:
        model_dim: The dimensionality of the model's hidden state.
        max_seq_length: Maximum sequence length.
        freq_constant: A constant used for computing frequencies.
    """

    def __init__(self, model_dim: int, max_seq_length: int, freq_constant: int = 10000) -> None:
        # 原始代码:
        # inv_freq = 1.0 / (
        #     freq_constant
        #     ** (torch.arange(0, model_dim, 2, dtype=torch.float32) / model_dim)
        # )
        inv_freq = 1.0 / (freq_constant ** (paddle.arange(0, model_dim, 2, dtype="float32") / model_dim))
        super().__init__()

        self.model_dim = model_dim
        self.freq_constant = freq_constant
        self.max_seq_length = max_seq_length

        # 原始代码: self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("inv_freq", inv_freq, persistable=False)
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = max_seq_length
        self._compute_sin_cos_embeddings(max_seq_length)

    def extra_repr(self) -> str:
        return (
            f"\tmodel_dim={self.model_dim}, max_seq_length={self.max_seq_length}, freq_constant={self.freq_constant}"
        )

    def _compute_sin_cos_embeddings(
        self,
        key_len: int,
        key_place=None,
        key_dtype: paddle.dtype = paddle.float32,
    ) -> None:
        """
        Compute sine and cos embeddings.

        原始代码: _compute_sin_cos_embeddings(self, key_len, key_device=torch.device("cpu"), key_dtype=torch.float32)

        Args:
            key_len: Number of tokens in the key embeddings in the transformer model.
            key_place: Device where the key embeddings are stored (paddle place).
            key_dtype: Data type of the key embeddings.

        Returns:
            None

        ...note:
            We recalculate the sine and cosine embeddings if any of the following conditions are met:
                1. The number of tokens in key embeddings are greater than the cached sequence length.
                2. Sine and cosine caches are empty.
                3. The device and data type of sine and cosine embeddings does not match with the key embeddings.
        """
        # 原始代码: 检查 device 和 dtype 是否匹配
        need_recompute = key_len > self._cached_seq_length or self._cached_cos is None or self._cached_sin is None
        if not need_recompute and key_place is not None:
            need_recompute = self._cached_cos.place != key_place or self._cached_sin.place != key_place
        if not need_recompute and self._cached_cos is not None:
            need_recompute = self._cached_cos.dtype != key_dtype or self._cached_sin.dtype != key_dtype

        if need_recompute:
            self._cached_seq_length = max(key_len, self._cached_seq_length)

            # 原始代码: pos_index = torch.arange(self._cached_seq_length, dtype=torch.float32, device=self.inv_freq.device)
            pos_index = paddle.arange(
                self._cached_seq_length,
                dtype="float32",
            )
            # 原始代码: pos_index_theta = torch.einsum("i,j->ij", pos_index, self.inv_freq)
            pos_index_theta = paddle.einsum("i,j->ij", pos_index, self.inv_freq)
            # 原始代码: emb = torch.cat((pos_index_theta, pos_index_theta), dim=-1)
            emb = paddle.concat((pos_index_theta, pos_index_theta), axis=-1)

            # 原始代码: cos_emb = emb.cos().to(dtype=key_dtype, device=key_device)
            cos_emb = emb.cos().cast(key_dtype)
            sin_emb = emb.sin().cast(key_dtype)

            # 原始代码: self._cached_cos = cos_emb[None, None, :, :]
            self._cached_cos = cos_emb[None, None, :, :]
            self._cached_sin = sin_emb[None, None, :, :]

    def forward(
        self,
        query: paddle.Tensor,
        key: paddle.Tensor,
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        The forward function of RoPE embeddings.

        原始代码: def forward(self, query: torch.Tensor, key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]

        Args:
            query: Query embeddings in the transformer model.
            key: Key embeddings in the transformer model.

        Returns:
            A tuple containing the query and key embeddings with positional information.
        """
        dim = key.shape[-1]
        key_len = key.shape[2]
        query_len = query.shape[2]

        assert dim == self.model_dim
        assert key.place == query.place
        assert key.dtype == query.dtype

        assert key_len >= query_len, "Number of keys has to be greater than or equal to number of queries."

        # 原始代码: query_float = query.float()
        query_float = query.astype(paddle.float32)
        key_float = key.astype(paddle.float32)

        self._compute_sin_cos_embeddings(key_len, key_place=key_float.place, key_dtype=key_float.dtype)
        query_float = _apply_rotary_pos_emb(
            x=query_float,
            pos_sin=self._cached_sin[..., key_len - query_len : key_len, :],
            pos_cos=self._cached_cos[..., key_len - query_len : key_len, :],
        )
        key_float = _apply_rotary_pos_emb(
            x=key_float,
            pos_sin=self._cached_sin[..., :key_len, :],
            pos_cos=self._cached_cos[..., :key_len, :],
        )

        # 原始代码: return query_float.type_as(query), key_float.type_as(key)
        return query_float.astype(query.dtype), key_float.astype(key.dtype)


class OpenELMMultiHeadCausalAttention(nn.Layer):
    """
    Multi-head causal attention with optional Group Query Attention (GQA).

    原始代码: class OpenELMMultiHeadCausalAttention(nn.Module)
    """

    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]

        # 原始代码: self.qkv_proj = nn.Linear(in_features=config.model_dim, out_features=(q_heads + k_heads + v_heads) * head_dim, bias=False)
        self.qkv_proj = nn.Linear(
            in_features=config.model_dim,
            out_features=(q_heads + k_heads + v_heads) * head_dim,
            bias_attr=False,
        )

        self.pos_embedding = OpenELMRotaryEmbedding(
            model_dim=config.head_dim,
            max_seq_length=config.rope_max_length,
            freq_constant=config.rope_freq_constant,
        )

        if config.normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
            self.k_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # 原始代码: self.out_proj = nn.Linear(in_features=q_heads * head_dim, out_features=config.model_dim, bias=False)
        self.out_proj = nn.Linear(
            in_features=q_heads * head_dim,
            out_features=config.model_dim,
            bias_attr=False,
        )

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.transformer_dim = config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"query_heads={self.num_q_heads}, key_heads={self.num_k_heads}, value_heads={self.num_v_heads}"
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """
        Forward pass of multi-head self-attention.

        原始代码: def forward(self, hidden_states: torch.Tensor, ...)

        Args:
            hidden_states: Input tensor of the shape [batch size, sequence length, model dimension].
            past_key_value: Tensor storing the cached keys and values.
            output_attentions: output attention weights.
            use_cache: Specifies whether to use kv-cache for generation.
            cache_position: used for updating the kv-cache.

        Returns:
            The output of the same shape as the input, optionally with a tensor containing cached keys and values.
        """

        # scaled_dot_product_attention does not return attention weights, set output_attentions to False
        output_attentions = False
        # 原始代码: batch_size, seq_length, d_model = hidden_states.size()
        batch_size, seq_length, d_model = hidden_states.shape

        # [B, S, d] --> [B, S, (q_h + k_h + v_h) * h]
        qkv = self.qkv_proj(hidden_states)
        # [B, S, (q_h + k_h + v_h) * h] --> [B, S, (q_h + k_h + v_h), h]
        qkv = qkv.reshape(
            [
                batch_size,
                seq_length,
                self.num_q_heads + self.num_k_heads + self.num_v_heads,
                self.head_dim,
            ]
        )
        # [B, S, (q_h + k_h + v_h), h] --> [B, (q_h + k_h + v_h), S, h]
        # 原始代码: qkv = qkv.transpose(1, 2)
        qkv = qkv.transpose([0, 2, 1, 3])
        # [B, (q_h + k_h + v_h), S, h] --> [B, q_h, S, h], [B, k_h, S, h], [B, v_h, S, h]
        # 原始代码: queries, keys, values = qkv.split([self.num_q_heads, self.num_k_heads, self.num_v_heads], dim=1)
        queries, keys, values = paddle.split(
            qkv,
            [self.num_q_heads, self.num_k_heads, self.num_v_heads],
            axis=1,
        )

        if self.q_norm is not None:
            queries = self.q_norm(queries)

        if self.k_norm is not None:
            keys = self.k_norm(keys)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # 原始代码:
            # cache_kwargs = {"cache_position": cache_position}
            # keys, values = past_key_value.update(keys, values, self.layer_idx, cache_kwargs)
            cache_kwargs = {"cache_position": cache_position}
            keys, values = past_key_value.update(keys, values, self.layer_idx, cache_kwargs)

        # Add positional embedding
        queries, keys = self.pos_embedding(queries, keys)

        if self.num_groups != 1:
            # GQA
            # [B, k_h, S, h] --> [B, q_h, S, h]
            # 原始代码: keys = keys.repeat_interleave(self.num_groups, dim=1)
            keys = paddle.repeat_interleave(keys, self.num_groups, axis=1)
            # [B, v_h, S, h] --> [B, q_h, S, h]
            # 原始代码: values = values.repeat_interleave(self.num_groups, dim=1)
            values = paddle.repeat_interleave(values, self.num_groups, axis=1)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : keys.shape[-2]]

        # 原始代码:
        # attn_output = F.scaled_dot_product_attention(
        #     queries, keys, values, attn_mask=causal_mask, dropout_p=0,
        # )
        # 迁移说明: 使用手动实现的 scaled dot-product attention 以确保兼容性
        scale = self.head_dim**-0.5
        # [B, q_h, S_q, h] x [B, q_h, h, S_k] -> [B, q_h, S_q, S_k]
        attn_weights = paddle.matmul(queries * scale, keys, transpose_y=True)
        if causal_mask is not None:
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, axis=-1)
        attn_output = paddle.matmul(attn_weights, values)

        # 原始代码: attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.transpose([0, 2, 1, 3])
        # 原始代码: attn_output = attn_output.reshape(batch_size, seq_length, self.num_q_heads * self.head_dim)
        attn_output = attn_output.reshape([batch_size, seq_length, self.num_q_heads * self.head_dim])
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class OpenELMFeedForwardNetwork(nn.Layer):
    """
    Feed-forward network with optional GLU.

    原始代码: class OpenELMFeedForwardNetwork(nn.Module)
    """

    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,
                divisor=config.ffn_dim_divisor,
            )
        )
        if config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            # 原始代码: self.proj_1 = nn.Linear(in_features=config.model_dim, out_features=2 * intermediate_dim, bias=False)
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=2 * intermediate_dim,
                bias_attr=False,
            )
            # 原始代码: self.proj_2 = nn.Linear(in_features=intermediate_dim, out_features=config.model_dim, bias=False)
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias_attr=False,
            )
            self.ffn_with_glu = True
        else:
            # Standard FFN, as described in https://arxiv.org/abs/1706.03762
            self.proj_1 = nn.Linear(
                in_features=config.model_dim,
                out_features=intermediate_dim,
                bias_attr=False,
            )
            self.proj_2 = nn.Linear(
                in_features=intermediate_dim,
                out_features=config.model_dim,
                bias_attr=False,
            )
            self.ffn_with_glu = False

        # 原始代码: self.act = ACT2FN[config.activation_fn_name]
        # 迁移说明: ACT2FN["swish"] 等价于 paddle.nn.Silu()
        if config.activation_fn_name == "swish":
            self.act = nn.Silu()
        elif config.activation_fn_name == "gelu":
            self.act = nn.GELU()
        elif config.activation_fn_name == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {config.activation_fn_name}")

    def extra_repr(self) -> str:
        return super().extra_repr() + f"(ffn_with_glu) : {self.ffn_with_glu}"

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of FFN layer.

        原始代码: def forward(self, x: Tensor) -> Tensor

        Args:
            x: Input tensor of the shape [batch size, sequence length, model dimension].

        Returns:
            A tensor of the same shape as the input.
        """
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            # 原始代码: y_1, y_2 = y_12.chunk(2, dim=-1)
            y_1, y_2 = paddle.chunk(y_12, chunks=2, axis=-1)
            y = self.act(y_1) * y_2
            return self.proj_2(y)
        else:
            return self.proj_2(self.act(self.proj_1(x)))


class OpenELMDecoderLayer(nn.Layer):
    """
    Transformer decoder layer.

    原始代码: class OpenELMDecoderLayer(nn.Module)
    """

    def __init__(self, config: OpenELMConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = OpenELMMultiHeadCausalAttention(config=config, layer_idx=layer_idx)
        self.ffn = OpenELMFeedForwardNetwork(config=config, layer_idx=layer_idx)
        self.ffn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )
        self.attn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        原始代码: def forward(self, hidden_states: torch.Tensor, ...)

        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned.
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


@register_base_model
class OpenELMModel(OpenELMPreTrainedModel):
    """
    OpenELM Model (base model without language modeling head).

    原始代码: class OpenELMModel(OpenELMPreTrainedModel)
    """

    config_class = OpenELMConfig

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.config = config

        # 原始代码: self.token_embeddings = nn.Embedding(embedding_dim=config.model_dim, num_embeddings=config.vocab_size)
        self.token_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.model_dim,
        )

        # 原始代码: self.layers = nn.ModuleList(OpenELMDecoderLayer(...) for layer_idx in range(...))
        self.layers = nn.LayerList(
            [
                OpenELMDecoderLayer(config=config, layer_idx=layer_idx)
                for layer_idx in range(config.num_transformer_layers)
            ]
        )
        self.norm = OpenELMRMSNorm(num_features=config.model_dim)
        if config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = nn.Linear(
                in_features=config.model_dim,
                out_features=config.vocab_size,
                bias_attr=False,
            )
        self.num_transformer_layers = config.num_transformer_layers
        self.gradient_checkpointing = False

        # Register a causal mask to separate causal and padding mask creation.
        # 原始代码:
        # causal_mask = torch.full((config.max_context_length, config.max_context_length), fill_value=True, dtype=torch.bool)
        # self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        causal_mask = paddle.full(
            [config.max_context_length, config.max_context_length],
            fill_value=True,
            dtype="bool",
        )
        self.register_buffer("causal_mask", paddle.triu(causal_mask, diagonal=1), persistable=False)

        # 注意: paddleformers 中不需要调用 self.post_init()，
        # PretrainedModel 的 _post_init 钩子会在 __init__ 完成后自动调用 init_weights()
        # reset_parameters 逻辑已整合到 init_weights() 方法中

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings: paddle.Tensor):
        self.token_embeddings = new_embeddings

    def init_weights(self):
        """Initialize weights for the model.

        原始代码: 通过 post_init() + reset_parameters() 实现
        迁移说明: paddleformers 中 _post_init 钩子会自动调用 init_weights()
        """
        super().init_weights()
        self.reset_parameters(self.config)

    def reset_parameters(self, config: OpenELMConfig) -> None:
        """Initialize the layers in Language Model

        原始代码: def reset_parameters(self, config: OpenELMConfig) -> None

        The initialization scheme is followed, following `OPT <https://arxiv.org/pdf/2205.01068.pdf>`_.

        Args:
            config: model configuration.

        Returns:
            None
        """
        for module in self.sublayers():
            if isinstance(module, nn.Linear):
                std = module.in_features**-0.5
                # 原始代码: torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                module.weight.set_value(paddle.normal(mean=0.0, std=std, shape=module.weight.shape))
                if module.bias is not None:
                    # 原始代码: torch.nn.init.zeros_(module.bias)
                    module.bias.set_value(paddle.zeros(module.bias.shape))
            elif isinstance(module, nn.Embedding):
                std = module._embedding_dim**-0.5
                # 原始代码: torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                module.weight.set_value(paddle.normal(mean=0.0, std=std, shape=module.weight.shape))
            elif isinstance(module, OpenELMRMSNorm):
                if module.weight is not None:
                    # 原始代码: torch.nn.init.ones_(module.weight)
                    module.weight.set_value(paddle.ones(module.weight.shape))

        model_dim = config.model_dim
        n_layers = config.num_transformer_layers
        std = (model_dim**-0.5) * ((2 * n_layers) ** -0.5)
        for param_name, param in self.named_parameters():
            if param_name.endswith("out_proj.weight") or param_name.endswith("ffn.proj_2.weight"):
                # 原始代码: torch.nn.init.normal_(param, mean=0.0, std=std)
                param.set_value(paddle.normal(mean=0.0, std=std, shape=param.shape))

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        原始代码: def forward(self, input_ids: torch.LongTensor = None, ...)

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            position_ids: Position IDs.
            past_key_values: Past key-value cache.
            inputs_embeds: Input embeddings.
            use_cache: Whether to use KV cache.
            output_attentions: Whether to output attention weights.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dict.
            cache_position: Cache position indices.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        # 原始代码:
        # if use_cache:
        #     if not isinstance(past_key_values, StaticCache):
        #         past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        #     past_seen_tokens = past_key_values.get_seq_length()
        # 迁移说明: paddleformers 的 DynamicCache 不需要 from_legacy_cache 转换
        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                # 原始代码: past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                # 迁移说明: paddleformers 的 DynamicCache 直接使用 config 初始化
                past_key_values = DynamicCache(config=self.config)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            # 原始代码: cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
            cache_position = paddle.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # 原始代码: layer_outputs = self._gradient_checkpointing_func(...)
                # 迁移说明: 使用 paddle.distributed.fleet.utils.recompute 实现梯度检查点
                from paddle.distributed.fleet.utils import recompute

                layer_outputs = recompute(
                    decoder_layer,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 原始代码:
        # next_cache = next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
        # 迁移说明: paddleformers 的 Cache 不需要 to_legacy_cache 转换
        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor):
        """
        Update the causal mask for the attention.

        原始代码: def _update_causal_mask(self, attention_mask, input_tensor)
        迁移说明: 移除了 JIT/Dynamo 追踪检测代码，简化了实现
        """
        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype

        # support going beyond cached `max_position_embedding`
        if seq_length > self.causal_mask.shape[-1]:
            # 原始代码: causal_mask = torch.full((2 * ...), fill_value=1)
            causal_mask = paddle.full(
                [2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]],
                fill_value=1,
            )
            self.register_buffer("causal_mask", paddle.triu(causal_mask, diagonal=1), persistable=False)

        # 原始代码: min_dtype = torch.finfo(dtype).min
        min_dtype = paddle.finfo(dtype).min
        # 原始代码: causal_mask = self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype) * min_dtype
        causal_mask = self.causal_mask[None, None, :, :].tile([batch_size, 1, 1, 1]).cast(dtype) * min_dtype

        if attention_mask is not None and len(attention_mask.shape) == 2:
            mask_length = attention_mask.shape[-1]
            # 原始代码: padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
            padding_mask = (causal_mask[..., :mask_length] == 0.0) * (attention_mask[:, None, None, :] == 0.0)
            # 原始代码: causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
            causal_mask[..., :mask_length] = paddle.where(
                padding_mask,
                paddle.full_like(causal_mask[..., :mask_length], min_dtype),
                causal_mask[..., :mask_length],
            )

        return causal_mask


class OpenELMForCausalLM(OpenELMPreTrainedModel):
    """
    OpenELM model with a language modeling head.

    原始代码: class OpenELMForCausalLM(OpenELMPreTrainedModel)
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: OpenELMConfig):
        super().__init__(config)
        self.transformer = OpenELMModel(config)
        self.vocab_size = config.vocab_size
        if config.share_input_output_layers:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias_attr=False)

        # 注意: paddleformers 中不需要调用 self.post_init()
        # PretrainedModel 的 _post_init 钩子会自动处理权重初始化

    def get_input_embeddings(self):
        return self.transformer.token_embeddings

    def set_input_embeddings(self, value):
        self.transformer.token_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[paddle.Tensor] = None,
        attn_mask_startend_row_indices: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        原始代码: def forward(self, input_ids: torch.LongTensor = None, ...)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.lm_head is None:
            # shared embedding weights
            # 原始代码: logits = F.linear(hidden_states, weight=self.transformer.token_embeddings.weight)
            # 迁移说明: Paddle 的 F.linear 期望 weight shape 为 (in_features, out_features)，
            # 而 PyTorch 期望 (out_features, in_features)，因此需要转置
            logits = paddle.matmul(hidden_states, self.transformer.token_embeddings.weight, transpose_y=True)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits[:, : self.config.vocab_size]
        loss = None
        if labels is not None:
            # PaddleFormers SFTDataset 已经做过 label shift（labels = labels[1:] + [-100]），
            # 这里不能再次 shift，否则会造成训练目标错位。
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits.reshape([-1, self.config.vocab_size])
            shift_labels = labels.reshape([-1])
            # 注意: labels 应保持 int64 类型，不需要 cast 到 logits 的 dtype
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if return_dict else None,
            hidden_states=outputs.hidden_states if return_dict else None,
            attentions=outputs.attentions if return_dict else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        原始代码: def prepare_inputs_for_generation(self, input_ids, past_key_values=None, ...)
        """
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = cache_length
                max_cache_length = past_key_values.get_max_cache_shape()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            # 原始代码: position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = attention_mask.cast(paddle.int64).cumsum(axis=-1) - 1
            # 原始代码: position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = paddle.where(
                attention_mask == 0,
                paddle.ones_like(position_ids),
                position_ids,
            )
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 原始代码: cache_position = torch.arange(past_length, past_length + position_ids.shape[-1], device=position_ids.device)
        cache_position = paddle.arange(
            past_length,
            past_length + position_ids.shape[-1],
        )

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        Reorder cache for beam search.

        原始代码: @staticmethod def _reorder_cache(past_key_values, beam_idx)
        """
        if isinstance(past_key_values, Cache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values

        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.place)) for past_state in layer_past),
            )
        return reordered_past
