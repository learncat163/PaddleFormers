import logging

import paddleformers

""" Phi4Flash model configuration"""
import math

logger = logging.getLogger(name=__name__)


class Phi4FlashConfig(paddleformers.transformers.PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Phi4FlashModel`]. It is used to instantiate an Phi4Flash
    model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 51200):
            Vocabulary size of the Phi4Flash model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Phi4FlashModel`].
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 8192):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        resid_pdrop (`float`, *optional*, defaults to 0.0):
            Dropout probability for mlp outputs.
        embd_pdrop (`int`, *optional*, defaults to 0.0):
            The dropout ratio for the embeddings.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio after computing the attention scores.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Phi-1 and Phi-1.5 supports up to 2048
            tokens.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`. Whether to tie weight embeddings or not.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.

    Example:

    ```python
    >>> from transformers import Phi4FlashModel, Phi4FlashConfig

    >>> # Initializing a Phi4Flash style configuration
    >>> configuration = Phi4FlashConfig.from_pretrained("microsoft/Phi4-mini-flash-reasoning")

    >>> # Initializing a model from the configuration
    >>> model = Phi4FlashModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "phi4flash"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=51200,
        hidden_size=2560,
        intermediate_size=9216,
        num_hidden_layers=32,
        num_attention_heads=40,
        num_key_value_heads=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        layer_norm_eps=1e-05,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window=2047,
        mb_per_layer=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank="auto",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.mb_per_layer = mb_per_layer
        self.sliding_window = [
            (
                sliding_window
                if layer_idx < num_hidden_layers // 2 and layer_idx % 2 == 1
                else None
            )
            for layer_idx in range(num_hidden_layers)
        ]
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = (
            math.ceil(self.hidden_size / 16)
            if mamba_dt_rank == "auto"
            else mamba_dt_rank
        )
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )

    @property
    def layers_block_type(self):
        layer_block_types = []
        for i in range(self.num_hidden_layers):
            if i % 2 == 1:
                layer_block_type = (
                    "attention"
                    if i <= self.num_hidden_layers // 2 + 1
                    else "shared_attention"
                )
            else:
                layer_block_type = "mamba"
            layer_block_types.append(layer_block_type)
        return layer_block_types
