
import paddle

############################## 相关utils函数，如下 ##############################
############################ PaConvert 自动生成的代码 ###########################

def _Tensor_max(self, *args, **kwargs):
    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(self, *args, **kwargs)
    elif len(args) == 1 and isinstance(args[0], paddle.Tensor):
        ret = paddle.maximum(self, *args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 1:
            ret = paddle.max(self, *args, **kwargs), paddle.argmax(self, *args, **kwargs)
        else:
            ret = paddle.max(self, *args, **kwargs)

    return ret

setattr(paddle.Tensor, "_max", _Tensor_max)

from typing import Optional

import paddleformers


def _convert_head_mask_to_5d(head_mask, num_hidden_layers):
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
    assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
    head_mask = head_mask.to(dtype=paddle.get_default_dtype())  # switch to float if need + fp16 compatibility
    return head_mask

def _get_head_mask(
    self,
    head_mask: Optional[paddle.Tensor],
    num_hidden_layers: int,
    is_attention_chunked: bool = False,
):
    if head_mask is not None:
        head_mask = _convert_head_mask_to_5d(head_mask, num_hidden_layers)
        if is_attention_chunked is True:
            head_mask = head_mask.unsqueeze(-1)
    else:
        head_mask = [None] * num_hidden_layers
    return head_mask
setattr(paddleformers.transformers.model_utils.PretrainedModel, "get_head_mask", _get_head_mask)

original_generate = paddleformers.generation.utils.GenerationMixin.generate
def _generate(self, input_ids, *args, **kwargs):
    return paddle.concat((input_ids, original_generate(self,input_ids, *args, **kwargs)[0]), axis=-1)
setattr(paddleformers.generation.utils.GenerationMixin, "generate", _generate)

setattr(paddleformers.transformers.model_utils.PretrainedModel, "device", None)

def _post_init(self):
    if hasattr(self, "init_weights"):
        self.init_weights()
    elif hasattr(self, "_init_weights"):
        self._init_weights()
setattr(paddleformers.transformers.model_utils.PretrainedModel, "post_init", _post_init)
############################## 相关utils函数，如上 ##############################

