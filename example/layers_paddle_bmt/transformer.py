from typing import Optional
import paddle
import paddle.nn as nn
import bmtrain_paddle as bmt

from layers_paddle_bmt import Layernorm, Feedforward, Attention

class TransformerEncoder(bmt.DistributedModule):
    def __init__(self,
                 dim_model: int,
                 dim_head: int,
                 num_heads: int,
                 dim_ff: int,
                 bias: bool = True,
                 dtype=None):
        super().__init__()

        # LayerNorm 和 Attention
        self.ln_attn = Layernorm(dim_model, dtype=dtype)
        self.attn = Attention(dim_model, dim_head, num_heads, bias=bias, dtype=dtype)

        # LayerNorm 和 Feedforward
        self.ln_ff = Layernorm(dim_model, dtype=dtype)
        self.ff = Feedforward(dim_model, dim_ff, bias=bias, dtype=dtype)

    def forward(self,
                hidden: paddle.Tensor,  # (batch, seq_len, dim_model)
                mask: paddle.Tensor,    # (batch, seq_len, seq_len)
                position_bias: Optional[paddle.Tensor] = None  # (batch, num_head, seq_len, seq_len)
                ) -> paddle.Tensor:
        # 记录隐藏状态（用于调试或检查）
        # print("Hidden:", hidden)
        # hidden = hidden.astype('float32')
        # print("Hidden:", hidden)
        # Self-Attention
        x = self.ln_attn(hidden)
        x = self.attn(x, x, mask, position_bias)
        hidden = hidden + x
        # print("after transformer Attention shape", hidden.shape)
        # Feedforward
        x = self.ln_ff(hidden)
        # print("after transformer before Feedforward shape", x.shape)
        x = self.ff(x)
        hidden = hidden + x
        # print("after transformer Feedforward shape", hidden.shape)
    
        return hidden