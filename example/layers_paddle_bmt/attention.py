from typing import Optional
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import bmtrain_paddle as bmt
from bmtrain_paddle.nn import (
    Linear,
    ColumnParallelLinear,
    RowParallelLinear,
)
import math
from bmtrain_paddle.global_var import config
from bmtrain_paddle.distributed import all_gather 

class Attention(bmt.DistributedModule):
    def __init__(self, 
                 dim_model: int, 
                 dim_head: int, 
                 num_heads: int, 
                 bias: bool = True, 
                 dtype=None):
        super().__init__()

        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_model = dim_model

        # print("dtype:", dtype, dtype.type())
        # datatype = "float16" if dtype == float16 else "float32"
        if config['tp_size'] > 1:
            self.project_q = ColumnParallelLinear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype, gather_input=False)
            self.project_k = ColumnParallelLinear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype, gather_input=False)
            self.project_v = ColumnParallelLinear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype, gather_input=False)
            self.project_out = RowParallelLinear(dim_head * num_heads, dim_model, bias=bias, dtype=dtype)
        else:
            self.project_q = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
            self.project_k = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
            self.project_v = Linear(dim_model, dim_head * num_heads, bias=bias, dtype=dtype)
            self.project_out = Linear(dim_head * num_heads, dim_model, bias=bias, dtype=dtype)

        self.softmax = paddle.nn.Softmax(axis=-1)

    def forward(self, 
                hidden_q: paddle.Tensor,        # (batch_size, seq_q, dim_model)
                hidden_kv: paddle.Tensor,       # (batch_size, seq_kv, dim_model)
                mask: paddle.Tensor,            # (batch_size, seq_q, seq_kv)
                position_bias: Optional[paddle.Tensor] = None  # (batch, num_heads, seq_q, seq_kv)
                ) -> paddle.Tensor:
        batch_size = hidden_q.shape[0]

        # Ensure hidden_q and hidden_kv share the same memory (if needed)
        assert hidden_q.data_ptr() == hidden_kv.data_ptr()

        # print("-------self.project_q.weight, self.project_k.weight, self.project_v.weight",
        #     self.project_q.weight.shape, self.project_k.weight.shape, self.project_v.weight.shape)
        # print("-------self.project_q.bias, self.project_k.bias, self.project_v.bias",
        #       self.project_q.bias.shape, self.project_k.bias.shape, self.project_v.bias.shape)
        if config['tp_size'] > 1:
            # print("!!!!!!!!!!!!!!!!!!!!!!!!hidden_q shape", hidden_q.shape, hidden_q,
            #       paddle.concat([self.project_q.weight, self.project_k.weight, self.project_v.weight], axis=-1).shape)
            # print("!!!!!!!!!!!!!!!11111111111  hidden_q weight", self.project_q.weight,
            #       self.project_k.weight, self.project_v.weight)
            # print("!!!!!!!!!!!!!!!2222222222  hidden_q weight", self.project_q.bias,
            #       self.project_k.bias, self.project_v.bias)
            hidden_q = bmt.nn.OpParallelLinear.apply(
                hidden_q,
                paddle.concat([self.project_q.weight, self.project_k.weight, self.project_v.weight], axis=-1),
                paddle.concat([self.project_q.bias, self.project_k.bias, self.project_v.bias], axis=-1),
                True, False,
                False, None
            )
            # print("@@@@@@@@@@@@@@@@@@@@@@@@hidden_q shape", hidden_q.shape, hidden_q)
            hidden_q = hidden_q.reshape([batch_size, -1, hidden_q.shape[-1]])
            h_q, h_k, h_v = hidden_q.chunk(3, axis=-1)
        else:
            h_q : paddle.Tensor = self.project_q(hidden_q)
            h_k : paddle.Tensor = self.project_k(hidden_kv)
            h_v : paddle.Tensor = self.project_v(hidden_kv)

        # print("&&&&&&&&&&&&&&&  attention ", h_q.shape, h_k.shape, h_v.shape )
        # print("&&&&&&&&&&&&&&&  attention h_q, h_k, h_v ", h_q, h_k, h_v )
        seq_q = h_q.shape[1]
        seq_kv = h_k.shape[1]

        # Reshape for multi-head attention
        h_q = h_q.reshape([batch_size, seq_q, -1, self.dim_head])
        h_k = h_k.reshape([batch_size, seq_kv, -1, self.dim_head])
        h_v = h_v.reshape([batch_size, seq_kv, -1, self.dim_head])

        # Transpose for attention computation
        h_q = h_q.transpose([0, 2, 1, 3])  # (batch_size, num_heads, seq_q, dim_head)
        h_k = h_k.transpose([0, 2, 1, 3])  # (batch_size, num_heads, seq_kv, dim_head)
        h_v = h_v.transpose([0, 2, 1, 3])  # (batch_size, num_heads, seq_kv, dim_head)

        # Flatten batch and head dimensions
        h_q = h_q.reshape([-1, seq_q, self.dim_head])
        h_k = h_k.reshape([-1, seq_kv, self.dim_head])
        h_v = h_v.reshape([-1, seq_kv, self.dim_head])

        # Compute attention scores
        score = paddle.bmm(h_q, h_k.transpose([0, 2, 1]))  # (batch_size * num_heads, seq_q, seq_kv)
        score = score / (self.dim_head ** 0.5)

        # Reshape for applying mask and position_bias
        score = score.reshape([batch_size, -1, seq_q, seq_kv])

        if position_bias is not None:
            score = score + position_bias

        # Apply mask
        score = paddle.where(mask.unsqueeze(1), score, paddle.full_like(score, float('-inf')))

        # Compute softmax
        score = self.softmax(score)

        # Apply mask again to zero out masked positions
        score = paddle.where(mask.unsqueeze(1), score, paddle.zeros_like(score))

        # Flatten batch and head dimensions again
        score = score.reshape([-1, seq_q, seq_kv])
        # print("!!!!!!!!!!paddle.bmm(score, h_v)", score.shape, h_v.shape, score, h_v)
        # Compute attention output
        h_out = paddle.bmm(score, h_v)  # (batch_size * num_heads, seq_q, dim_head)

        # print("!!!!!!!!!!!!!!!! 1 self.project_out", h_out)
        h_out = h_out.reshape([batch_size, -1, seq_q, self.dim_head])
        # print("!!!!!!!!!!!!!!!! 2 self.project_out", h_out)
        h_out = h_out.transpose([0, 2, 1, 3])  # (batch_size, seq_q, num_heads, dim_head)
        # print("!!!!!!!!!!!!!!!! 3 self.project_out", h_out)
        h_out = h_out.reshape([batch_size, seq_q, -1])
        # print("!!!!!!!!!!!!!!!! 4 before self.project_out", h_out)
        if config['tp_size'] > 1:
            h_out = h_out.reshape([batch_size * config['tp_size'], -1, h_out.shape[-1]])
        # print("!!!!!!!!!!!!!!!! 5 before self.project_out", h_out)
        attn_out = self.project_out(h_out)
        # print("!!!!!!!!!!!!!!!! 6 after self.project_out", attn_out)
        return attn_out