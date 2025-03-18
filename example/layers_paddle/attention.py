from typing import Optional
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops
from paddle.distributed.fleet.utils import recompute

from paddle.distributed.fleet.layers.mpu import ColumnParallelLinear, RowParallelLinear

class Attention(nn.Layer):
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

        tp_size = fleet.get_hybrid_communicate_group().get_model_parallel_world_size()
        tp_rank = fleet.get_hybrid_communicate_group().get_model_parallel_rank()
        # print("dtype:", dtype, dtype.type())
        # datatype = "float16" if dtype == float16 else "float32"
        if tp_size > 1:
            self.project_q = ColumnParallelLinear(dim_model, dim_head * num_heads, has_bias=bias)
            self.project_k = ColumnParallelLinear(dim_model, dim_head * num_heads, has_bias=bias)
            self.project_v = ColumnParallelLinear(dim_model, dim_head * num_heads, has_bias=bias)
            self.project_out = RowParallelLinear(dim_head * num_heads, dim_model, has_bias=bias)
        else:
            paddle.set_default_dtype(dtype)
            # weight_attr = paddle.ParamAttr(dtype)
            # bias_attr = paddle.ParamAttr(dtype) if bias else False

            # self.project_q = nn.Linear(dim_model, dim_head * num_heads, weight_attr=weight_attr, bias_attr=bias_attr)
            # self.project_k = nn.Linear(dim_model, dim_head * num_heads, weight_attr=weight_attr, bias_attr=bias_attr)
            # self.project_v = nn.Linear(dim_model, dim_head * num_heads, weight_attr=weight_attr, bias_attr=bias_attr)
            # self.project_out = nn.Linear(dim_head * num_heads, dim_model, weight_attr=weight_attr, bias_attr=bias_attr)

            self.project_q = nn.Linear(dim_model, dim_head * num_heads, bias_attr=bias)
            self.project_k = nn.Linear(dim_model, dim_head * num_heads, bias_attr=bias)
            self.project_v = nn.Linear(dim_model, dim_head * num_heads, bias_attr=bias)
            self.project_out = nn.Linear(dim_head * num_heads, dim_model, bias_attr=bias)

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, 
                hidden_q: paddle.Tensor,        # (batch_size, seq_q, dim_model)
                hidden_kv: paddle.Tensor,       # (batch_size, seq_kv, dim_model)
                mask: paddle.Tensor,            # (batch_size, seq_q, seq_kv)
                position_bias: Optional[paddle.Tensor] = None  # (batch, num_heads, seq_q, seq_kv)
                ) -> paddle.Tensor:
        batch_size = hidden_q.shape[0]

        # Ensure hidden_q and hidden_kv share the same memory (if needed)
        assert hidden_q.data_ptr() == hidden_kv.data_ptr()

        tp_size = fleet.get_hybrid_communicate_group().get_model_parallel_world_size()
        tp_rank = fleet.get_hybrid_communicate_group().get_model_parallel_rank()

        if tp_size > 1:
            print("hidden_q shape:", hidden_q.shape)  # 应该是 [2, 256, 2560]
            adjust_linear = nn.Linear(2560, 1280)  # 将特征维度从 2560 调整为 1280
            hidden_q = adjust_linear(hidden_q)
            # 检查 project_q、project_k 和 project_v 的权重形状
            print("project_q.weight shape:", self.project_q.weight.shape)  # 应该是 [2560, 1280]
            print("project_k.weight shape:", self.project_k.weight.shape)  # 应该是 [2560, 1280]
            print("project_v.weight shape:", self.project_v.weight.shape)  # 应该是 [2560, 1280]
            # Concatenate QKV projections and split them later
            qkv_weight = paddle.concat([self.project_q.weight, self.project_k.weight, self.project_v.weight], axis=0)

            print("qkv_weight shape:", qkv_weight.shape)  #[7680, 1280]
            self.project_q.bias = paddle.randn([853])
            self.project_k.bias = paddle.randn([853])
            self.project_v.bias = paddle.randn([853])
            qkv_bias = paddle.concat([self.project_q.bias, self.project_k.bias, self.project_v.bias], axis=0)

            print("qkv_bias shape:", qkv_bias.shape)  #[3840]
            hidden_qkv = mp_ops._linear(hidden_q, qkv_weight, qkv_bias)    # hidden_q.shape[2] == qkv_weight.shape[1]  qkv_weight.shape[0] == qkv_bias.shape[0]  
            hidden_qkv = hidden_qkv.reshape([batch_size, -1, hidden_qkv.shape[-1]])
            h_q, h_k, h_v = paddle.split(hidden_qkv, num_or_sections=3, axis=-1)
        else:
            h_q = self.project_q(hidden_q)
            h_k = self.project_k(hidden_kv)
            h_v = self.project_v(hidden_kv)

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

        # Compute attention output
        h_out = paddle.bmm(score, h_v)  # (batch_size * num_heads, seq_q, dim_head)
        h_out = h_out.reshape([batch_size, -1, seq_q, self.dim_head])
        h_out = h_out.transpose([0, 2, 1, 3])  # (batch_size, seq_q, num_heads, dim_head)
        h_out = h_out.reshape([batch_size, seq_q, -1])

        if tp_size > 1:
            h_out = h_out.reshape([batch_size * tp_size, -1, h_out.shape[-1]])

        attn_out = self.project_out(h_out)

        return attn_out