import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed.fleet.utils import recompute

# 假设以下层已经用 PaddlePaddle 实现
from layers_paddle import TransformerEncoder, Layernorm, Embedding

class GPT_paddle(nn.Layer):
    def __init__(self,
                 num_layers: int,
                 vocab_size: int,
                 dim_model: int,
                 dim_head: int,
                 num_heads: int,
                 dim_ff: int,
                 max_distance: int,
                 bias: bool = True,
                 dtype=None):
        super().__init__()

        self.max_distance = max_distance

        # 初始化分布式环境
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        # 词嵌入和位置嵌入
        if self.tp_size > 1:
            self.word_emb = Embedding(vocab_size, dim_model, dtype=dtype)
            self.pos_emb = Embedding(max_distance, dim_model, dtype=dtype)
        else:
            # print(f"vocab_size:{vocab_size}, dim_model:{dim_model}")
            self.word_emb = Embedding(vocab_size, dim_model, dtype=dtype)
            self.pos_emb = Embedding(max_distance, dim_model, dtype=dtype)

        # Transformer 层
        self.transformers = nn.LayerList([
            TransformerEncoder(dim_model, dim_head, num_heads, dim_ff, bias, dtype)
            for _ in range(num_layers)
        ])

        # LayerNorm
        self.layernorm = Layernorm(dim_model, dtype=dtype)
        self.projection_layer = nn.Linear(dim_model, vocab_size)

    def forward(self,
                input: paddle.Tensor,   # (batch, seq_len)
                pos: paddle.Tensor,     # (batch, seq_len)
                mask: paddle.Tensor     # (batch, seq_len)
                ) -> paddle.Tensor:

        # 创建二维掩码
        mask_2d = paddle.unsqueeze(mask, axis=1) & paddle.unsqueeze(mask, axis=2)  # (batch, seq_len, seq_len)
        mask_2d = mask_2d & (paddle.unsqueeze(pos, axis=1) >= paddle.unsqueeze(pos, axis=2))

        # 如果使用模型并行，分割输入
        if self.tp_size > 1:
            input = paddle.split(input, num_or_sections=self.tp_size, axis=1)[self.tp_rank]
            pos = paddle.split(pos, num_or_sections=self.tp_size, axis=1)[self.tp_rank]

        # print("before  out = self.pos_emb(pos) + self.word_emb(input)")
        # 嵌入层
        out = self.pos_emb(pos) + self.word_emb(input)

        # Transformer 层
        for layer in self.transformers:
            out = layer(out, mask_2d, None)
        # out = self.transformers(out, mask_2d, None)

        # LayerNorm
        out = self.layernorm(out)

        # 词嵌入投影
        # logits = self.word_emb(out, projection=True)   #out维度和weight不匹配
        logits = self.projection_layer(out) 
        # print("after self.word_emb")

        # 记录 logits（用于调试或检查）
        # print("Logits:", logits)

        return logits