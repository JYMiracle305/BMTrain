import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed.fleet.utils import recompute
import bmtrain_paddle as bmt
from bmtrain_paddle.global_var import config
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

        # 词嵌入和位置嵌入
        if config["tp_size"] > 1:
            self.word_emb = Embedding(vocab_size, dim_model, dtype=dtype)
            self.pos_emb = Embedding(max_distance, dim_model, dtype=dtype)
        else:
            print(f"vocab_size:{vocab_size}, dim_model:{dim_model}")
            self.word_emb = Embedding(vocab_size, dim_model, dtype=dtype)
            self.pos_emb = Embedding(max_distance, dim_model, dtype=dtype)
            print("词嵌入维度:", self.word_emb.weight.shape)  # 应为 [vocab_size//tp_size, dim_model]
            print("位置嵌入维度:", self.pos_emb.weight.shape)  # 应为 [max_distance, dim_model]
        # Transformer 层
        # if config['pipe_size'] > 1:
        #     self.transformers = bmt.PipelineTransformerBlockList([
        #         bmt.Block(
        #             TransformerEncoder(
        #                 dim_model, dim_head, num_heads, dim_ff, bias, dtype
        #             )
        #             , mode="PIPE"
        #         )
        #         for _ in range(num_layers)
        #     ])
        # else:
        #     self.transformers = bmt.TransformerBlockList([
        #         bmt.Block(
        #             TransformerEncoder(
        #                 dim_model, dim_head, num_heads, dim_ff, bias, dtype
        #             )
        #         )
        #         for _ in range(num_layers)
        #     ])
        self.transformers = nn.LayerList([
            TransformerEncoder(dim_model, dim_head, num_heads, dim_ff, bias, dtype)
            for _ in range(num_layers)
        ])

        # LayerNorm
        self.layernorm = Layernorm(dim_model, dtype=dtype)

    def forward(self,
                input: paddle.Tensor,   # (batch, seq_len)
                pos: paddle.Tensor,     # (batch, seq_len)
                mask: paddle.Tensor     # (batch, seq_len)
                ) -> paddle.Tensor:

        # 创建二维掩码
        mask_2d = paddle.unsqueeze(mask, axis=1) & paddle.unsqueeze(mask, axis=2)  # (batch, seq_len, seq_len)
        mask_2d = mask_2d & (paddle.unsqueeze(pos, axis=1) >= paddle.unsqueeze(pos, axis=2))

        # 如果使用模型并行，分割输入
        if config["tp_size"] > 1:
            input = paddle.split(input, num_or_sections=self.tp_size, axis=1)[config["tp_rank"]]
            pos = paddle.split(pos, num_or_sections=self.tp_size, axis=1)[config["tp_rank"]]

        # print("before  out = self.pos_emb(pos) + self.word_emb(input)")
        # 嵌入层
        out = self.pos_emb(pos) + self.word_emb(input)
        print("After adding position and word embeddings shape:", out.shape)

        # Transformer 层
        for i, layer in enumerate(self.transformers):
            out = layer(out, mask_2d, None)
            print(f"After Transformer layer {i} shape:", out.shape)
        # out = self.transformers(out, mask_2d, None)

        # LayerNorm
        out = self.layernorm(out)
        print("After LayerNorm shape:", out.shape)

        try:
            logits = self.word_emb(out, projection=True)
            print("Logits shape:", logits.shape)
        except Exception as e:
            print("Error in word embedding projection:", e)
            print("Shape of out before projection:", out.shape)
        # 词嵌入投影
        # logits = self.word_emb(out, projection=True)   #out维度和weight不匹配
        # logits = self.projection_layer(out) 
        # print("after self.word_emb")

        # 记录 logits（用于调试或检查）
        # print("Logits:", logits)
        # bmt.inspect.record_tensor(logits, "logits")
        return logits
