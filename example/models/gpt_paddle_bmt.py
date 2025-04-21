import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute
import bmtrain_paddle as bmt
from bmtrain_paddle.global_var import config
from layers_paddle_bmt import TransformerEncoder, Layernorm, Embedding

class GPT_paddle_bmt(bmt.DistributedModule):
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
            self.word_emb = bmt.nn.VPEmbedding(vocab_size, dim_model, dtype=dtype)
        else:
            self.word_emb = Embedding(vocab_size, dim_model, dtype=dtype)
        self.pos_emb = Embedding(max_distance, dim_model, dtype=dtype)
        # Transformer 层
        if config['pipe_size'] > 1:
            self.transformers = bmt.PipelineTransformerBlockList([
                bmt.Block(
                    TransformerEncoder(
                        dim_model, dim_head, num_heads, dim_ff, bias, dtype
                    )
                    , mode="PIPE"
                )
                for _ in range(num_layers)
            ])
        else:
            self.transformers = paddle.nn.LayerList([
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
        # print("input", input.shape)
        # 嵌入层
        out = self.pos_emb(pos) + self.word_emb(input)
        # print("After adding position and word embeddings shape:", out.shape)

        # out = self.transformers(out, mask_2d, None)
        for i, layer in enumerate(self.transformers):
            out = layer(out, mask_2d, None)
            # print(f"After Transformer layer {i} shape:", out.shape)
        # print(f"After Transformer layer shape:", out.shape)

        out = self.layernorm(out)
        # print("After LayerNorm shape:", out.shape)

        # # 词嵌入投影
        logits = self.word_emb(out, projection=True)

        return logits
