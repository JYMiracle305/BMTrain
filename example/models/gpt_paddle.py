import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute
import bmtrain_paddle as bmt
from bmtrain_paddle.global_var import config
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
            self.word_emb = bmt.nn.VPEmbedding(vocab_size, dim_model, dtype=dtype)
        else:
            self.word_emb = Embedding(vocab_size, dim_model, dtype=dtype)
        self.pos_emb = Embedding(max_distance, dim_model, dtype=dtype)
        # Transformer 层
        self.transformers = paddle.nn.LayerList([
            TransformerEncoder(dim_model, dim_head, num_heads, dim_ff, bias, dtype)
            for _ in range(num_layers)
        ])
        # LayerNorm
        self.layernorm = Layernorm(dim_model, dtype=dtype)
        self.projection_layer = nn.Linear(dim_model, vocab_size)
        self.projection_layer.weight = self.word_emb.weight

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

        # embeding层
        out = self.pos_emb(pos) + self.word_emb(input)

        # Transformer 层
        for i, layer in enumerate(self.transformers):
            out = layer(out, mask_2d, None)
            # print(f"After Transformer layer {i} shape:", out.shape)
        
        # LayerNorm
        out = self.layernorm(out)
        try:
            logits = self.word_emb(out, projection=True)
            # print("Logits shape:", logits.shape)
        except Exception as e:
            print("Error in word embedding projection:", e)
            print("Shape of out before projection:", out.shape)
        # 词嵌入投影
        logits = self.word_emb(out, projection=True)
        # print("after self.word_emb")

        # 记录 logits（用于调试或检查）
        # print("Logits:", logits)
        # bmt.inspect.record_tensor(logits, "logits")
        return logits
