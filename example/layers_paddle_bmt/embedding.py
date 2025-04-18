import math
from typing import Optional
import paddle
import paddle.nn.functional as F
import bmtrain_paddle as bmt

class Embedding(bmt.DistributedModule):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, 
                 norm_type: float = 2., 
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False, 
                 _weight: Optional[paddle.Tensor] = None,
                 dtype=None):
        super().__init__()
        print("embeddings init", num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        # if _weight is None:
        #     self.weight = bmt.DistributedParameter(paddle.empty([num_embeddings, embedding_dim], dtype=dtype).cuda())
        # else:
        #     self.weight = bmt.DistributedParameter(_weight)
        if _weight is None:
            print("if _weight is None:", num_embeddings, embedding_dim)
            self.weight = self.create_parameter(
                shape=[num_embeddings, embedding_dim],
                dtype=dtype if dtype else paddle.get_default_dtype(),
                default_initializer=paddle.nn.initializer.XavierNormal()
            )
        else:
            self.weight = self.create_parameter(
                shape=_weight.shape,
                dtype=_weight.dtype,
                default_initializer=paddle.nn.initializer.Assign(_weight)
            )
        print("-------------Embedding-----------", self.weight.shape)
        self.sparse = sparse
        print("sparse-----------", sparse)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        assert embeddings.ndim == 2, 'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        print("embeddings from_pretrained", rows, cols)
        embedding = cls(
            num_embeddings=rows,
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse
        )
        embedding.weight.stop_gradient = freeze
        return embedding

    def forward(self, input: paddle.Tensor, projection: bool = False) -> paddle.Tensor:
        if not projection:
            # print("F.embedding-------------", input.shape, self.weight.shape)
            # print(paddle.max(input))  # 检查输入的最大值
            # print(paddle.min(input)) 
            # out = F.embedding(
            #     input, self.weight, self.padding_idx, self.max_norm,
            #     self.norm_type, self.scale_grad_by_freq, self.sparse
            # )
            print(f"not projection Embedding input:{input.shape}, self.weight:{self.weight.shape}")
            out = F.embedding(
                input, self.weight, self.padding_idx, self.sparse
            )
            return out
        else:
            #需要确保 input 的最后一个维度与 self.weight 的第一个维度一致
            print(f"Embedding input:{input.shape}, self.weight:{self.weight.shape}")
            out = F.linear(input, self.weight.T)
            return out

    def extra_repr(self) -> str:
        s = f'{self.num_embeddings}, {self.embedding_dim}'
        if self.padding_idx is not None:
            s += f', padding_idx={self.padding_idx}'
        if self.max_norm is not None:
            s += f', max_norm={self.max_norm}'
        if self.norm_type != 2:
            s += f', norm_type={self.norm_type}'
        if self.scale_grad_by_freq is not False:
            s += f', scale_grad_by_freq={self.scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s
