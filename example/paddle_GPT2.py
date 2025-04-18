import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class GPT2Embeddings(nn.Layer):
    """GPT-2 嵌入层"""
    def __init__(self, vocab_size, hidden_size, max_position_embeddings):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
    
    def forward(self, input_ids):
        seq_length = input_ids.shape[1]
        position_ids = paddle.arange(seq_length, dtype='int64')
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        embeddings = inputs_embeds + position_embeds
        return embeddings

class GPT2Attention(nn.Layer):
    """GPT-2 的多头自注意力机制"""
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states, attention_mask=None):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # 分头
        query = query.reshape([0, 0, self.num_attention_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key = key.reshape([0, 0, self.num_attention_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value = value.reshape([0, 0, self.num_attention_heads, self.head_dim]).transpose([0, 2, 1, 3])
        
        # 计算注意力分数
        attn_scores = paddle.matmul(query, key, transpose_y=True) * self.scale
        
        # 应用注意力掩码（可选）
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, axis=-1)
        
        # 加权求和
        context = paddle.matmul(attn_weights, value)
        context = context.transpose([0, 2, 1, 3]).reshape([0, 0, -1])
        
        # 投影到输出
        context = self.out_proj(context)
        return context

class GPT2Block(nn.Layer):
    """GPT-2 的 Transformer 块"""
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.attn = GPT2Attention(hidden_size, num_attention_heads)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size)
        )
    
    def forward(self, hidden_states, attention_mask=None):
        # 自注意力
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states + residual
        
        # 前馈网络
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        
        return hidden_states

class GPT2Model(nn.Layer):
    """GPT-2 主体模型"""
    def __init__(self, vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size, max_position_embeddings):
        super().__init__()
        self.embeddings = GPT2Embeddings(vocab_size, hidden_size, max_position_embeddings)
        self.layers = nn.LayerList([
            GPT2Block(hidden_size, num_attention_heads, intermediate_size)
            for _ in range(num_layers)
        ])
    
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        
        return hidden_states

class GPT2LMHeadModel(nn.Layer):
    """GPT-2 语言模型"""
    def __init__(self, vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size, max_position_embeddings):
        super().__init__()
        self.transformer = GPT2Model(
            vocab_size, hidden_size, num_layers, num_attention_heads,
            intermediate_size, max_position_embeddings
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias_attr=False)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask)
        lm_logits = self.lm_head(hidden_states)
        
        if labels is not None:
            # 计算损失
            loss = F.cross_entropy(
                lm_logits.reshape([-1, lm_logits.shape[-1]]),
                labels.reshape([-1]),
                reduction='mean'
            )
            return loss
        return lm_logits