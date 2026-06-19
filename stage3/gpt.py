"""
阶段③ / 核心：从零手写一个 GPT（生成式预训练 Transformer）。

这个文件是整个阶段的灵魂。我们不调用任何现成的 Transformer 模块，
自己一层层把 GPT 搭出来，重点讲透**自注意力 (self-attention)**——
它是 GPT、以及所有现代大语言模型的核心机制。

GPT 干的事其实只有一件：给定前面的字符，预测下一个字符。
把这件事做好、再反复"预测→接上→再预测"，就能生成文本。

阅读顺序建议：先看最下面的 GPT 类总览，再回头细看 Head（注意力）。
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ----------------------------------------------------------------------------
# 超参数配置（一个简单的容器）
# ----------------------------------------------------------------------------
class GPTConfig:
    def __init__(self, vocab_size, block_size=256, n_layer=6, n_head=6,
                 n_embd=384, dropout=0.2):
        self.vocab_size = vocab_size   # 词表大小（不同字符数，这里 65）
        self.block_size = block_size   # 上下文长度：一次最多看多少个字符
        self.n_layer = n_layer         # Transformer block 堆几层
        self.n_head = n_head           # 多头注意力的"头"数
        self.n_embd = n_embd           # 每个字符向量的维度
        self.dropout = dropout


# ----------------------------------------------------------------------------
# 单个注意力头 —— 整个 GPT 最核心的地方
# ----------------------------------------------------------------------------
# 直觉：句子里每个字，想知道"我该参考前面哪些字来决定下一个字"。
# 注意力就是让每个位置去"看"前面所有位置，按相关程度加权汇总它们的信息。
#
# 三个角色（每个都是对输入做一次线性变换得到的向量）：
#   Query (查询)：我在找什么样的信息？
#   Key   (键)  ：我能提供什么样的信息？
#   Value (值)  ：我实际携带的信息内容。
# 某位置的 Query 和各位置的 Key 做点积 -> 得到"我跟谁最相关"的分数 ->
# 用这个分数对各位置的 Value 加权求和 -> 就是该位置"参考别人后"的新表示。
class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        # 三个线性层，把 n_embd 维输入投影成 head_size 维的 Q/K/V
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # 因果 mask：一个下三角矩阵。保证位置 t 只能看到 0..t（看不到未来）。
        # register_buffer：它是固定常量、不是要训练的参数，但要跟着模型走（如搬到GPU）。
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape          # 批量, 时间步(字符数), 通道(n_embd)
        k = self.key(x)            # (B, T, head_size)
        q = self.query(x)          # (B, T, head_size)

        # 1) 算注意力分数：每个位置的 Query 和每个位置的 Key 做点积
        #    除以 sqrt(head_size) 做缩放，防止分数过大导致 softmax 梯度消失。
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)

        # 2) 因果 mask：把"未来位置"的分数设成 -inf，softmax 后就变 0（看不到）。
        #    这是 GPT 能"自回归生成"的关键——预测下一个字时绝不能偷看答案。
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # 3) softmax 把分数变成"权重"（每行加起来=1，表示注意力如何分配）。
        wei = F.softmax(wei, dim=-1)   # (B, T, T)
        wei = self.dropout(wei)

        # 4) 用权重对 Value 加权求和 -> 每个位置"参考前文后"的新表示。
        v = self.value(x)          # (B, T, head_size)
        out = wei @ v              # (B, T, head_size)
        return out


# ----------------------------------------------------------------------------
# 多头注意力：并行跑多个 Head，让模型从不同"角度"关注信息
# ----------------------------------------------------------------------------
# 一个头可能学会关注"主谓关系"，另一个头关注"上一个标点"……多头各看各的，
# 再把结果拼起来，表达能力更强。
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head   # 每个头分到的维度
        self.heads = nn.ModuleList(
            [Head(config, head_size) for _ in range(config.n_head)]
        )
        self.proj = nn.Linear(config.n_embd, config.n_embd)  # 拼接后再投影回原维度
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # 拼接所有头 (B,T,n_embd)
        out = self.dropout(self.proj(out))
        return out


# ----------------------------------------------------------------------------
# 前馈网络：注意力之后，让每个位置各自再"想一想"（做非线性变换）
# ----------------------------------------------------------------------------
# 注意力负责"在位置之间交换信息"，前馈网络负责"在单个位置内部加工信息"。
# 标准做法：先放大 4 倍维度过 ReLU，再压回去。
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# Transformer Block：多头注意力 + 前馈，配残差连接和 LayerNorm
# ----------------------------------------------------------------------------
# 两个关键技巧（让深层网络训得动）：
#   · 残差连接 (x + ...)：给梯度一条"高速公路"，避免深层网络梯度消失。
#   · LayerNorm：对每个位置的向量做归一化，稳定训练。
#     这里用 Pre-LN：先归一化再进子层（现代 GPT 的标准做法，更稳）。
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # 注意力子层 + 残差
        x = x + self.ffwd(self.ln2(x))   # 前馈子层 + 残差
        return x


# ----------------------------------------------------------------------------
# GPT 总装：embedding -> 多个 Block -> 输出层
# ----------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 词嵌入：每个字符整数 -> 一个 n_embd 维向量（表示它的"含义"）
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置嵌入：每个位置(0..block_size-1) -> 一个向量（告诉模型"第几个字"）
        # 因为注意力本身不区分顺序，必须显式注入位置信息。
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)            # 最后一层归一化
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)  # 映射到词表，得到每个字符的分数

    def forward(self, idx, targets=None):
        # idx: (B, T) 一批字符整数序列
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                       # (B,T,n_embd)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)                    # (T,n_embd)
        x = tok_emb + pos_emb                                     # 含义 + 位置
        x = self.blocks(x)                                        # 过所有 Transformer block
        x = self.ln_f(x)
        logits = self.lm_head(x)                                  # (B,T,vocab_size) 预测下一个字符的分数

        loss = None
        if targets is not None:
            # 交叉熵：把 (B,T,vocab) 摊平成 (B*T, vocab) 和 (B*T,) 对齐
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """自回归生成：给定起始序列 idx (B,T)，逐字往后生成 max_new_tokens 个。"""
        for _ in range(max_new_tokens):
            # 只取最后 block_size 个字符作为上下文（模型一次最多看这么长）
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)                # (B,T,vocab)
            logits = logits[:, -1, :] / temperature   # 只要最后一个位置的预测；温度调随机性
            probs = F.softmax(logits, dim=-1)         # 变成概率分布
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率采样下一个字符
            idx = torch.cat([idx, idx_next], dim=1)   # 接到末尾，继续下一轮
        return idx
