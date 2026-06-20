"""
阶段⑥ / 核心：从零手写 ViT（Vision Transformer）。

一句话：ViT = 把阶段③那个 Transformer 拿来处理图像。
和阶段③ GPT 的区别只有两点：
  1. 输入不是"文字 token"，而是把图片切成一个个小方块(patch)，每块当一个 token。
  2. 注意力**不加因果 mask**——看图可以前后左右都看（GPT 生成文本才需要"不能看未来"）。

流程：
  图片(3×32×32) → 切成 patch → 每块线性嵌入成向量 → 加 [CLS] token + 位置嵌入
    → N 层 Transformer 编码器 → 取 [CLS] 向量 → 分类头 → 10 类
"""

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# 1. Patch 嵌入：把图片切成小方块，每块变成一个 token 向量
# ----------------------------------------------------------------------------
# 用一个 stride=patch 的卷积一步到位：它等价于"把每个 patch 拉平后做线性变换"。
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch=4, in_ch=3, n_embd=192):
        super().__init__()
        self.n_patches = (img_size // patch) ** 2          # 32/4=8 -> 8×8=64 个 patch
        self.proj = nn.Conv2d(in_ch, n_embd, kernel_size=patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)                  # (B,3,32,32) -> (B,n_embd,8,8)
        x = x.flatten(2).transpose(1, 2)  # -> (B, 64, n_embd)，每个 patch 一个 token
        return x


# ----------------------------------------------------------------------------
# 2. 多头自注意力（和阶段③几乎一样，但没有因果 mask）
# ----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)      # 一次性算出 Q、K、V
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)              # (3, B, n_head, T, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 注意力分数（缩放点积）—— 和阶段③一模一样，唯独不做 masked_fill
        att = (q @ k.transpose(-2, -1)) * self.head_size ** -0.5  # (B,n_head,T,T)
        att = att.softmax(dim=-1)
        att = self.dropout(att)
        out = att @ v                                 # (B,n_head,T,head_size)
        out = out.transpose(1, 2).reshape(B, T, C)    # 拼回各头
        return self.proj(out)


# ----------------------------------------------------------------------------
# 3. Transformer 编码器块（多头注意力 + FFN，配残差 + Pre-LN，同阶段③）
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # 注意力子层 + 残差
        x = x + self.ffwd(self.ln2(x))     # 前馈子层 + 残差
        return x


# ----------------------------------------------------------------------------
# 4. ViT 总装
# ----------------------------------------------------------------------------
class ViT(nn.Module):
    def __init__(self, img_size=32, patch=4, in_ch=3, n_classes=10,
                 n_embd=192, n_head=6, depth=6, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, in_ch, n_embd)
        n_patches = self.patch_embed.n_patches

        # [CLS] token：一个可学习的"汇总 token"，拼在序列最前，最后用它做分类
        # （和 bge 嵌入模型用 [CLS] 代表整句是一个道理）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        # 位置嵌入：注意力本身不分先后，必须显式告诉它每个 patch 在图里的位置
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, n_embd))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout)
                                      for _ in range(depth)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, n_classes)   # 分类头

        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, 64, n_embd)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, n_embd)
        x = torch.cat([cls, x], dim=1)                   # 拼上 CLS -> (B, 65, n_embd)
        x = self.dropout(x + self.pos_embed)             # 加位置嵌入
        x = self.blocks(x)                               # 过 Transformer 编码器
        x = self.ln_f(x)
        cls_out = x[:, 0]                                # 只取 CLS token 的输出
        return self.head(cls_out)                        # -> (B, 10) 类别分数
