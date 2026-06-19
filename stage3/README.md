# 阶段③ — 从零手写一个 GPT，生成莎士比亚

四阶段学习路径第 3 步。质的飞跃：第一次接触 Transformer / 大语言模型的核心机制。
我们不调用任何现成 Transformer 模块，自己一层层把 GPT 搭出来。

## 怎么跑

```bash
python stage3/train.py        # 训练（约几分钟，自动下载语料、存模型权重）
python stage3/generate.py     # 用训练好的模型生成文本
python stage3/generate.py --prompt "ROMEO:" --tokens 1000 --temp 0.8
```

## 实测结果

训练 5000 步后验证 loss ≈ 1.47（字符级莎士比亚的合理水平）。生成示例（起始 "First Citizen:"）：

```
First Citizen:
Your own prompt is enough; and therefore is burnened,
no world to be accused; let him go: ...

Second Murderer:
No, by the news abroad?

MENENIUS:
Be you, let me speak as you.
```

从零训练几分钟，模型就学会了：**人物名、对白格式（名字+冒号+台词）、剧本结构、古英语腔调**，
单词大体成形。它不是在背原文，而是学到了"莎士比亚的写法"。

## 核心原理

### GPT 只做一件事
**看前面的字符，预测下一个字符。** 把这件事做好，再反复"预测→接上→再预测"，
就能源源不断地生成文本——这叫**自回归生成 (autoregressive generation)**。

### tokenize（字符级）
把文本里 65 个不同字符各映射成一个整数。"hi" → [20, 21]。最简单的分词方式。
（真实大模型用更高级的 BPE 子词分词，但原理一样：文字 ↔ 整数。）

### 注意力机制（self-attention）—— 本阶段灵魂，见 gpt.py 的 `Head`
让每个位置去"看"前面所有位置，按相关程度加权汇总信息。三个角色：
- **Query（查询）**：我在找什么信息？
- **Key（键）**：我能提供什么信息？
- **Value（值）**：我实际携带的内容。

某位置的 Query 与各位置的 Key 做点积 → 得到"我跟谁最相关"的分数 →
softmax 变成权重 → 对各位置的 Value 加权求和 → 该位置"参考前文后"的新表示。

**因果 mask**：预测下一个字时绝不能偷看未来的字，所以把未来位置的注意力分数设成 -∞。
这是 GPT 能自回归生成的关键。

**多头**：并行跑多个注意力，各自从不同角度关注（一个看语法、一个看标点……），再拼起来。

### Transformer Block
`多头注意力 + 前馈网络`，每个子层都配：
- **残差连接** `x + ...`：给梯度一条高速公路，让深层网络训得动。
- **LayerNorm**：归一化，稳定训练。

堆 6 层这样的 block，就是我们的 GPT（约 1079 万参数）。

### 温度 temperature（生成时调随机性）
- **低温 (0.3)**：保守稳妥，但易重复（"the state of the state"）。
- **高温 (1.4)**：大胆有创意，但易混乱、生造词（"Slympal Trutle"）。
- **0.8 左右**：常见的平衡点。

### 保存 / 加载模型（补上阶段②留的一课）
训练几分钟成本不低，所以训完用 `torch.save` 把权重+字符映射存成 `.pt` 文件；
`generate.py` 用 `torch.load`（带 `weights_only=True` 更安全）加载，随时生成、不必重训。

## 和前两阶段的联系
最底层那套循环——**前向 → 算 loss → 反向 → 更新**——和阶段① numpy 手写的完全一样。
损失还是**交叉熵**（阶段②见过）。结尾训练 loss 远低于验证 loss，又是**过拟合**（阶段②学过）。
变的只是模型结构：从全连接 → 卷积 → 现在的 Transformer。

## 文件
```
gpt.py        从零手写的 GPT（注意力、多头、block、生成）
train.py      tokenize + 训练 + 存盘
generate.py   加载 + 自回归生成
input.txt     语料（自动下载，不进 git）
gpt_shakespeare.pt  训练好的权重（不进 git，可重新训练）
```

## 下一步
阶段④：微调 (fine-tune) 一个现成的开源模型——不再从零训练，
而是站在别人训好的大模型肩膀上，在自己的数据/任务上做调整。
