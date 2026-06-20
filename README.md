# 从零学机器学习：从一个神经元到 RAG 助手

一条**由浅入深、亲手实现**的机器学习/深度学习学习路径。不调包速成，每个阶段都从原理出发、
重注释、可运行、可复现——从用 numpy 手推反向传播，一路走到从零手写 GPT、微调大模型、再到挂 RAG。

> 全程在单卡 RTX 5090 上完成，代码以中文注释为主，适合中文学习者跟读。

## 学习路径总览

| 阶段 | 主题 | 核心收获 | 实测结果 |
|------|------|---------|---------|
| **①** [XOR 从零训练](stage1/) | numpy 手写神经网络 | 前向 → loss → 反向传播 → 梯度下降；手推梯度 + 数值检查 | XOR 完美分类，梯度误差 ~1e-9 |
| **②** [MNIST 分类](stage2/) | MLP vs CNN | mini-batch、训练/验证/测试集、过拟合、卷积 | MLP 97.5% / CNN 99.1% |
| **③** [从零写 GPT](stage3/) | 手写 Transformer | 自注意力、多头、自回归生成、tokenize、温度 | 1080万参数，生成莎士比亚 |
| **④** [LoRA 微调](stage4/) | 微调 Qwen2.5-0.5B | LoRA、chat模板、prompt masking、微调 vs RAG、量化 | 只训 0.22% 参数，注入领域风格 |
| **⑤** [RAG 问答](stage5/) | 检索增强生成 | 嵌入、向量检索、切块、可溯源问答 | 9423块索引，答案基于真实原文+出处 |

每个阶段目录下都有自己的 `README.md` 详解，设计文档在 [`docs/superpowers/specs/`](docs/superpowers/specs/)。

## 一条贯穿始终的主线

最底层的训练循环**从头到尾没变过**：

```
前向 → 算 loss → 反向求梯度 → 更新参数
```

从阶段① numpy 手写的 XOR，到阶段③的 GPT，到阶段④的微调，做的都是这件事。
变的只有**模型结构**（全连接 → 卷积 → Transformer）和**起点**（从零训 vs 站在预训练模型肩上）。
理解了这一点，所有"大模型"都不再神秘。

## 微调 vs RAG（阶段④⑤的关键对比）

用同一个目标——"让模型更懂某个领域知识库"——同时实践了两条路：

| | 微调 (阶段④) | RAG (阶段⑤) |
|---|---|---|
| 擅长 | 改变模型的**风格/语气/格式** | 基于文档**准确回答 + 给出处** |
| 事实可靠性 | 弱（会编造细节） | 强（答案锚定真实原文） |
| 知识更新 | 要重新训练 | 重建索引即可 |
| 结论 | 学"怎么说" | 学"说得对" |

## 环境与运行

```bash
# 依赖（按需）
pip install torch torchvision numpy transformers peft datasets modelscope matplotlib

# 各阶段独立运行，例如：
python stage1/01_xor_numpy.py
python stage2/02_mnist_cnn.py
python stage3/train.py && python stage3/generate.py
python stage4/finetune_instruct.py && python stage4/chat_instruct.py
python stage5/build_index.py && python stage5/rag.py --q "你的问题"
```

- Python 3.13 + PyTorch (CUDA)。阶段①纯 CPU 即可；②③④⑤建议 GPU。
- 阶段④⑤的基座/嵌入模型从 ModelScope 下载（国内稳定）。
- 数据集、模型权重、向量索引等大文件均不入库（见 `.gitignore`），可按脚本重新生成。

## 技术栈

NumPy · PyTorch · Hugging Face Transformers · PEFT (LoRA) · Datasets · ModelScope · BGE 嵌入模型 · Qwen2.5

---

这是一个学习项目，欢迎参考。如果它帮你理清了"训练到底在干什么"，那就达到目的了。🚀
