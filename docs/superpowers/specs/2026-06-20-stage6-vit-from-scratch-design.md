# 阶段⑥ 设计：从零手写 ViT，并用吴恩达方法论分析

> 学习路径第 6 阶段。多模态里"视觉"那一半的核心：用 Transformer 处理图像。
> 是阶段②(图像/分类) + 阶段③(Transformer) 的合体，并套用吴恩达《ML Yearning》的诊断框架。

## 核心洞见

ViT = 把阶段③手写的 Transformer 用到图像上。区别只有两点：
① 输入是"图像 patch token"而非文字 token；② 注意力不加因果 mask（看图前后都能看）。

## 决策（已确认）

- 数据集：CIFAR-10（彩色，更接近真实，能体现 ViT 特点与短板）。
- 带 CNN 基线做对比。
- 分析部分套用吴恩达方法论（偏差/方差、人类水平对标、误差分析）。

## 产出物

```
stage6/
├── vit.py            从零手写 ViT（patch嵌入、CLS、位置嵌入、Transformer编码器、分类头）
├── cnn_baseline.py   小 CNN 基线（同数据集，用于对比）
├── train_vit.py      CIFAR-10 加载(含增强) + 训练 + 评估 + 吴恩达式诊断
└── README.md         ViT 原理 + ViT vs CNN「数据饥渴」一课 + bias/variance 分析
```

## 数据

- torchvision 自动下载 CIFAR-10（10 类彩色 32×32，5万训练+1万测试）。
- 数据增强：RandomCrop(32,padding=4) + RandomHorizontalFlip + Normalize。ViT 小数据上很依赖增强。
- 切出验证集用于偏差/方差诊断。

## ViT 结构（从零手写，复用阶段③积木）

- **Patch 嵌入**：32×32 切成 4×4 patch（8×8=64 个），每 patch 投影成 n_embd 向量（用 Conv2d 等价实现）。
- **[CLS] token**：可学习的汇总 token，拼到序列最前。
- **位置嵌入**：可学习的位置向量（含 CLS 共 65 个位置）。
- **Transformer 编码器**：~6 层（多头注意力 + FFN + 残差 + Pre-LN），注意力**无因果 mask**。
- **分类头**：取 CLS 经 LayerNorm + Linear → 10 类。
- 规模：n_embd≈192, heads=6, depth=6, 约几百万参数。

## CNN 基线

- 3 段 Conv+ReLU+Pool + 全连接头，规模与 ViT 相当，用于诚实对比。

## 训练 & 诚实预期

- AdamW + 余弦/恒定 lr + 数据增强，几分钟~十几分钟（RTX 5090）。
- 预期：从零小 ViT ~70-78%，CNN 基线 ~82-88%。**对比是教学重点**。

## 吴恩达式分析（写进 README + 训练脚本打印）

1. 列 ViT/CNN 各自的 训练误差、验证误差 → 算偏差/方差。
2. 与 CIFAR-10 人类水平(~94%) 对标 → 瓶颈是可避免偏差还是方差。
3. 误差分析：统计 ViT 主要把哪些类互相混淆（归类，不只打印）。
4. 结论：用 bias/variance + 数据量解释"ViT 为何数据饥渴、为何需要预训练"。

## 成功标准

- 三脚本可跑；ViT 训到合理准确率；CNN 基线更高。
- 输出偏差/方差诊断与混淆统计。
- README 讲清 ViT 原理 + ViT vs CNN + 吴恩达式结论。

## 范围之外（YAGNI）

- 不追求 SOTA 准确率（重点是理解机制与方法论）。
- 不做预训练/大数据（恰恰要展示"没有预训练"的 ViT 短板）。
- 不做多模态融合（本阶段只到"视觉 Transformer"，多模态融合留作后续）。
