# Stage4 指令微调版 设计：让模型"会答题"

> 阶段④的延伸。续写式微调（已完成）让模型学到风格但不会规整答题；
> 本次改用**指令微调**，让它用问答方式回应。仍是学习目的，事实可靠性仍需 RAG。

## 决策（已确认）

- 问答数据来源：**规则法从 wiki 结构自动造**（标题/小节→问，正文→答，答案是真实 wiki 文本）。
- 基座：**复用现有 Qwen2.5-0.5B 基座**（不再下载），用 chat 模板训练让其学会答题。
- 方法：LoRA。

## 产出物

```
stage4/
├── build_qa.py            从 wiki 造问答对 -> qa.jsonl
├── finetune_instruct.py   chat 模板 + prompt masking + LoRA 训练 -> lora_adapter_instruct/
└── chat_instruct.py       交互式问答（chat 模板）
```

## 造问答数据（build_qa.py）

遍历每篇 md（去 frontmatter）：
- 提取标题（首个 `# ` 行，否则用文件名）与正文。
- 按 `## ` 小节切分。
- 生成问答对：
  - 整篇：多套问法（"什么是X？""请介绍一下X""X是指什么"）→ 摘要/正文（截断）。
  - 小节：（"在「X」中，{小节}是什么？"等）→ 小节内容。
- 过滤：答案长度 40~1200 字符；问题合理。
- 输出 `qa.jsonl`，每行 `{"question":..., "answer":...}`。

## 训练（finetune_instruct.py）

- **Chat 模板**：`<|im_start|>user\n{问}<|im_end|>\n<|im_start|>assistant\n{答}<|im_end|>`。
- **Prompt masking**：问题部分的 label 设为 -100（交叉熵忽略），**只对答案 token 算 loss**。
  这是指令微调与续写微调的本质区别——模型学"给定问题生成答案"，而非续写全文。
- LoRA（r=8, alpha=16, dropout=0.05, target q/k/v/o），复用现有基座，bf16，GPU。
- 适配器存到 `lora_adapter_instruct/`（与续写版 `lora_adapter/` 分开）。

## 成功标准

- 训练 loss 下降。
- chat_instruct.py 提问，模型用**答题方式**回应（不再续写成选择题），带数据治理内容。
- 诚实预期：像助手了，但 0.5B 事实仍不保证（RAG 才解决事实）。

## 范围之外

- 不调用外部大模型造数据（纯规则）。
- 不追求事实准确（RAG 职责）。
- 不做多轮对话（单轮问答）。
