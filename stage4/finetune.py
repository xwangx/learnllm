"""
阶段④ / 微调主脚本：用 LoRA 在数据治理语料上微调 Qwen2.5-0.5B。

和阶段③的对比：
  · 阶段③：从零造一个空白 GPT，从头学语言 —— 训完只会模仿莎士比亚。
  · 阶段④：拿一个已经懂中文、懂世界的预训练模型，只用少量数据"调味"。
    而且不动原模型的 5 亿参数，只训练旁边插入的一小撮 LoRA 参数。

LoRA 是什么：冻结原模型，在每个目标层旁边加一对小矩阵(A、B)，只训练它们。
  原本要更新 5 亿参数 -> 现在只训练几百万(<1%)。又快又省，适配器只有几 MB，可插拔。

运行：  python stage4/finetune.py
"""

import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-0.5B"     # 基座模型（首次运行自动从 HuggingFace 下载，约 1GB）
BLOCK_SIZE = 512                    # 每个训练样本的 token 长度
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus.txt")
ADAPTER_DIR = os.path.join(HERE, "lora_adapter")


def main():
    if not os.path.exists(CORPUS):
        print("找不到 corpus.txt，请先运行: python stage4/prepare_data.py")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # ---- 1. 加载 tokenizer 和基座模型 ----
    # 优先用本地模型文件夹（你在别的机器下好拷过来的）；没有才联网下载。
    local_dir = os.path.join(HERE, "Qwen2.5-0.5B")
    if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
        model_path = local_dir
        print(f"使用本地模型: {model_path}")
    else:
        from modelscope import snapshot_download
        print(f"本地未找到，从 ModelScope 下载 {MODEL_ID} ...")
        model_path = snapshot_download(MODEL_ID)
        print(f"模型本地路径: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)

    # ---- 2. 读语料 -> tokenize -> 切成固定长度的块 ----
    with open(CORPUS, "r", encoding="utf-8") as f:
        text = f.read()
    print("正在 tokenize 语料 ...")
    ids = tokenizer(text)["input_ids"]      # 整篇语料变成一长串 token id
    # 切成 BLOCK_SIZE 长度的块（丢弃结尾不足一块的零头）
    blocks = [ids[i:i + BLOCK_SIZE]
              for i in range(0, len(ids) - BLOCK_SIZE, BLOCK_SIZE)]
    print(f"语料 {len(ids)} tokens -> {len(blocks)} 个训练块（每块 {BLOCK_SIZE}）")
    dataset = Dataset.from_dict({"input_ids": blocks})

    # 因果语言模型的 collator：自动把 labels 设成 input_ids（"预测下一个 token"）
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # ---- 3. 套上 LoRA ----
    lora_config = LoraConfig(
        r=8,                    # 低秩矩阵的"秩"，越大容量越大、参数越多
        lora_alpha=16,          # 缩放系数
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 注意力的 Q/K/V/O 投影层
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    # 打印可训练参数占比 —— 你会看到只训练了 <1% 的参数
    model.print_trainable_parameters()

    # ---- 4. 训练 ----
    args = TrainingArguments(
        output_dir=os.path.join(HERE, "_trainer_out"),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=2,
        learning_rate=2e-4,
        bf16=True,                 # RTX 5090 支持 bf16，省显存提速
        logging_steps=20,
        save_strategy="no",        # 训练中不存中间 checkpoint，最后手动存适配器
        report_to="none",
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=dataset, data_collator=collator)
    print("\n=== 开始 LoRA 微调 ===")
    trainer.train()

    # ---- 5. 只保存 LoRA 适配器（几 MB，不是整个模型）----
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    size_mb = sum(
        os.path.getsize(os.path.join(ADAPTER_DIR, f))
        for f in os.listdir(ADAPTER_DIR)
        if os.path.isfile(os.path.join(ADAPTER_DIR, f))
    ) / 1e6
    print(f"\n微调完成。LoRA 适配器已保存到 {ADAPTER_DIR}（约 {size_mb:.1f} MB）")
    print("运行 compare.py 对比微调前后的生成效果。")


if __name__ == "__main__":
    main()
