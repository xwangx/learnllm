"""
阶段④ 指令微调版 / 第二步：用问答数据 + chat 模板做 LoRA 指令微调。

和"续写式微调"(finetune.py) 的两个本质区别：
  1. Chat 模板：把每条数据包装成对话格式
     <|im_start|>user\n{问题}<|im_end|>\n<|im_start|>assistant\n{答案}<|im_end|>
  2. Prompt masking（只对答案算 loss）：
     把"问题部分"的 label 设成 -100（交叉熵会忽略它），
     模型只学"给定问题 -> 生成答案"，而不是把问题也一起续写。
     这正是模型从"续写机"变成"会答题的助手"的关键。

运行：  python stage4/finetune_instruct.py
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-0.5B"
MAX_LEN = 768
SYSTEM = "你是一个数据治理领域的助手，请根据问题给出专业、简洁的回答。"
HERE = os.path.dirname(os.path.abspath(__file__))
QA_PATH = os.path.join(HERE, "qa.jsonl")
ADAPTER_DIR = os.path.join(HERE, "lora_adapter_instruct")


def main():
    if not os.path.exists(QA_PATH):
        print("找不到 qa.jsonl，请先运行: python stage4/build_qa.py")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # ---- 1. 加载基座（复用本地 Qwen2.5-0.5B）----
    local_dir = os.path.join(HERE, "Qwen2.5-0.5B")
    model_path = local_dir if os.path.isdir(local_dir) else MODEL_ID
    print(f"加载基座: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.to(device)

    # ---- 2. 读问答数据 ----
    rows = [json.loads(l) for l in open(QA_PATH, encoding="utf-8")]
    print(f"问答对 {len(rows)} 条")
    dataset = Dataset.from_list(rows)

    # ---- 3. 把每条问答转成 (input_ids, labels)，并对问题部分做 mask ----
    def tokenize(ex):
        # 完整对话（含答案）
        full_msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["question"]},
            {"role": "assistant", "content": ex["answer"]},
        ]
        # 只到"该助手回答了"之前的提示部分（含 assistant 起始标记）
        prompt_msgs = full_msgs[:2]

        full_ids = tokenizer.apply_chat_template(full_msgs, tokenize=True)
        prompt_ids = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True)

        # labels：问题部分全设 -100（忽略），只保留答案部分参与 loss
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        # 截断到最大长度
        full_ids = full_ids[:MAX_LEN]
        labels = labels[:MAX_LEN]
        return {"input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels}

    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)
    # 过滤掉答案被截没了的（labels 全是 -100 就没东西可学）
    dataset = dataset.filter(lambda e: any(t != -100 for t in e["labels"]))
    print(f"可用训练样本 {len(dataset)} 条")

    # 动态 padding：input_ids 用 pad token 补齐，labels 用 -100 补齐
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)

    # ---- 4. 套 LoRA ----
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- 5. 训练 ----
    args = TrainingArguments(
        output_dir=os.path.join(HERE, "_trainer_out_instruct"),
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=25,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=dataset, data_collator=collator)
    print("\n=== 开始指令微调 ===")
    trainer.train()

    # ---- 6. 保存适配器（和续写版分开存）----
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\n指令微调完成。适配器已保存到 {ADAPTER_DIR}")
    print("运行 chat_instruct.py 提问试试。")


if __name__ == "__main__":
    main()
