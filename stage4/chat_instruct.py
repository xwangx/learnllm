"""
阶段④ 指令微调版 / 第三步：和指令微调后的模型对话。

和续写版 ask.py 的区别：这里用 chat 模板做推理（喂对话格式、只解码新生成的答案），
所以模型会"回答问题"而不是"续写"。

两种用法：
  python stage4/chat_instruct.py --q "什么是数据血缘？"     # 单次
  python stage4/chat_instruct.py                            # 交互模式
  python stage4/chat_instruct.py --base                     # 对比：不加适配器（看原始基座）
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-0.5B"
SYSTEM = "你是一个数据治理领域的助手，请根据问题给出专业、简洁的回答。"
HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(HERE, "lora_adapter_instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"


def load(use_adapter=True):
    local_dir = os.path.join(HERE, "Qwen2.5-0.5B")
    model_path = local_dir if os.path.isdir(local_dir) else MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to(device).eval()
    if use_adapter and os.path.isdir(ADAPTER_DIR):
        model = PeftModel.from_pretrained(model, ADAPTER_DIR).eval()
    return model, tokenizer


def chat(model, tokenizer, question, max_new_tokens=256):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    # 用 chat 模板拼成对话，并加上"该助手回答了"的起始标记
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只解码新生成的部分（去掉输入的提示）
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", default=None)
    parser.add_argument("--base", action="store_true", help="不加适配器，看原始基座")
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()

    print("加载模型" + ("（原始基座）" if args.base else "（指令微调后）") + " ...")
    model, tokenizer = load(use_adapter=not args.base)

    if args.q is not None:
        print(chat(model, tokenizer, args.q, args.tokens))
        return

    print("进入交互模式（输入 q 退出）\n")
    while True:
        try:
            q = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if not q:
            continue
        print("助手>", chat(model, tokenizer, q, args.tokens), "\n")


if __name__ == "__main__":
    main()
