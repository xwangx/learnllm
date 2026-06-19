"""
阶段④ / 提问脚本：直接和微调后的模型聊（只用微调后的模型，不对比基座）。

注意：这是"续写式"微调的模型，它会顺着你的话往下写，而不是规整地答题。
想效果好，把提示写成一个"开头"而不是疑问句，例如：
  "数据质量管理的核心是"   比   "什么是数据质量管理？"   更顺。

两种用法：
  python stage4/ask.py --q "数据资产建设的步骤包括"      # 单次
  python stage4/ask.py                                    # 交互模式，连续提问
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-0.5B"
HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(HERE, "lora_adapter")
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    local_dir = os.path.join(HERE, "Qwen2.5-0.5B")
    if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
        model_path = local_dir
    else:
        from modelscope import snapshot_download
        model_path = snapshot_download(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to(device).eval()
    model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()
    return model, tokenizer


def answer(model, tokenizer, prompt, max_new_tokens=180):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", default=None, help="单次提问；不给则进入交互模式")
    parser.add_argument("--tokens", type=int, default=180)
    args = parser.parse_args()

    print("加载微调后的模型 ...")
    model, tokenizer = load_model()

    if args.q is not None:
        print(answer(model, tokenizer, args.q, args.tokens))
        return

    # 交互模式：连续提问，输入 q 退出
    print("进入交互模式（输入 q 或 quit 退出）\n")
    while True:
        try:
            prompt = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("q", "quit", "exit"):
            break
        if not prompt:
            continue
        print("模型>", answer(model, tokenizer, prompt, args.tokens), "\n")


if __name__ == "__main__":
    main()
