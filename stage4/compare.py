"""
阶段④ / 对比脚本：同一个提示，看"原始 Qwen2.5-0.5B"和"LoRA 微调后"生成的区别。

这是检验微调效果的最直观方式：微调后的模型续写应更贴你 wiki 的
数据治理术语和行文风格（本体、血缘、数据标准、owner、Policy 等）。

运行：  python stage4/compare.py
       python stage4/compare.py --prompt "数据质量管理的核心是"
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


def generate(model, tokenizer, prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="数据资产建设")
    parser.add_argument("--tokens", type=int, default=150)
    args = parser.parse_args()

    # 优先用本地模型文件夹，没有才联网下载（和 finetune.py 一致）
    local_dir = os.path.join(HERE, "Qwen2.5-0.5B")
    if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
        model_path = local_dir
    else:
        from modelscope import snapshot_download
        model_path = snapshot_download(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # ---- 原始基座模型 ----
    print("加载原始基座模型 ...")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16).to(device).eval()

    print(f"\n{'='*60}\n提示: {args.prompt!r}\n{'='*60}")
    print("\n【原始 Qwen2.5-0.5B】")
    print(generate(base, tokenizer, args.prompt, args.tokens))

    # ---- 叠加 LoRA 适配器 ----
    if not os.path.isdir(ADAPTER_DIR):
        print("\n(还没有 LoRA 适配器，请先运行 finetune.py)")
        return
    print("\n加载 LoRA 适配器（叠加到基座上）...")
    tuned = PeftModel.from_pretrained(base, ADAPTER_DIR).eval()

    print("\n【LoRA 微调后（带数据治理味儿）】")
    print(generate(tuned, tokenizer, args.prompt, args.tokens))


if __name__ == "__main__":
    main()
