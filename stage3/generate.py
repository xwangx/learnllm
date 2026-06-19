"""
阶段③ / 生成脚本：加载训练好的 GPT，让它生成莎士比亚风格的文本。

演示两件事：
  1. 怎么从硬盘加载训练好的模型（torch.load）——这是阶段②故意留的一课。
  2. 自回归生成：从一个起始字符出发，一个字一个字地往后写。

运行：  python stage3/generate.py
       python stage3/generate.py --prompt "ROMEO:" --tokens 1000 --temp 0.8
"""

import os
import argparse
import torch

from gpt import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="\n", help="起始文本")
    parser.add_argument("--tokens", type=int, default=500, help="生成多少个字符")
    parser.add_argument("--temp", type=float, default=0.8,
                        help="温度：<1 更保守稳妥，>1 更随机大胆")
    args = parser.parse_args()

    # ---- 1. 加载 checkpoint（权重 + 字符映射）----
    ckpt_path = os.path.join(HERE, "gpt_shakespeare.pt")
    if not os.path.exists(ckpt_path):
        print("找不到模型文件，请先运行: python stage3/train.py")
        return
    # weights_only=True：只反序列化张量和基础数据类型，避免执行任意代码（安全做法）
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # 用存下来的 config 重建同样结构的模型，再把权重灌进去
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    stoi, itos = ckpt["stoi"], ckpt["itos"]
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join(itos[i] for i in l)

    # ---- 2. 自回归生成 ----
    # 把起始文本编码成整数序列，作为生成的"种子"
    start = encode(args.prompt)
    idx = torch.tensor([start], dtype=torch.long, device=device)  # (1, len)

    print(f"=== 起始: {args.prompt!r} | 生成 {args.tokens} 字符 | 温度 {args.temp} ===\n")
    out = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temp)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
