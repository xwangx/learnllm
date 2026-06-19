"""
阶段③ / 训练脚本：把莎士比亚文本喂给 GPT，训练它预测下一个字符。

流程：
  1. 读文本 -> 建字符<->整数的映射 (tokenize)
  2. 切成 训练/验证 两份
  3. 训练循环：随机取一批文本块 -> 预测下一个字符 -> 算 loss -> 反向更新
  4. 训完把模型权重 + 字符映射存盘，供 generate.py 使用

运行：  python stage3/train.py
"""

import os
import torch

from gpt import GPT, GPTConfig

# ----------------------------------------------------------------------------
# 超参数（小而能出效果，RTX 5090 几分钟训完）
# ----------------------------------------------------------------------------
batch_size = 64          # 每批多少个文本块
block_size = 256         # 上下文长度：一次看 256 个字符
max_iters = 5000         # 训练迭代次数
eval_interval = 500      # 每隔多少步评估一次
eval_iters = 200         # 评估时平均多少个 batch
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    torch.manual_seed(1337)

    # ---- 1. 读文本并 tokenize（字符级）----
    input_path = os.path.join(HERE, "input.txt")
    if not os.path.exists(input_path):
        import urllib.request
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
               "master/data/tinyshakespeare/input.txt")
        print("语料不存在，正在下载莎士比亚文本...")
        urllib.request.urlretrieve(url, input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))            # 所有不同字符，排序保证可复现
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}   # 字符 -> 整数
    itos = {i: ch for i, ch in enumerate(chars)}   # 整数 -> 字符
    encode = lambda s: [stoi[c] for c in s]        # 字符串 -> 整数列表
    print(f"语料 {len(text)} 字符，词表大小 {vocab_size}")

    # 整个文本编码成一个长整数张量
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))             # 90% 训练，10% 验证
    train_data, val_data = data[:n], data[n:]

    # ---- 2. 取一批数据 ----
    # 随机选 batch_size 个起点，每个截取 block_size+1 个字符：
    # x = 前 block_size 个，y = 右移一位（即每个位置"下一个字符"的答案）
    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i + block_size] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)

    # ---- 3. 评估当前 loss（训练/验证集各算几个 batch 取平均）----
    @torch.no_grad()
    def estimate_loss(model):
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                _, loss = model(*get_batch(split))
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ---- 4. 建模型 ----
    config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=6, n_head=6, n_embd=384, dropout=0.2)
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f}M，训练设备: {device}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # ---- 5. 训练循环 ----
    for it in range(max_iters + 1):
        # 定期评估并打印
        if it % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {it:5d} | 训练 loss {losses['train']:.4f} | "
                  f"验证 loss {losses['val']:.4f}")

        # 一步训练：取数据 -> 前向算loss -> 反向 -> 更新（和前两阶段同一套路）
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ---- 6. 存盘：权重 + 字符映射（generate.py 需要用同一套映射）----
    ckpt = {
        "model_state": model.state_dict(),
        "config": config.__dict__,
        "stoi": stoi,
        "itos": itos,
    }
    out_path = os.path.join(HERE, "gpt_shakespeare.pt")
    torch.save(ckpt, out_path)
    print(f"\n训练完成，已保存到 {out_path}")
    print("现在可以运行 generate.py 让它写莎士比亚了。")


if __name__ == "__main__":
    main()
