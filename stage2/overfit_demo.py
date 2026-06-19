"""
阶段② / 演示脚本：故意制造一次明显的过拟合，亲眼看它发生。

放大过拟合的三个手段：
  ① 去掉 Dropout（不再抑制死记硬背）
  ② 用更大的网络（参数多 = 记忆力强）
  ③ 只用很少的训练数据（1000 张，容易被整本背下来）

观察重点：
  · 训练准确率会冲到 ~100%（把这 1000 张全背下来了）
  · 验证准确率早早卡住、不再提升
  · 验证 loss 训练到后面甚至"掉头向上"——这是过拟合的经典信号
  · "过拟合间隙"越拉越大

运行：  python stage2/overfit_demo.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import common  # 复用 evaluate 和 device


def main():
    torch.manual_seed(0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("data", train=False, download=True, transform=transform)

    # ② 只取前 1000 张做训练 —— 数据少，模型很容易"背下来"
    small_train = Subset(full_train, range(1000))
    train_loader = DataLoader(small_train, batch_size=64, shuffle=True)
    # 验证集照常用大一点，好看出泛化差距
    val_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # ① + ③ 更大的网络、且没有 Dropout
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 10),
        # 注意：故意不放 nn.Dropout
    ).to(common.device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"训练样本只有 {len(small_train)} 张 | 大网络 | 无 Dropout")
    print(f"在 {common.device} 上训练 40 轮，观察过拟合\n")
    print(f"{'epoch':>5} | {'训练loss':>8} {'训练acc':>7} | "
          f"{'验证loss':>8} {'验证acc':>7} | {'间隙':>7}")
    print("-" * 60)

    best_val_loss = float("inf")
    for epoch in range(1, 41):
        model.train()
        for x, y in train_loader:
            x, y = x.to(common.device), y.to(common.device)
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()

        train_loss, train_acc = common.evaluate(model, train_loader, loss_fn)
        val_loss, val_acc = common.evaluate(model, val_loader, loss_fn)
        gap = train_acc - val_acc

        # 标记验证 loss 开始回升的时刻（过拟合的转折点）
        flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        elif val_loss > best_val_loss * 1.05:
            flag = "  ← 验证loss回升，过拟合中"

        if epoch <= 5 or epoch % 5 == 0:
            print(f"{epoch:>5} | {train_loss:>8.4f} {train_acc:>7.4f} | "
                  f"{val_loss:>8.4f} {val_acc:>7.4f} | {gap:>+7.4f}{flag}")

    print("\n结论：训练准确率几乎 100%（背下了那 1000 张），"
          "但验证准确率远低于它——\n这就是过拟合：在见过的数据上完美，在没见过的数据上平庸。")


if __name__ == "__main__":
    main()
