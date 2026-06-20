"""
阶段⑥ / 训练 + 诊断：在 CIFAR-10 上训练 ViT 或 CNN，并用吴恩达方法论分析。

运行：
  python stage6/train_vit.py --model vit    # 训练 ViT
  python stage6/train_vit.py --model cnn    # 训练 CNN 基线
  python stage6/train_vit.py --model vit --epochs 30

训练完会打印：训练/验证/测试准确率、偏差/方差诊断（对标人类水平）、混淆分析。
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))

from vit import ViT
from cnn_baseline import CNN

device = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "蛙", "马", "船", "卡车"]
HUMAN_ACC = 0.94      # CIFAR-10 人类水平约 94%（贝叶斯误差的估计，用于吴恩达式诊断）


def get_loaders(batch_size=128):
    # 训练集做数据增强（ViT 小数据上很依赖它）；测试集只标准化
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    # 优先用 get_cifar.py 下好的图片文件夹（fast.ai 镜像）；没有才走 torchvision 下载
    img_train = os.path.join(HERE, "data", "cifar10", "train")
    img_test = os.path.join(HERE, "data", "cifar10", "test")
    if os.path.isdir(img_train) and os.path.isdir(img_test):
        # ImageFolder 按类名字母序分配标签，正好是 CIFAR-10 的标准顺序
        full = datasets.ImageFolder(img_train, transform=train_tf)
        test = datasets.ImageFolder(img_test, transform=test_tf)
    else:
        full = datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10("data", train=False, download=True, transform=test_tf)
    # 切出验证集（注意：验证集也带增强，仅用于训练中粗看；最终用测试集）
    n_val = 5000
    train_set, val_set = random_split(
        full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(0))
    return (DataLoader(train_set, batch_size, shuffle=True, num_workers=0),
            DataLoader(val_set, batch_size, shuffle=False, num_workers=0),
            DataLoader(test, batch_size, shuffle=False, num_workers=0))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return correct / total


@torch.no_grad()
def confusion(model, loader, n=10):
    model.eval()
    mat = np.zeros((n, n), dtype=int)
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            mat[t][p] += 1
    return mat


def diagnose(train_acc, val_acc, test_acc):
    """吴恩达式 偏差/方差 诊断。"""
    human_err = 1 - HUMAN_ACC
    train_err = 1 - train_acc
    val_err = 1 - val_acc
    avoidable_bias = train_err - human_err   # 可避免偏差：训练误差 vs 人类水平
    variance = val_err - train_err           # 方差：验证误差 vs 训练误差

    print("\n" + "=" * 56)
    print("吴恩达式诊断（偏差 vs 方差）")
    print("=" * 56)
    print(f"  人类水平误差(估计) : {human_err:.1%}")
    print(f"  训练误差           : {train_err:.1%}")
    print(f"  验证误差           : {val_err:.1%}")
    print(f"  → 可避免偏差(训练−人类) : {avoidable_bias:.1%}")
    print(f"  → 方差(验证−训练)       : {variance:.1%}")
    if avoidable_bias > variance:
        print("  诊断：偏差为主 → 模型欠拟合。对策：更大/更强模型、训练更久。")
    else:
        print("  诊断：方差为主 → 过拟合。对策：更多数据、更强增强、正则化。")
    print(f"  最终测试准确率: {test_acc:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vit", "cnn"], default="vit")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    train_loader, val_loader, test_loader = get_loaders()

    if args.model == "vit":
        model = ViT().to(device)
        lr = args.lr or 5e-4
    else:
        model = CNN().to(device)
        lr = args.lr or 1e-3
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型: {args.model.upper()}  参数量: {n_params/1e6:.2f}M  设备: {device}")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    print(f"\n=== 训练 {args.epochs} 轮 ===")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            va = evaluate(model, val_loader)
            print(f"epoch {epoch:3d} | 验证准确率 {va:.4f}")

    # 最终评估（训练集准确率用于偏差/方差诊断）
    train_acc = evaluate(model, train_loader)
    val_acc = evaluate(model, val_loader)
    test_acc = evaluate(model, test_loader)
    diagnose(train_acc, val_acc, test_acc)

    # 误差分析：最容易混淆的类别对
    mat = confusion(model, test_loader)
    pairs = []
    for i in range(10):
        for j in range(10):
            if i != j:
                pairs.append((mat[i][j], i, j))
    pairs.sort(reverse=True)
    print("\n最容易混淆的 5 对（真值 → 误判）：")
    for cnt, i, j in pairs[:5]:
        print(f"  {CLASSES[i]} → {CLASSES[j]}: {cnt} 次")


if __name__ == "__main__":
    main()
