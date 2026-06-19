"""
阶段② / 共享模块：数据加载 + 通用训练/评估函数。

这是本阶段的一个新主题：把可复用的代码抽出来。
01_mnist_mlp.py 和 02_mnist_cnn.py 只负责"定义各自的网络"，
训练循环、评估、数据加载全都调用这里——你会体会到：
训练流程是通用的，换模型不用换训练代码。
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ----------------------------------------------------------------------------
# 选设备：有 GPU 就用 GPU。阶段①数据小用 CPU 就行；MNIST 6 万张，上 GPU 快很多。
# ----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------
def get_dataloaders(batch_size=64, val_size=10000):
    """下载 MNIST，切出验证集，返回三个 DataLoader：训练 / 验证 / 测试。

    新概念：
    - transform: 把 PIL 图片转成张量，并做标准化（让像素值分布更适合训练）。
    - mini-batch: DataLoader 每次吐出一小批（batch_size 张）样本，而不是一次全部。
      相比阶段①"4 个点一次性全喂"，真实数据集太大，必须分批。
    - 验证集 (validation): 从训练数据里切出来、训练时"不参与学习"、只用来评估的一份。
      它帮我们发现过拟合——模型在没见过的数据上到底行不行。
    """
    # ToTensor: [0,255] 的图片 -> [0,1] 的张量，形状 (1, 28, 28)
    # Normalize: 用 MNIST 的全局均值/标准差做标准化，数值更稳、训练更快
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 公认的均值与标准差
    ])

    # 下载（第一次会联网下载到 data/，之后直接复用）
    full_train = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("data", train=False, download=True, transform=transform)

    # 把 6 万训练样本切成 "训练" + "验证" 两份
    train_size = len(full_train) - val_size
    # 固定随机种子，保证每次切分一致、结果可复现
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    # DataLoader: 负责分批、打乱。训练集要 shuffle（每轮顺序不同，训练更好）；
    # 验证/测试不需要 shuffle。
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------------
# 评估：在某个数据集上算 loss 和准确率（不更新参数）
# ----------------------------------------------------------------------------
@torch.no_grad()  # 评估不需要梯度，关掉省显存、提速
def evaluate(model, loader, loss_fn):
    model.eval()  # 切到"评估模式"（会关掉 dropout 等只在训练时生效的东西）
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)                      # 前向，得到每类的"分数"
        total_loss += loss_fn(logits, y).item() * x.size(0)
        preds = logits.argmax(dim=1)           # 分数最高的那一类就是预测
        correct += (preds == y).sum().item()   # 数对了几个
        total += x.size(0)
    return total_loss / total, correct / total  # 平均 loss, 准确率


# ----------------------------------------------------------------------------
# 训练：完整的训练循环（两个模型共用这一套）
# ----------------------------------------------------------------------------
def train(model, train_loader, val_loader, epochs=5, lr=1e-3):
    """标准训练循环。注意它和阶段①的循环本质一样，只是多了：
    分 batch、搬到 GPU、每个 epoch 在验证集上评估。
    """
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()  # 多分类标准损失（内部含 softmax）
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 比纯 SGD 收敛快

    print(f"在 {device} 上训练，共 {epochs} 轮\n")
    for epoch in range(1, epochs + 1):
        # ---- 训练一遍（遍历所有 batch）----
        model.train()  # 训练模式（dropout 等生效）
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # 这四步和阶段①一模一样：前向 → 算loss → 反向 → 更新
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- 每轮结束，在训练集和验证集上各评估一次 ----
        # 对比这两个准确率就能看出"过拟合"：训练高、验证低 = 模型在死记训练集
        train_loss, train_acc = evaluate(model, train_loader, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        gap = train_acc - val_acc
        print(f"epoch {epoch} | "
              f"训练 loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"验证 loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"过拟合间隙 {gap:+.4f}")

    return model


# ----------------------------------------------------------------------------
# 展示几个被错分的样本（训练后调用，直观看看模型在哪栽了）
# ----------------------------------------------------------------------------
@torch.no_grad()
def show_mistakes(model, loader, n=8):
    model.eval()
    shown = 0
    print(f"\n看几个被错分的例子（真值 → 预测）：")
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        wrong = (preds != y).nonzero(as_tuple=True)[0]
        for i in wrong:
            print(f"  真值 {y[i].item()} → 预测 {preds[i].item()}")
            shown += 1
            if shown >= n:
                return
