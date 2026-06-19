"""
阶段② / 辅助脚本：把训练集 / 验证集 / 测试集"翻出来看看"。

不训练任何东西，只是直观展示数据：各集合多大、标签分布、真实的图片长啥样。
运行：  python stage2/peek_data.py
"""

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms


def main():
    # 注意：这里只用 ToTensor（像素 0~1），不做 Normalize ——
    # 因为我们要"看原图"，标准化后的像素值是为训练服务的、看起来会发灰。
    to_tensor = transforms.ToTensor()

    full_train = datasets.MNIST("data", train=True, download=True, transform=to_tensor)
    test_set = datasets.MNIST("data", train=False, download=True, transform=to_tensor)

    # 用和 common.py 完全相同的种子切分，保证你看到的就是训练时真正用的那一份
    val_size = 10000
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    # ---- 1. 三个集合各多大 ----
    print("=" * 50)
    print("数据集大小（每张图 1×28×28 灰度，标签 0~9）")
    print("=" * 50)
    print(f"  训练集 train : {len(train_set):>6} 张  ← 拿来学习、更新参数")
    print(f"  验证集 val   : {len(val_set):>6} 张  ← 训练时不学，只用来监控过拟合")
    print(f"  测试集 test  : {len(test_set):>6} 张  ← 全程不碰，最后才用一次")
    print(f"  （train+val 来自原始 6 万训练数据的切分；test 是独立的 1 万）")

    # ---- 2. 标签分布（每个数字大概多少张）----
    def label_counts(ds):
        counts = [0] * 10
        for _, y in ds:
            counts[y] += 1
        return counts

    print("\n" + "=" * 50)
    print("标签分布（每个数字各有多少张，应大致均匀）")
    print("=" * 50)
    print("        " + "".join(f"{d:>6}" for d in range(10)))
    for name, ds in [("训练", train_set), ("验证", val_set), ("测试", test_set)]:
        c = label_counts(ds)
        print(f"  {name}  " + "".join(f"{n:>6}" for n in c))

    # ---- 3. 把真实图片画出来（终端 ASCII 版）----
    # 每张 28×28，用字符浓淡表示像素亮度，直接在终端看到数字长什么样。
    chars = " .:-=+*#%@"  # 从暗到亮

    def ascii_image(img_tensor):
        img = img_tensor.squeeze().numpy()  # (28,28)，值 0~1
        lines = []
        for row in img:
            line = "".join(chars[min(int(v * len(chars)), len(chars) - 1)] for v in row)
            lines.append(line)
        return "\n".join(lines)

    print("\n" + "=" * 50)
    print("各集合第一张图长什么样（ASCII）")
    print("=" * 50)
    for name, ds in [("训练集", train_set), ("验证集", val_set), ("测试集", test_set)]:
        img, label = ds[0]
        print(f"\n--- {name} 第 1 张，真实标签 = {label} ---")
        print(ascii_image(img))

    # ---- 4. 存一张拼图（浏览器/图片查看器里看更清楚）----
    try:
        import matplotlib
        matplotlib.use("Agg")  # 不弹窗，直接存文件
        import matplotlib.pyplot as plt

        # 注：matplotlib 默认字体不含中文，标题用英文避免显示成方块
        fig, axes = plt.subplots(3, 10, figsize=(12, 4.2))
        row_names = ["train (50000)", "val (10000)", "test (10000)"]
        for row, (name, ds) in enumerate(
            [("train", train_set), ("val", val_set), ("test", test_set)]
        ):
            for col in range(10):
                img, label = ds[col]
                ax = axes[row][col]
                ax.imshow(img.squeeze().numpy(), cmap="gray")
                ax.set_title(str(label), fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])  # 去刻度但保留边框/标签
            # 给每行最左边标上集合名
            axes[row][0].set_ylabel(row_names[row], fontsize=10, rotation=90)
        fig.suptitle("MNIST samples — top 10 of each set (number above = true label)",
                     fontsize=12)
        fig.tight_layout()
        out = "data/mnist_samples.png"
        fig.savefig(out, dpi=120)
        print(f"\n已保存拼图到 {out}（用图片查看器打开看更清楚）")
    except ImportError:
        print("\n（未安装 matplotlib，跳过图片拼图；ASCII 版已经能看了）")


if __name__ == "__main__":
    main()
