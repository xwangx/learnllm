"""
阶段② / 脚本 2：用卷积网络 (CNN) 分类 MNIST。

和 01_mnist_mlp.py 相比，**只换了模型结构，训练循环完全没动**
（都调用 common.train）——这正是把训练逻辑抽到 common.py 的好处。

为什么图像要用卷积 (CNN)？
  · MLP 把图片拉平成 784 个独立像素，丢掉了"哪些像素挨在一起"的空间信息。
  · 卷积用一个小窗口（kernel）在图上滑动，专门捕捉局部图案（边缘、笔画、角）。
  · 同一个 kernel 在整张图共享参数 -> 参数更少、还能识别"图案出现在哪都算数"。
  结果：同样几轮训练，CNN 准确率明显高于 MLP（~99% vs ~97%）。

运行：  python stage2/02_mnist_cnn.py
"""

import torch
import torch.nn as nn

import common


# ----------------------------------------------------------------------------
# CNN 模型：两层卷积 + 池化，再接全连接输出
# ----------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 卷积层1：1 个输入通道(灰度) -> 32 个特征图，3×3 窗口，padding 保持尺寸
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B,1,28,28) -> (B,32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 下采样，尺寸减半 -> (B,32,14,14)

            # 卷积层2：32 -> 64 个特征图
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (B,64,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (B,64,7,7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),            # (B,64,7,7) -> (B,64*7*7=3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),        # 缓解过拟合
            nn.Linear(128, 10),      # 10 类的分数
        )

    def forward(self, x):
        x = self.features(x)         # 先用卷积提取空间特征
        return self.classifier(x)    # 再用全连接做分类


def main():
    torch.manual_seed(0)

    train_loader, val_loader, test_loader = common.get_dataloaders(batch_size=64)

    model = CNN()
    print("=== CNN 训练 ===")
    common.train(model, train_loader, val_loader, epochs=5, lr=1e-3)

    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_acc = common.evaluate(model, test_loader, loss_fn)
    print(f"\n=== 最终测试集准确率: {test_acc:.4f} ===")

    common.show_mistakes(model, test_loader)


if __name__ == "__main__":
    main()
