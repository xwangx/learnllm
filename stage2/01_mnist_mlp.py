"""
阶段② / 脚本 1：用全连接网络 (MLP) 分类 MNIST 手写数字。

这是阶段① PyTorch 版的直接放大：还是"全连接 + 激活"，
只不过输入从 2 维变成 784 维（28×28 像素拉平），输出从 1 个变成 10 个（0~9 十类）。
你会发现：同样的套路，换个大数据集就能跑——训练循环是通用的。

运行：  python stage2/01_mnist_mlp.py
"""

import torch
import torch.nn as nn

import common


# ----------------------------------------------------------------------------
# MLP 模型：把图片拉平成向量，过两层全连接
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),            # (B,1,28,28) -> (B,784) 把图片拉成一维向量
            nn.Linear(784, 128),     # 全连接：784 维 -> 128 维（隐藏层）
            nn.ReLU(),               # 非线性激活（比 tanh 更常用、训练更快）
            nn.Dropout(0.2),         # 随机丢弃 20% 神经元 -> 缓解过拟合（仅训练时生效）
            nn.Linear(128, 10),      # 输出层：128 -> 10，每类一个"分数"(logit)
        )

    def forward(self, x):
        return self.net(x)           # 注意：不在这里做 softmax，交叉熵损失内部会做


def main():
    torch.manual_seed(0)

    # 拿到分好批的数据
    train_loader, val_loader, test_loader = common.get_dataloaders(batch_size=64)

    # 建模型并训练（训练循环来自 common.py，和阶段①本质相同）
    model = MLP()
    print("=== MLP 训练 ===")
    common.train(model, train_loader, val_loader, epochs=5, lr=1e-3)

    # 在"测试集"上做最终评估（测试集全程没参与训练/调参，是最诚实的成绩）
    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_acc = common.evaluate(model, test_loader, loss_fn)
    print(f"\n=== 最终测试集准确率: {test_acc:.4f} ===")

    common.show_mistakes(model, test_loader)


if __name__ == "__main__":
    main()
