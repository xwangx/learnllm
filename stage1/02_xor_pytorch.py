"""
阶段① / 脚本 2：用 PyTorch 重做一遍 XOR。

和 01_xor_numpy.py 是同一个网络、同一份数据、同样的超参，
但这次让 PyTorch 的 autograd（自动微分）替我们算梯度。

对照重点——和 numpy 版比，你"不用"再写的东西：
  · 不用手推 dW1/db1/dW2/db2，autograd 全自动（loss.backward() 一行搞定）
  · 不用手写 sigmoid/tanh 的导数
  · 不用手写 BCE 公式（用现成的 loss 函数）
你"还在"做的，本质和 numpy 版一模一样：前向 → 算 loss → 反向 → 更新参数。

运行：  python stage1/02_xor_pytorch.py
"""

import torch
import torch.nn as nn

# 固定随机种子，结果可复现
torch.manual_seed(0)


# ----------------------------------------------------------------------------
# 1. 数据：和 numpy 版完全相同的 XOR
# ----------------------------------------------------------------------------
X = torch.tensor([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]])      # (4, 2)
y = torch.tensor([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]])           # (4, 1)


# ----------------------------------------------------------------------------
# 2. 模型：同样的结构 输入(2)→tanh隐藏层(4)→sigmoid输出(1)
# ----------------------------------------------------------------------------
# nn.Linear 内部就帮我们存了 W 和 b，并且默认 requires_grad=True
# （意味着 autograd 会自动追踪它们、为它们算梯度）。
class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)   # 对应 numpy 版的 W1, b1
        self.fc2 = nn.Linear(4, 1)   # 对应 numpy 版的 W2, b2

    def forward(self, x):
        h = torch.tanh(self.fc1(x))      # 隐藏层 + tanh
        out = torch.sigmoid(self.fc2(h)) # 输出层 + sigmoid（概率）
        return out


def main():
    model = XORNet()

    # 损失函数：二元交叉熵。BCELoss 等价于 numpy 版手写的那条 BCE 公式。
    loss_fn = nn.BCELoss()

    # 优化器：SGD。它替我们做 "param -= lr * grad" 这一步。
    # （numpy 版里我们是手写四行减法，这里 optimizer.step() 一行全包。）
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

    epochs = 5000

    print("=== 开始训练（PyTorch autograd）===")
    for epoch in range(epochs + 1):
        # 1) 前向：算预测
        y_hat = model(X)
        # 2) 算 loss
        loss = loss_fn(y_hat, y)

        # 3) 反向：autograd 自动算出所有参数的梯度
        optimizer.zero_grad()   # 先清零上一轮的梯度（PyTorch 默认会累加）
        loss.backward()         # ★这一行替代了 numpy 版整个 backward() 函数

        # 4) 更新参数：优化器按梯度走一步
        optimizer.step()        # ★这一行替代了 numpy 版的四行手动减法

        if epoch % 1000 == 0:
            preds = y_hat.detach().squeeze().numpy()
            print(f"epoch {epoch:5d} | loss {loss.item():.4f} | "
                  f"预测 {preds.round(3)}")

    # ---- 训练结果 ----
    print("\n=== 训练完成，最终预测 ===")
    with torch.no_grad():                 # 推理阶段不需要算梯度，关掉省资源
        y_hat = model(X).squeeze()
    for xi, yi, pi in zip(X, y.squeeze(), y_hat):
        ok = "✓" if (pi > 0.5) == (yi > 0.5) else "✗"
        print(f"  输入 {xi.tolist()}  真值 {int(yi)}  "
              f"预测 {pi.item():.3f}  {ok}")


if __name__ == "__main__":
    main()
