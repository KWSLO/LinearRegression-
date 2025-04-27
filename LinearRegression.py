import torch
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def synthetic_data(w, b, num_examples):
    """随机生成特征X和标签y，并添加噪声"""
    x = torch.normal(0, 0.01, size=(num_examples, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, size=y.shape)
    return x, y


true_w = torch.tensor([2.4, -1.2, 1.2, 3.1])
true_b = 4.1


class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.x = features
        self.y = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 生成真实数据
features, labels = synthetic_data(true_w, true_b, 1000)
dataset = MyDataset(features, labels)

batch_size = 256
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

for X, y in dataloader:
    print(X.shape)
    print(y.shape)
    break

# 初始化权重
w = torch.normal(0, 0.01, size=true_w.size(), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def net(X, w, b):
    """简简单单线性回归小模型。"""
    return torch.matmul(X, w) + b


def MSE(y_hat, y):
    """.MSE."""
    return torch.mean((y_hat - y) ** 2)


def SGD(params, batch_size, lr):
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param.data = param.data - lr * param.grad / batch_size
            param.grad.zero_()


lr = 1.5
num_epochs = 300
train_loss = torch.tensor(0.0)
valid_loss = torch.tensor(0.0)
train_acc = torch.tensor(0.0)
valid_acc = torch.tensor(0.0)

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []
epochs = []

# 创建图
plt.ion()  # 开启交互模式
fig, ax = plt.subplots()
train_loss_line, = ax.plot([], [], label="Train Loss", color='blue')
valid_loss_line, = ax.plot([], [], label="Valid Loss", color='orange')
train_acc_line, = ax.plot([], [], label="Train Acc", color='green')
valid_acc_line, = ax.plot([], [], label="Valid Acc", color='red')

ax.set_xlabel("Epoch")
ax.set_ylabel("Metric Value")
ax.legend()

# 开始训练
for epoch in range(num_epochs):
    train_loss = 0.0  # 每个epoch重置
    train_acc = 0.0  # 每个epoch重置
    valid_loss = 0.0  # 每个epoch重置
    valid_acc = 0.0  # 每个epoch重置

    # 训练阶段
    for data in dataloader:
        X, y = data
        y_hat = net(X, w, b)
        loss = MSE(y_hat, y)

        loss.backward()
        SGD(params=[w, b], batch_size=batch_size, lr=lr)

        train_loss += loss.item()  # 累加训练损失
        train_acc += (loss / batch_size).item()

    # 测试阶段
    with torch.no_grad():
        for data in dataloader:  # 逐批加载数据
            features, labels = data
            y_hat = net(features, w, b)
            loss = MSE(y_hat, labels)

            valid_loss += loss.item()  # 累加测试损失
            valid_acc += (loss / len(features)).item()

    # 保存当前epoch的数据
    epochs.append(epoch + 1)
    train_losses.append(train_loss / len(dataloader))
    train_accs.append(train_acc / len(dataloader))
    valid_losses.append(valid_loss / len(dataloader))
    valid_accs.append(valid_acc / len(dataloader))

    # 更新曲线数据
    train_loss_line.set_data(epochs, train_losses)
    valid_loss_line.set_data(epochs, valid_losses)
    train_acc_line.set_data(epochs, train_accs)
    valid_acc_line.set_data(epochs, valid_accs)

    # 动态调整坐标轴范围
    ax.relim()  # 重新计算数据范围
    ax.autoscale_view()  # 自动缩放视图

    plt.draw()
    plt.pause(0.01)  # 暂停一下，让图显示出来

    # 输出每个epoch的结果
    print(f"epoch:{epoch + 1}\n"
          f"train_loss: {train_loss / len(dataloader):f}, "
          f"train_acc : {train_acc / len(dataloader):f}\n "
          f"valid_loss: {valid_loss / len(dataloader):f}, "
          f"valid_acc : {valid_acc / len(dataloader):f}\n")

    print(f"w的估计误差: {true_w - w.reshape(true_w.shape)} ")
    print(f"b的估计误差: {true_b - b}")
    print("-----------------------------------------------------")


plt.ioff()
plt.show()




