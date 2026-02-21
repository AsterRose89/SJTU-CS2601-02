import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class FuYunwan:
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=1, device='cuda'):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim


        self.W1 = torch.nn.init.xavier_uniform_(torch.empty(hidden_dim, input_dim, device=self.device))
        self.b1 = torch.nn.init.zeros_(torch.empty(hidden_dim, device=self.device))
        self.W2 = torch.nn.init.xavier_uniform_(torch.empty(output_dim, hidden_dim, device=self.device))
        self.b2 = torch.nn.init.zeros_(torch.empty(output_dim, device=self.device))

        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None
        self.x = None
        self.z1 = None
        self.a1 = None

    def forward(self, x):

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device).float()
        self.x = x


        self.z1 = torch.matmul(self.x, self.W1.T) + self.b1
        self.a1 = torch.clamp(self.z1, min=0.0)


        pred = torch.matmul(self.a1, self.W2.T) + self.b2
        return pred

    def compute_loss(self, pred, y):

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32, device=self.device)
        else:
            y = y.to(self.device).float()
        if y.dim() == 1:
            y = y.unsqueeze(1)

        loss = torch.mean(torch.square(y - pred))
        return loss

    def backward(self, pred, y):

        N = pred.shape[0]

        dL_dpred = 2 * (pred - y) / N

        self.grad_W2 = torch.matmul(dL_dpred.T, self.a1)

        self.grad_b2 = torch.sum(dL_dpred, dim=0)


        dL_da1 = torch.matmul(dL_dpred, self.W2)

        dσ_dz1 = (self.z1 > 0).float()
        dL_dz1 = dL_da1 * dσ_dz1

        self.grad_W1 = torch.matmul(dL_dz1.T, self.x)

        self.grad_b1 = torch.sum(dL_dz1, dim=0)

    def update_params(self, lr=1e-3):

        assert all(g is not None for g in [self.grad_W1, self.grad_b1, self.grad_W2, self.grad_b2]), \
            "请先调用backward()计算梯度后再更新参数！"

        self.W1 -= lr * self.grad_W1
        self.b1 -= lr * self.grad_b1
        self.W2 -= lr * self.grad_W2
        self.b2 -= lr * self.grad_b2

    def get_state_dict(self):

        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim
        }

    def load_state_dict(self, state_dict):

        self.W1 = state_dict['W1'].to(self.device)
        self.b1 = state_dict['b1'].to(self.device)
        self.W2 = state_dict['W2'].to(self.device)
        self.b2 = state_dict['b2'].to(self.device)
        self.input_dim = state_dict['input_dim']
        self.hidden_dim = state_dict['hidden_dim']
        self.output_dim = state_dict['output_dim']



def save_model(model, path):

    torch.save({
        'model_state_dict': model.get_state_dict(),
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'output_dim': model.output_dim
    }, path)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_path = "homework_features_256_50000.pth"
    try:
        data = torch.load(data_path, weights_only=True)
        print("数据集的键名：", data.keys())
    except Exception as e:
        print(f"加载数据集失败：{e}")
        exit()


    x_train = data["features"]
    y_train = data["labels"]
    if y_train.dim() == 1:
        y_train = y_train.unsqueeze(1)
    print(f"数据集加载完成：x形状={x_train.shape}, y形状={y_train.shape}")


    model = FuYunwan(device='cuda')
    print(f"模型初始化完成，计算设备：{model.device}")


    epochs = 150
    batch_size = 256
    lr = 1e-3
    num_batches = x_train.shape[0] // batch_size
    loss_history = []

    print("开始训练...")
    for epoch in range(epochs):
        total_loss = 0.0
        perm = torch.randperm(x_train.shape[0])
        x_shuffled = x_train[perm].to(model.device)
        y_shuffled = y_train[perm].to(model.device)

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            pred = model.forward(x_batch)
            loss = model.compute_loss(pred, y_batch)
            model.backward(pred, y_batch)
            model.update_params(lr=lr)

            total_loss += loss.item() * batch_size

        avg_loss = total_loss / x_train.shape[0]
        loss_history.append(avg_loss)


        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Average MSE Loss: {avg_loss:.6f}")

            if (epoch + 1) % 50 == 0:
                lr *= 0.5
                print(f"学习率衰减为：{lr:.6f}")


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), loss_history, color='#1f77b4', linewidth=2)
    plt.xlabel("训练轮次 (Epoch)", fontsize=12)
    plt.ylabel("MSE损失", fontsize=12)
    plt.title("MSE损失随训练轮次的变化", fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.savefig("loss_curve.png", dpi=300, bbox_inches='tight')
    print("✅ 训练曲线已保存为 loss_curve.png")


    save_model(model, "FuYunwan_model.pth")
    print("✅ 模型已保存为 FuYunwan_model.pth")