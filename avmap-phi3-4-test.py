import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
from encoder.transformer import MultiheadAttention, SimpleTransformer
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import autocast, GradScaler
# Define the new L2 loss function (Squared L2 Norm)
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, predicted_audio, audio_latent):
        return torch.norm(predicted_audio - audio_latent, p=2)**2  # Squared L2 norm
class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        """
        初始化 Huber Loss 类
        :param delta: 超参数，控制误差平方和线性之间的切换点
        """
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, predicted, target):
        """
        计算 Huber Loss
        :param predicted: 模型预测值 (Tensor)
        :param target: 实际值 (Tensor)
        :return: Huber Loss 值
        """
        error = predicted - target
        abs_error = torch.abs(error)
        is_small_error = abs_error <= self.delta

        squared_loss = 0.5 * (error ** 2)
        linear_loss = self.delta * (abs_error - 0.5 * self.delta)

        return torch.mean(torch.where(is_small_error, squared_loss, linear_loss))

# 数据加载函数
def adjust_video_size(video_tensor, target_frames, target_height, target_width):
    """
    调整视频尺寸，填充不足的部分或裁剪多余的部分。
    """
    current_frames = video_tensor.size(2)
    current_width = video_tensor.size(3)
    current_height = video_tensor.size(4)

    # 裁剪帧数
    if current_frames > target_frames:
        video_tensor = video_tensor[:, :, :target_frames, :, :]  # 裁剪到目标帧数
    elif current_frames < target_frames:
        # 填充帧数
        padding_frames = target_frames - current_frames
        #video_tensor = F.pad(video_tensor, (0, 0, 0, padding_frames), "constant", 0)
                # 使用前后帧填充不足的帧
        # 前后帧填充，重复边界帧
        video_tensor = torch.cat(
            [video_tensor] + [video_tensor[:, :, -1:, :, :]] * padding_frames, 
            dim=2
        )  # 在帧的维度上使用重复的最后一帧进行填充

    # 裁剪高度
    if current_height > target_height:
        video_tensor = video_tensor[:, :, :, :target_height, :]  # 裁剪到目标高度
    elif current_height < target_height:
        # 填充高度
        padding_height = target_height - current_height
        video_tensor = F.pad(video_tensor, (0, 0, 0, padding_height), "constant", 0)

    # 裁剪宽度
    if current_width > target_width:
        video_tensor = video_tensor[:, :, :, :, :target_width]  # 裁剪到目标宽度
    elif current_width < target_width:
        # 填充宽度
        padding_width = target_width - current_width
        video_tensor = F.pad(video_tensor, (0, padding_width, 0, 0), "constant", 0)

    return video_tensor
  
def collate_fn(batch):
    """
    自定义的collate_fn，用于处理每个批次的数据，确保视频和音频大小一致。
    通过调整视频尺寸，使得视频批次的帧数、高度和宽度一致。
    """
    # 获取批次中的视频和音频数据
    video_batch = [x[0] for x in batch]
    audio_batch = [x[1] for x in batch]

    # 目标帧数、宽度和高度
    target_frames = 25
    target_height = 45
    target_width = 80

    # 对视频序列进行调整（填充或裁剪）
    #video_batch = [adjust_video_size(x, target_frames, target_height, target_width) for x in video_batch]
    
    # 打印每个视频的尺寸（检查是否一致）
    #for i, video in enumerate(video_batch):
        #print(f"Video {i} size: {video.shape}")
    # 将视频和音频批次堆叠成Tensor
    video_batch = torch.stack(video_batch, 0)
    audio_batch = torch.stack(audio_batch, 0)
    #print(f"Loaded video_batch: {video_batch.shape}, audio_batch: {audio_batch.shape}")
    return video_batch, audio_batch



# 自定义数据集
class VideoAudioDataset(Dataset):
    def __init__(self, data_paths):
        self.data_paths = data_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        video_path, audio_path = self.data_paths[idx]
        video_latent = np.load(video_path)  # 加载视频特征 [8, 25, 80, 45]
        audio_latent = np.load(audio_path)  # 加载音频特征 [4, 10, 78]
        #print(f"Loaded video shape: {video_latent.shape}, audio shape: {audio_latent.shape}")

        return torch.tensor(video_latent, dtype=torch.float32), torch.tensor(audio_latent, dtype=torch.float32)


# 从 JSON 文件中加载数据路径
def load_data_from_json(json_file):
    with open(json_file, 'r') as f:
        data_paths = list(json.load(f).items())
    return VideoAudioDataset(data_paths)


# 定义 Phi 模型
class Phi(nn.Module):
    def __init__(self, input_dim=80 * 45, output_dim=4 * 10 * 78):
        super(Phi, self).__init__()
        self.flatten = nn.Flatten(start_dim=2)  # 展平帧和特征图
        self.projection1 = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(200),  # 批归一化 (8 是视频的通道数)
                nn.ReLU(),
                nn.Dropout(0.3),  # 加入 Dropout
                nn.Linear(512, 256),
                nn.BatchNorm1d(200),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
        # self.projection1 = nn.Sequential(
        #     nn.Linear(input_dim, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        # )
        self.hint_blocks = SimpleTransformer(
            attn_target=partial(
                MultiheadAttention,
                embed_dim=256,
                num_heads=8,
                bias=True,
                add_bias_kv=True,
            ),
            embed_dim=256,
            num_blocks=3,
            weight_init_style="pytorch",
        )
        self.projection2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
        self.unflatten = nn.Unflatten(2, (4, 10, 78))  # 转换为目标音频形状 [batch_size, 4, 10, 78]

    def forward(self, x):
        #print(f"Input shape: {x.shape}")  # 检查输入形状
                # 移除冗余维度
        x = x.squeeze(1)  # 从 [batch_size, 1, channels, frames, height, width] 到 [batch_size, channels, frames, height, width]
        batch, channels, frames, width, height = x.shape
        x = rearrange(x, 'b c f w h -> b (c f) (w h)')
        #print(f"rearrange shape: {x.shape}")  # 检查输入形状 [32, 200, 3600])
        x = self.projection1(x)
        #print(f"rearrange projection1: {x.shape}")  # 检查输入形状 ([32, 200, 256])
        x = self.hint_blocks(x)
        #print(f"rearrange shape3: {x.shape}")  # 检查输入形状 ([1, 200, 256])
        x = self.projection2(x)
        #print(f"rearrange shape4: {x.shape}")  # 检查输入形状 ([1, 200, 3120])
        x = self.unflatten(x)
        #print(f"rearrange shape5: {x.shape}")  # 检查输入形状  [1, 200, 4, 10, 78]
        # 添加目标维度 [batch_size, 1, 4, 10, 78] 
        x = x.unsqueeze(1)
        #print(f"Final output shape: {x.shape}")  [1, 1, 4, 10, 78]

        return x        
# 权重初始化函数
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

scaler = GradScaler()
# 定义训练函数
def train(model, dataloader, optimizer, criterion, device, epoch,scheduler):
    model.train()
    total_loss = 0
    #progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{epochs} Training", ncols=100)
    for video_latent, audio_latent in progress_bar:
        video_latent, audio_latent = video_latent.to(device), audio_latent.to(device)

        optimizer.zero_grad()

        with autocast():  # 自动混合精度
            predicted_audio = model(video_latent)
            loss = criterion(predicted_audio, audio_latent)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 前向传播
        #predicted_audio = model(video_latent)
        #print(f"Video batch shape: {video_latent.shape}, Audio shape: {audio_latent.shape},Pre Audio shape: {predicted_audio.shape}")

        # 计算损失
        #loss = criterion(predicted_audio, audio_latent)
        total_loss += loss.item()

        # 反向传播
        #optimizer.zero_grad()
        #loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        #optimizer.step()

        #progress_bar.set_postfix({"loss": loss.item()})
        progress_bar.set_postfix(total_loss=total_loss / (len(progress_bar)))

         # 更新学习率
    scheduler.step()

    return total_loss / len(dataloader)


# 定义验证函数
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Validating Epoch {epoch+1}/{epochs} Validation", ncols=100)
    
    with torch.no_grad():
        for video_latent, audio_latent in progress_bar:
            video_latent, audio_latent = video_latent.to(device), audio_latent.to(device)

            # 前向传播
            predicted_audio = model(video_latent)

            # 计算损失
            loss = criterion(predicted_audio, audio_latent)
            total_loss += loss.item()

           # progress_bar.set_postfix({"val_loss": loss.item()})
            progress_bar.set_postfix(total_loss=total_loss / (len(progress_bar)))

    return total_loss / len(dataloader)


# 绘制 Loss 曲线并保存
def plot_loss_curve(train_losses, val_losses, save_path="checkpoint/checkpoints-phi-meanv2afast/loss_curve-lr4-4.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.scatter(np.argmin(val_losses), min(val_losses), color='red', label='Best Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# 主程序
if __name__ == "__main__":
    # 配置
    train_json = "/home/chenghaonan/qqt/data/audiocaps-wav/promeanv2a-video/train-data.json" #"/home/chenghaonan/qqt/data/audiocaps-wav/latent-train.json" # 训练集 JSON 文件
    val_json = "/home/chenghaonan/qqt/data/audiocaps-wav/promeanv2a-video/val-data.json"#"/home/chenghaonan/qqt/data/audiocaps-wav/latent-val.json"  # 验证集 JSON 文件
    batch_size = 32
    learning_rate = 0.0001 #0.001 0.01
    epochs = 200
    weight_decay = 1e-4  # 权重衰减
    patience = 10  # Early Stopping 的耐心值
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # 创建 checkpoints 文件夹
    os.makedirs("checkpoint/checkpoints-phi-meanv2afast", exist_ok=True)

    # 数据集和加载器
    train_dataset = load_data_from_json(train_json)
    val_dataset = load_data_from_json(val_json)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=4,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=4,pin_memory=True)

    # 模型、损失函数和优化器
    model = Phi().to(device)
    criterion = HuberLoss(delta=1.0)#nn.MSELoss()
    init_weights(model)  # 初始化权重
    #criterion = L2Loss()
   # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 学习率调度器（ReduceLROnPlateau 或 StepLR）
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 存储训练和验证损失
    train_losses = []
    val_losses = []

    # 提前停止变量
    best_val_loss = float("inf")  # 初始化为正无穷
    best_epoch = 0
    patience_counter = 0

    # 训练和验证
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, scheduler)
        val_loss = validate(model, val_loader, criterion, device, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0  # 重置耐心计数
            torch.save(model.state_dict(), "checkpoint/checkpoints-phi-meanv2afast/best_model-lr4-4.pth")
            print(f"Best model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs. Best loss: {best_val_loss:.4f} (Epoch {best_epoch})")

        # 提前停止检查
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
            break


        # 保存模型
        #torch.save(model.state_dict(), f"checkpoints-phi-2/model_epoch_{epoch+1}.pth")

    # 绘制 Loss 曲线
    plot_loss_curve(train_losses, val_losses)




