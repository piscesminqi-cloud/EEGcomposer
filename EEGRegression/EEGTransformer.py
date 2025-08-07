import glob
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops.layers.torch import Rearrange
from natsort import natsorted
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()  # 清空显存缓存

os.environ["USE_PEFT_BACKEND"] = "True"
test_device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# %% 辅助模块定义 ----------------------------------------------------------------
class EarlyStopper:
    """ 早停策略 """

    def __init__(self, patience=5, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.min_loss - self.min_delta:
            self.min_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class WarmupCosineSchedule:
    """ 线性增长+余弦退火 """

    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.optimizer = optimizer
        self.step_count = 0

        # 将当前学习率保存为 'initial_lr'
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group.get('lr', 0)  # 确保存在'lr'

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # 线性增长
            lr_scale = min(1.0, self.step_count / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['initial_lr'] * lr_scale
        else:
            # 余弦退火
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                lr = param_group['initial_lr'] * (1 + np.cos(np.pi * progress)) / 2
                param_group['lr'] = max(lr, 1e-6)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


# 可学习的位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, learnable: bool = False):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (16,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if learnable:
            self.pe = nn.Parameter(pe)  # 可学习参数
        else:
            self.register_buffer('pe', pe)  # 固定编码

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)].unsqueeze(0)


class EEGTransformer(nn.Module):
    def __init__(self, input_dim=17, embed_dim=2048, num_patches=100, learnable_pe=True):  #2048
        super().__init__()
        # 分块嵌入层
        self.patch_embed = nn.Sequential(
            nn.Conv1d(input_dim, embed_dim, kernel_size=1, stride=1), 
            nn.AdaptiveAvgPool1d(num_patches),
            Rearrange('b d n -> b n d'), 
            nn.LayerNorm(embed_dim)
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 位置编码模块
        self.pos_encoder = PositionalEncoding(embed_dim, num_patches, learnable_pe)

        # Transformer编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=16,
                dim_feedforward=8192, activation="gelu",
                batch_first=True, norm_first=True
            ), num_layers=12
        )  # (B, 16, 1024)

        # Transformer解码器
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=16,
                dim_feedforward=8192, activation="gelu",
                batch_first=True, norm_first=True
            ), num_layers=8
        )

        # 重建头部
        self.recon_head = nn.Sequential(
            Rearrange('b n d -> b d n'),  # (B,16,1024) --->(B,1024,16)
            nn.Conv1d(embed_dim, 512, kernel_size=1, stride=1),
            Rearrange('b d n -> b n d'),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, input_dim)
        )

    def forward(self, x: torch.Tensor, mask_ratio=0.3):
        B, C, T = x.size()

        # 分块嵌入
        x_patch = self.patch_embed(x)  # (B, N, D)
        N = x_patch.size(1)

        # 生成随机掩码
        mask = torch.rand(B, N, device=x.device) < mask_ratio

        # 获取位置编码
        pe = self.pos_encoder.pe[:N].unsqueeze(0)  # (1, N, D)
        # 创建带PE的mask tokens
        mask_tokens = self.mask_token.expand(B, N, -1) + pe
        # 原始token添加PE
        x_patch = x_patch + pe
        # 替换被遮蔽的token
        x = torch.where(mask.unsqueeze(-1), mask_tokens, x_patch)

        # 编码-解码流程
        memory = self.encoder(x)  # (B, 16, 1024)
        tgt_mask = generate_square_subsequent_mask(N).to(x.device)
        # 调用解码器
        output = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)

        # 重建输出
        output = self.recon_head(output).permute(0, 2, 1)
        return memory, output



class EEGPreprocessor:
    def __init__(self):
        self.scalers = []  # 为每个通道创建一个独立的scaler
        self.is_fitted = False
    
    def fit(self, eeg_data):
        """
        eeg_data形状: (n_samples, channels, timesteps)
        对每个通道独立进行标准化
        """
        n_samples, n_channels, n_timesteps = eeg_data.shape
        self.scalers = [StandardScaler() for _ in range(n_channels)]
        # 对每个通道独立拟合
        for channel in range(n_channels):
            # 获取该通道的所有时间步数据 (n_samples, timesteps)
            channel_data = eeg_data[:, channel, :]
            self.scalers[channel].fit(channel_data)
        self.is_fitted = True
    
    def transform(self, eeg_data):
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted before transform")
        n_samples, n_channels, n_timesteps = eeg_data.shape
        normalized = np.zeros_like(eeg_data)
        # 对每个通道独立应用标准化
        for channel in range(n_channels):
            channel_data = eeg_data[:, channel, :]
            normalized[:, channel, :] = self.scalers[channel].transform(channel_data)
        
        return normalized


def load_all_subject_eeg(mode='train', subject_preprocessors=None):
    all_eeg = []
    
    # 初始化预处理器字典（如果未提供）
    if subject_preprocessors is None:
        subject_preprocessors = {}
    
    # 遍历所有被试
    for subject_id in range(1, 11):
        # 加载EEG数据
        eeg_path = os.path.join(
            "data", 
            f"sub-{subject_id:02d}", 
            f"{'train' if mode == 'train' else 'test'}_thingseeg2_avg.npy"
        )
        
        if not os.path.exists(eeg_path):
            print(f"Warning: EEG file not found for subject {subject_id}")
            continue
            
        eeg_data = np.load(eeg_path)
        print(f"Loaded EEG data for subject {subject_id}: {eeg_data.shape}")
        
        # 获取或创建该被试的预处理器
        if subject_id not in subject_preprocessors:
            subject_preprocessors[subject_id] = EEGPreprocessor()
            
            # 如果是训练模式，拟合该被试的预处理器
            if mode == 'train':
                print(f"Fitting preprocessor for subject {subject_id}")
                subject_preprocessors[subject_id].fit(eeg_data)
        
        # 使用该被试特定的预处理器
        eeg_data = subject_preprocessors[subject_id].transform(eeg_data)
        
        # 添加到总数据集
        all_eeg.append(eeg_data)
    return all_eeg,  subject_preprocessors  # 返回所有预处理器

def train_simple(resume_checkpoint=None):
    # 初始化
    device = torch.device("cuda")
    model = EEGTransformer().to(device)
    print('模型初始化完成！')

    # 优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    
    # 如果提供了检查点路径，加载模型和优化器状态
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"从检查点恢复: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        print(f"从 epoch {start_epoch} 继续训练")
    
    # 加载数据
    all_eeg, preprocessors = load_all_subject_eeg(mode='train')
    train_data = np.concatenate(all_eeg, axis=0)
    val_data, _ = load_all_subject_eeg('test', subject_preprocessors=preprocessors)
    val_data = np.concatenate(val_data, axis=0)
    
    # 数据集和数据加载器
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # 学习率调度
    num_epochs = 100
    total_steps = num_epochs * len(train_loader)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, total_steps=total_steps)

    # 早停机制
    early_stopper = EarlyStopper(patience=8, min_delta=0.01)
    criterion = nn.MSELoss()
    print('开始训练！')

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        for batch in pbar:
            x = batch[0].to(device)
            
            _, recon = model(x, mask_ratio=0.3)
            loss = criterion(recon, x)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * x.size(0)
            pbar.set_postfix({
                'Train Loss': f'{loss.item():.4f}',
                'Avg Train Loss': f'{train_loss / (pbar.n + 1):.4f}'
             })
        
        train_loss = train_loss / len(train_dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                
                _, recon = model(x, mask_ratio=0.3)
                loss = criterion(recon, x)
                
                val_loss += loss.item() * x.size(0)
        
            val_loss = val_loss / len(val_dataset)
        
        # 打印训练信息
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # 早停判断
        if early_stopper(val_loss):
            print("Early stopping triggered!")
            break

        # 保存检查点
        save_checkpoint(model, optimizer, epoch, val_loss)

def save_checkpoint(model, optimizer, epoch, loss):
    """ 保存模型检查点 """
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(state, f"checkpoints/model_{epoch + 1}.pth")
    print(f"检查点保存至: checkpoints/model_{epoch + 1}.pth")

if __name__ == "__main__":
    # 指定要恢复的检查点路径
    checkpoint_path = "checkpoints/model_5.pth"
    train_simple(resume_checkpoint=checkpoint_path)
