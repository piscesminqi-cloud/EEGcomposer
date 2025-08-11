import glob
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from Transformer_EncDec import Encoder, EncoderLayer
from SelfAttention_Family import FullAttention, AttentionLayer
from Embed import DataEmbedding
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


class ATMS(nn.Module):
    def __init__(self, num_channels=17, seq_len=100, num_subjects=10, emb_size=40):
        super(ATMS, self).__init__()
        # iTransformer配置
        self.transformer = iTransformer(
            seq_len=seq_len,
            d_model=250,
            enc_in=num_channels,
            num_subjects=num_subjects
        )
        
        # 时空卷积模块
        self.spatio_temporal = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),  # 时间卷积
            nn.AvgPool2d((1, 25), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), stride=(1, 1)),  # 空间卷积
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Conv2d(40, emb_size, (1, 1)),
            nn.Flatten(start_dim=2)
        )
        
        # 投影头
        self.projection = nn.Sequential(
            nn.Linear(emb_size * seq_len, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(1024)
        )

    def forward(self, x, subject_ids=None):
        # 输入形状: [batch, channels, timesteps]
        x = self.transformer(x, subject_ids)  # [batch, timesteps, d_model]
        x = x.unsqueeze(1)  # [batch, 1, timesteps, channels]
        x = self.spatio_temporal(x)  # [batch, emb_size, timesteps]
        x = x.permute(0, 2, 1).reshape(x.size(0), -1)  # 展平 [batch, timesteps*emb_size]
        return self.projection(x)  # [batch, 1024]

# 完整EEG模型
class EEGTransformer(nn.Module):
    def __init__(self, input_dim=17, num_subjects=10):
        super().__init__()
        self.encoder = ATMS(
            num_channels=input_dim,
            seq_len=100,
            num_subjects=num_subjects,
            emb_size=40
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=1024, nhead=8,
                dim_feedforward=2048, activation="gelu",
                batch_first=True, norm_first=True
            ), num_layers=4
        )
        
        self.recon_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, input_dim * 100),
            nn.Tanh()
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1024))

    def forward(self, x, mask_ratio=0.3, subject_ids=None):
        B, C, T = x.size()
        latent = self.encoder(x, subject_ids).unsqueeze(1)  # [B, 1, 1024]
        
        # 掩码处理
        mask = torch.rand(B, 1, device=x.device) < mask_ratio
        latent_masked = torch.where(mask, self.mask_token.expand(B, 1, -1), latent)
        
        # 解码重建
        output = self.decoder(
            tgt=latent_masked,
            memory=latent_masked,
            tgt_mask=generate_square_subsequent_mask(1).to(x.device)
        ).squeeze(1)
        
        recon = self.recon_head(output).view(B, C, T)
        return latent.squeeze(1), recon

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

def load_all_subject_eeg(mode='train', subject_preprocessors=None):
    all_eeg_list = []
    
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
        
        # 添加到列表
        all_eeg_list.append(eeg_data)
    
    # 合并EEG数据
    all_eeg = np.concatenate(all_eeg_list, axis=0)
    
    # 创建subject_labels (0-9)
    subject_labels = np.concatenate([np.full(len(eeg), sid - 1) for sid, eeg in enumerate(all_eeg_list, 1)])
    
    return all_eeg, subject_preprocessors, subject_labels

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
    train_eeg, preprocessors, train_subjects = load_all_subject_eeg(mode='train')
    val_eeg, _, val_subjects = load_all_subject_eeg('test', subject_preprocessors=preprocessors)
    
    # 数据集和数据加载器
    train_dataset = TensorDataset(
        torch.tensor(train_eeg, dtype=torch.float32),
        torch.tensor(train_subjects, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_eeg, dtype=torch.float32),
        torch.tensor(val_subjects, dtype=torch.long)
    )
    
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
            x, sub_ids = batch
            x = x.to(device)
            sub_ids = sub_ids.to(device)
            
            _, recon = model(x, mask_ratio=0.3, subject_ids=sub_ids)
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
                x, sub_ids = batch
                x = x.to(device)
                sub_ids = sub_ids.to(device)
                
                _, recon = model(x, mask_ratio=0.3, subject_ids=sub_ids)
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
