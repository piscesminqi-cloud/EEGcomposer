import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
import os
from einops.layers.torch import Rearrange
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()  # 清空显存缓存
# 修改后的EEGTransformer模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, learnable: bool = False):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        if learnable:
            self.pe = nn.Parameter(pe)
        else:
            self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)].unsqueeze(0)


class EEGTransformer(nn.Module):
    def __init__(self, input_dim=17, embed_dim=2048, num_patches=16, learnable_pe=False):
        super().__init__()
        self.num_patches = 100

        # Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.AdaptiveAvgPool1d(num_patches),  # (B, 17, 100)
            Rearrange('b c t -> b t c'),       # (B, 100, 17)
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim),    # 投影到高维空间
            nn.LayerNorm(embed_dim)
        )

        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=num_patches, learnable=learnable_pe)

        # Transformer编码器
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=16,
                dim_feedforward=8192, activation="gelu",
                batch_first=True, norm_first=True
            ), num_layers=12
        )

        # Transformer解码器
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=16,
                dim_feedforward=8192, activation="gelu",
                batch_first=True, norm_first=True
            ), num_layers=4
        )

        # 重构头设计
        self.recon_head = nn.Sequential(
        # 维度转换：(B, 16, 2048) -> (B, 2048, 16)
        Rearrange('b n d -> b d n'),
    
        # 第一个反卷积层：快速上采样
        nn.ConvTranspose1d(
        in_channels=embed_dim,     # 2048
        out_channels=512,          # 降维到512
        kernel_size=8,             # 中等卷积核
        stride=4,                  # 步长4，上采样4倍
        padding=2                  # 保持信息不丢失
        ),  # 输出尺寸: (B, 512, 64)
    
        nn.GELU(),
        nn.BatchNorm1d(512),

        nn.ConvTranspose1d(
        in_channels=512,           # 512
        out_channels=128,          # 进一步降维到128
        kernel_size=5,             # 较小卷积核捕获细节
        stride=2,                  # 步长2，上采样2倍
        padding=2,                 # 输入填充
        output_padding=1           # 微调输出尺寸
        ),  # 输出尺寸: (B, 128, 129)
    
        # 调整到目标时间长度100
        nn.AdaptiveAvgPool1d(100),    # 输出尺寸: (B, 128, 100)
    
        # 维度转换：准备进入线性层
        Rearrange('b d n -> b n d'),  # (B, 128, 100) -> (B, 100, 128)
    
        # 多层线性映射：逐步降维
        nn.Linear(128, 64),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.GELU(),
        nn.Linear(32, input_dim),     # 映射到17个EEG通道
    
        # 最终维度转换：(B, 100, 17) -> (B, 17, 100)
        Rearrange('b n d -> b d n')
    )

    def forward(self, x: torch.Tensor, mask_ratio=0.3):
        B, C, T = x.size()  # (B, 17, 100)
        # 分块嵌入
        x_patch = self.patch_embed(x)  # (B, 16, 2048)
        N = x_patch.size(1)
        # 生成掩码
        num_masked = int(mask_ratio * N)
        mask = torch.rand(B, N, device=x.device) < mask_ratio
        mask_token = self.mask_token.expand(B, N, -1)
        pe = self.pos_encoder.pe[:N].unsqueeze(0)  # (1, N, D)
        mask_tokens = mask_token + pe  # 被屏蔽的token也包含位置信息
        # 对所有patch添加位置编码
        x_patch = x_patch + pe
        # 应用掩码
        x_masked = torch.where(mask.unsqueeze(-1), mask_tokens, x_patch)
        # 编码-解码流程
        memory = self.encoder(x_masked)
        # 调用解码器
        output = self.decoder(tgt=x_masked, memory=memory)
        output = self.recon_head(output)  # (B, N, input_dim)
        
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


# 训练函数
def train_eeg_transformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据（修改为划分训练集和验证集）
    all_eeg, preprocessors = load_all_subject_eeg(mode='train')
    eeg_data = np.concatenate(all_eeg, axis=0)
    
    # 划分训练集和验证集（80%/20%）
    train_size = int(0.8 * len(eeg_data))
    val_size = len(eeg_data) - train_size
    train_data, val_data = np.split(eeg_data, [train_size])
    
    # 转换为TensorDataset
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_data, dtype=torch.float32))
    
    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 初始化模型
    model = EEGTransformer(
        input_dim=17,
        embed_dim=2048,
        num_patches=100,
        learnable_pe=True
    ).to(device)
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # 早停参数
    patience = 10          # 容忍多少个epoch没有提升
    min_delta = 0.001      # 损失提升的最小幅度
    best_val_loss = float('inf')  # 最佳验证损失
    counter = 0            # 连续未提升的epoch计数器
    epochs_no_improve = 0  # 记录连续未提升的epoch数
    early_stop = False     # 早停标志
    
    # 训练循环
    num_epochs = 200       # 设置更大的最大epoch数
    for epoch in range(num_epochs):
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
            optimizer.step()
            
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
        
        # 早停逻辑
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            counter = 0
            # 保存最佳模型
            with open('best_eeg_transformer.pth', 'wb') as f:
                torch.save(model.state_dict(), f)
            print(f"Best model saved, val loss: {best_val_loss:.4f}")
        else:
            counter += 1
            print(f"Epoch {epoch+1} no improvement, counter: {counter}/{patience}")
            
            if counter >= patience:
                early_stop = True
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # 如果没有触发早停，保存最后一个模型
    if not early_stop:
        torch.save(model.state_dict(), 'eeg_transformer.pth')
    print("训练完成!")


if __name__ == "__main__":
    train_eeg_transformer()