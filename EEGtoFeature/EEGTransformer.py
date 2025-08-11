在整个对话过程中，只要用户的提问与当前主题存在任何直接或间接的关联（包括延伸问题、换个角度的提问、举例、比较、类比、补充说明等），都必须视为当前话题的自然延续，而不是新的、不相关的话题。不要提示用户“回到主题”、不要拒绝回答。即使用户的问题是拓展、跳跃、从细节转向更广泛的内容，也应当将其与之前的对话内容建立联系，并继续作答。
下面是我目前的代码，请你先知悉：
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

# 设备自动选择
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


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
    """ 线性增长 + 余弦退火 学习率调度（按 step 调用）"""
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.optimizer = optimizer
        self.step_count = 0

        # 保存当前 lr 为 initial_lr
        for param_group in self.optimizer.param_groups:
            param_group['initial_lr'] = param_group.get('lr', 0.0)

    def step(self):
        self.step_count += 1
        for param_group in self.optimizer.param_groups:
            initial_lr = param_group.get('initial_lr', 0.0)
            if self.step_count <= self.warmup_steps and self.warmup_steps > 0:
                lr_scale = float(self.step_count) / float(max(1, self.warmup_steps))
                param_group['lr'] = initial_lr * lr_scale
            else:
                # 进入余弦退火阶段
                progress = float(self.step_count - self.warmup_steps) / float(max(1, (self.total_steps - self.warmup_steps)))
                progress = min(max(progress, 0.0), 1.0)
                lr = initial_lr * (1 + math.cos(math.pi * progress)) / 2.0
                param_group['lr'] = max(lr, 1e-8)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
    return mask


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        # 正确设置 requires_grad
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # 当 d_model 为奇数时，最后一列保持 0
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: (B, N, D) -> 返回 (1, N, D) 并自动广播到 batch，确保 device/dtype 一致
        n = x.size(1)
        pe = self.pe[:, :n, :].to(x.device).type_as(x)
        return pe


class TokenEmbedding(nn.Module):
    """
    将原始 EEG (B, C, T) -> 使用 Conv1d 投影为 (B, d_model, T)（保持时间长度 T 不变）
    """
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 输入 (B, c_in, T) -> 输出 (B, d_model, T)
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model,
            kernel_size=3, padding=padding, padding_mode='circular', bias=False
        )
        # 初始化卷积权重
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor):
        # x expected shape: (B, C, T)
        # conv1d expects (B, in_channels, L) -> OK
        out = self.tokenConv(x)  # (B, d_model, T)
        return out


class SubjectEmbedding(nn.Module):
    """
    被试嵌入：每个被试一个 learnable embedding；同时保留一个 shared_embedding 用于未知/缺失被试
    subject_ids expected shape: (B,) with ints in [0, num_subjects-1]
    """
    def __init__(self, num_subjects, d_model):
        super().__init__()
        self.subject_embedding = nn.Embedding(num_subjects, d_model)
        self.shared_embedding = nn.Parameter(torch.randn(1, d_model))  # 未知被试共享 token

    def forward(self, subject_ids):
        # subject_ids: None 或 (B,)
        if subject_ids is None:
            batch_size = 1
            return self.shared_embedding.expand(batch_size, 1, -1)  # (B=1, 1, D)
        # 若 subject_ids 中存在越界索引，我们把对应位置替换为 shared_embedding
        # 先确保是 long dtype
        subject_ids = subject_ids.long()
        device = subject_ids.device
        num_emb = self.subject_embedding.num_embeddings
        mask_invalid = (subject_ids < 0) | (subject_ids >= num_emb)
        if mask_invalid.any():
            # 构建 output
            emb = self.subject_embedding(subject_ids.clone().clamp(0, num_emb - 1))  # (B, D)
            emb = emb.unsqueeze(1)  # (B,1,D)
            shared = self.shared_embedding.to(device).unsqueeze(0).expand(subject_ids.size(0), -1, -1)  # (B,1,D)
            # 将 invalid 的位置替换为 shared
            if mask_invalid.any():
                emb[mask_invalid, :, :] = shared[mask_invalid, :, :]
            return emb
        else:
            return self.subject_embedding(subject_ids).unsqueeze(1)  # (B,1,D)


class EEGTransformer(nn.Module):
    def __init__(self, input_dim=17, embed_dim=512, num_layers_enc=6, num_layers_dec=4, nhead=8, num_subjects=10, max_len=5000):
        """
        input_dim: 原始通道数 (C)
        embed_dim: Transformer 的 d_model
        我把 embed_dim 和层数降低到更常见的数值，避免显存爆炸（你可以按需要调回 2048 / 12）
        """
        super().__init__()

        # 1. TokenEmbedding: 输出 (B, d_model, T)
        self.token_embed = TokenEmbedding(input_dim, embed_dim)

        # 2. patch->sequence (保持时间步 T 不变)
        self.patch_embed_ln = nn.Sequential(
            Rearrange('b d n -> b n d'),  # after token_embed we'll pool? but here token_embed returns (B,d_model,T) -> Rearrange gives (B,T,d_model)
            nn.LayerNorm(embed_dim)
        )

        # 3. subject embedding
        self.subject_embedding = SubjectEmbedding(num_subjects, embed_dim)

        # 4. positional encoding
        self.pos_encoder = PositionalEmbedding(embed_dim, max_len=max_len)

        # 5. encoder & decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_enc)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers_dec)

        # 6. mask token (用于替换被掩码位置)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 7. reconstruction head: (B, N, D) -> (B, input_dim, N) -> (B, input_dim, T)
        self.recon_head = nn.Sequential(
            Rearrange('b n d -> b d n'),  # (B, D, N)
            nn.Conv1d(embed_dim, 512, kernel_size=1, stride=1),
            Rearrange('b d n -> b n d'),  # (B, N, 512)
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, input_dim)  # -> (B, N, input_dim)
        )

    def forward(self, x: torch.Tensor, subject_ids=None, mask_ratio=0.3):
        """
        x: (B, C, T)
        subject_ids: (B,) long 或 None
        returns: memory (encoder output), recon (B, C, T)
        """
        B, C, T = x.size()
        # token embedding -> (B, d_model, T)
        x_tokens = self.token_embed(x)  # (B, D, T)

        # to sequence (B, T, D)
        x_seq = self.patch_embed_ln(x_tokens)  # (B, T, D)
        N = x_seq.size(1)  # = T

        # add subject embedding at the beginning if provided
        if subject_ids is not None:
            sub_emb = self.subject_embedding(subject_ids)  # (B, 1, D)
            x_seq = torch.cat([sub_emb, x_seq], dim=1)  # (B, N+1, D)
            has_subject = True
        else:
            has_subject = False

        # positional encoding: same length as current sequence length
        pe = self.pos_encoder(x_seq)  # (1, seq_len, D)
        x_seq = x_seq + pe  # broadcast over batch

        seq_len = x_seq.size(1)

        # mask处理：随机掩码（对 entire sequence 的每个token），但我们通常不掩 subject token
        rand = torch.rand((B, seq_len), device=x.device)
        if has_subject:
            # 不掩 subject token（索引0）
            rand[:, 0] = 1.0  # 保证 subject token 不被掩
        mask = rand < mask_ratio  # True 表示被掩

        # 构造 mask tokens
        mask_tokens = self.mask_token.expand(B, seq_len, -1).to(x_seq.device) + pe  # (B, seq_len, D)
        x_masked = torch.where(mask.unsqueeze(-1), mask_tokens, x_seq)  # (B, seq_len, D)

        # encoder
        memory = self.encoder(x_masked)  # (B, seq_len, D)

        # decoder: 我们期望 decoder 的 tgt 长度 = 原始序列长度去掉 subject token（如果有）
        if has_subject:
            # 去掉第一位 subject token，保持 tgt 与 memory 对齐（memory 也需要去掉 subject）
            memory_no_sub = memory[:, 1:, :]  # (B, N, D)
            tgt = x_seq[:, 1:, :]             # (B, N, D)
        else:
            memory_no_sub = memory
            tgt = x_seq

        N_tgt = tgt.size(1)
        tgt_mask = generate_square_subsequent_mask(N_tgt).to(x.device)

        # decoder expects (tgt, memory)
        output = self.decoder(tgt=tgt, memory=memory_no_sub, tgt_mask=tgt_mask)  # (B, N_tgt, D)

        # reconstruction head: (B, N_tgt, D) -> (B, input_dim, N_tgt)
        recon_seq = self.recon_head(output)  # (B, N_tgt, input_dim)
        recon_seq = recon_seq.permute(0, 2, 1)  # (B, input_dim, N_tgt)

        return memory, recon_seq


class EEGPreprocessor:
    def __init__(self):
        self.scalers = []  # per-channel scalers
        self.is_fitted = False

    def fit(self, eeg_data):
        """
        eeg_data shape: (n_samples, channels, timesteps)
        fit per-channel StandardScaler
        """
        n_samples, n_channels, n_timesteps = eeg_data.shape
        self.scalers = [StandardScaler() for _ in range(n_channels)]
        for ch in range(n_channels):
            channel_data = eeg_data[:, ch, :]  # (n_samples, timesteps)
            # Sklearn expects 2D (n_samples, n_features) -> we treat timesteps as features
            self.scalers[ch].fit(channel_data)
        self.is_fitted = True

    def transform(self, eeg_data):
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted before transform")
        n_samples, n_channels, n_timesteps = eeg_data.shape
        normalized = np.zeros_like(eeg_data, dtype=np.float32)
        for ch in range(n_channels):
            channel_data = eeg_data[:, ch, :]  # (n_samples, timesteps)
            normalized[:, ch, :] = self.scalers[ch].transform(channel_data)
        return normalized


def load_all_subject_eeg(mode='train', subject_preprocessors=None, data_root="data", subject_count=10):
    """
    加载所有被试的 EEG 数据。如果 subject_preprocessors 提供，则使用对应 preprocessor transform。
    返回: all_eeg (n_total, channels, timesteps), subject_preprocessors (dict), subject_labels (n_total,)
    """
    all_eeg_list = []
    subject_labels_list = []

    if subject_preprocessors is None:
        subject_preprocessors = {}

    for subject_id in range(1, subject_count + 1):
        eeg_path = os.path.join(
            data_root,
            f"sub-{subject_id:02d}",
            f"{'train' if mode == 'train' else 'test'}_thingseeg2_avg.npy"
        )
        if not os.path.exists(eeg_path):
            print(f"Warning: EEG file not found for subject {subject_id}: {eeg_path}")
            continue

        eeg_data = np.load(eeg_path)  # expect shape (n_samples, channels, timesteps)
        print(f"Loaded EEG data for subject {subject_id}: {eeg_data.shape}")

        # create preprocessor if missing
        if subject_id not in subject_preprocessors:
            subject_preprocessors[subject_id] = EEGPreprocessor()
            if mode == 'train':
                print(f"Fitting preprocessor for subject {subject_id}")
                subject_preprocessors[subject_id].fit(eeg_data)

        # transform with subject-specific preprocessor
        eeg_data = subject_preprocessors[subject_id].transform(eeg_data)

        all_eeg_list.append(eeg_data)
        subject_labels_list.append(np.full(len(eeg_data), subject_id - 1, dtype=np.int64))

    if len(all_eeg_list) == 0:
        raise RuntimeError("No EEG data loaded. Please check data paths.")

    all_eeg = np.concatenate(all_eeg_list, axis=0)
    subject_labels = np.concatenate(subject_labels_list, axis=0)

    return all_eeg, subject_preprocessors, subject_labels


def save_checkpoint(model, optimizer, epoch, loss, path_dir="checkpoints"):
    """ 保存模型检查点 """
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
    }
    os.makedirs(path_dir, exist_ok=True)
    path = os.path.join(path_dir, f"model_{epoch + 1}.pth")
    torch.save(state, path)
    print(f"检查点保存至: {path}")


def train_simple(resume_checkpoint=None, data_root="data", subject_count=10):
    # 初始化 model & device
    model = EEGTransformer(input_dim=17, embed_dim=512, num_layers_enc=6, num_layers_dec=4, nhead=8, num_subjects=subject_count).to(device)
    print('模型初始化完成！')

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4)

    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"从检查点恢复: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"从 epoch {start_epoch} 继续训练")

    # 加载数据（训练时为 train 模式拟合各 subject 的 scaler）
    train_eeg, preprocessors, train_subjects = load_all_subject_eeg(mode='train', data_root=data_root, subject_count=subject_count)
    val_eeg, _, val_subjects = load_all_subject_eeg(mode='test', subject_preprocessors=preprocessors, data_root=data_root, subject_count=subject_count)

    # train_eeg & val_eeg 已经是合并后的 numpy arrays
    train_data = train_eeg  # shape (N_train, C, T)
    val_data = val_eeg      # shape (N_val, C, T)

    # 转为 TensorDataset
    train_dataset = TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_subjects, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_data, dtype=torch.float32),
        torch.tensor(val_subjects, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # scheduler
    num_epochs = 100
    total_steps = max(1, num_epochs * len(train_loader))
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, total_steps=total_steps)

    # early stop & loss
    early_stopper = EarlyStopper(patience=8, min_delta=0.01)
    criterion = nn.MSELoss()

    print('开始训练！')

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        total_train_samples = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        for batch in pbar:
            x = batch[0].to(device)  # (B, C, T)
            subject_ids = batch[1].to(device)

            # forward
            _, recon = model(x, subject_ids=subject_ids, mask_ratio=0.3)  # recon: (B, C, T)

            # ensure shapes match
            if recon.shape != x.shape:
                # 如果时间轴或通道不匹配，尝试做必要的 reshape 或报错
                raise RuntimeError(f"Shape mismatch: recon {recon.shape} vs x {x.shape}")

            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            batch_size = x.size(0)
            train_loss += loss.item() * batch_size
            total_train_samples += batch_size

            avg_train_loss = train_loss / total_train_samples
            pbar.set_postfix({'Train Loss': f'{loss.item():.6f}', 'Avg Train Loss': f'{avg_train_loss:.6f}'})

        train_loss = train_loss / len(train_dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                subject_ids = batch[1].to(device)

                _, recon = model(x, subject_ids=subject_ids, mask_ratio=0.3)
                if recon.shape != x.shape:
                    raise RuntimeError(f"Validation shape mismatch: recon {recon.shape} vs x {x.shape}")

                loss = criterion(recon, x)
                val_loss += loss.item() * x.size(0)
                total_val_samples += x.size(0)

        val_loss = val_loss / total_val_samples

        print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

        # 保存检查点
        save_checkpoint(model, optimizer, epoch, val_loss)

        # 早停判断
        if early_stopper(val_loss):
            print("Early stopping triggered!")
            break

    print("训练完成。")


if __name__ == "__main__":
    # 指定要恢复的检查点路径（如果不需要恢复可以设为 None）
    checkpoint_path = "checkpoints/model_5.pth" if os.path.exists("checkpoints/model_5.pth") else None
    train_simple(resume_checkpoint=checkpoint_path, data_root="data", subject_count=10)
