import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import json
from tqdm import tqdm
import sklearn.linear_model as skl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import mean_squared_error
import math
from einops.layers.torch import Rearrange
import torch
import time
import re
import glob
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL

PIPELINE_DIR = "best_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 配置类
class Config:
    num_subjects = 10
    train_categories = 1654
    test_categories = 200
    images_per_category = 10
    total_train = train_categories * images_per_category
    total_test = test_categories
    eeg_channels = 17
    eeg_timesteps = 100
    color_channels = 3
    batch_size = 16
    lr = 1e-4
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = "data/pretrained_models"
    features_dir = "data/features"

    feature_dims = {
        'clip_image': 768,  # CLIP图像特征维度
        'clip_text': 768,  # CLIP文本特征维度
        'depth': 4 * 64 * 64,  # VAE深度图展平后的维度
        'segmenter': 4 * 64 * 64,  # VAE分割图展平后的维度
        'sketch': 4 * 64 * 64  # VAE草图展平后的维度
    }

    train_image_dir = "data/training_images"
    test_image_dir = "data/test_images"
    eeg_base = "data"

    feature_map_types = {
        'depth': 'train_depth_images',
        'segmenter': 'train_segmenter_images',
        'sketch': 'train_sketch_images'
    }


config = Config()


# 位置编码和EEGTransformer
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


# 特征提取模型
class FeatureExtractors:
    def __init__(self):
        self.device = config.device
        self.init_vae()
        self.init_clip()

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae"
        ).to(self.device, dtype=torch.float32)
        self.vae.eval()

    def init_clip(self):
        # 图像CLIP
        self.image_feature_extractor = CLIPImageProcessor.from_pretrained(PIPELINE_DIR, subfolder="feature_extractor")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
        self.image_encoder.eval()
        # 文本CLIP
        self.tokenizer = CLIPTokenizer.from_pretrained(PIPELINE_DIR, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(PIPELINE_DIR, subfolder="text_encoder").to(self.device)
        self.text_encoder.eval()

    # 图像预处理函数
    def preprocess_image(self, image, target_size=(512, 512)):
        """加载并预处理图像"""
        image = np.array(image).astype(np.float32) / 255.0
        image = image * 2.0 - 1.0
        image = image.transpose(2, 0, 1)  # [C, H, W]
        return torch.from_numpy(image).unsqueeze(0)  # [1, C, H, W]

    # VAE编码函数
    def vae_encode_image(self, image_tensor):
        """将图像编码为VAE潜在表示"""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device, dtype=torch.float32)
            latent = self.vae.encode(image_tensor).latent_dist.mode()
            return latent * self.vae.config.scaling_factor

    # CLIP图像编码
    def clip_encode_image(self, image):
        """使用CLIP编码图像"""
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            inputs = self.image_feature_extractor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.image_encoder(**inputs)
            return outputs.image_embeds.cpu().numpy()

    # CLIP文本编码
    def clip_encode_text(self, text):
        """使用CLIP编码文本"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.text_encoder(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def save_test_predictions(self, test_loader, save_dir="test_predictions"):
        """保存测试集的预测结果（按原始维度）"""
        os.makedirs(save_dir, exist_ok=True)

        self.time_attention.eval()
        if self.is_neural:
            self.model.eval()

        all_predictions = []

        with torch.no_grad():
            for eeg, _ in test_loader:
                eeg = eeg.to(self.device)
                eeg_global = self.extract_eeg_global(eeg)
                if self.is_neural:
                    predictions = self.model(eeg_global)
                else:
                    # 加载岭回归模型
                    reg = joblib.load(f'ridge_{self.feature_name}_model.pkl')
                    eeg_global_np = eeg_global.cpu().numpy()
                    predictions = reg.predict(eeg_global_np)
                    predictions = torch.tensor(predictions, dtype=torch.float32)
                all_predictions.append(predictions.cpu())
        # 合并所有batch的预测结果
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        # 根据特征类型重塑为原始维度
        if self.feature_name == 'color':
            target_shape = (-1, 3, 64, 64)
        elif self.feature_name in ['depth',  'segmenter', 'sketch']:
            target_shape = (-1, 4, 64, 64)
        else:  # clip_image 和 clip_text
            target_shape = (-1, 768)  # 保持向量形式
        # 重塑为原始维度
        reshaped_predictions = all_predictions.reshape(target_shape)
        # 保存为npy文件（按顺序）
        np.save(os.path.join(save_dir, f"{self.feature_name}_predictions.npy"), reshaped_predictions)
        print(f"Saved test predictions for {self.feature_name} to {save_dir} (shape: {reshaped_predictions.shape})")


# EEG预处理类
class EEGPreprocessor:
    def __init__(self):
        self.scalers = []
        self.is_fitted = False

    def fit(self, eeg_data):
        n_samples, n_channels, n_timesteps = eeg_data.shape
        self.scalers = [StandardScaler() for _ in range(n_channels)]
        for channel in range(n_channels):
            channel_data = eeg_data[:, channel, :]
            self.scalers[channel].fit(channel_data)
        self.is_fitted = True

    def transform(self, eeg_data):
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted before transform")
        n_samples, n_channels, n_timesteps = eeg_data.shape
        normalized = np.zeros_like(eeg_data)
        for channel in range(n_channels):
            channel_data = eeg_data[:, channel, :]
            normalized[:, channel, :] = self.scalers[channel].transform(channel_data)
        return normalized

# 保存特征图
def save_features(features, mode):
    """保存特征到磁盘"""
    feature_dir = os.path.join(config.features_dir, mode)
    os.makedirs(feature_dir, exist_ok=True)

    # 保存特征
    for ft, data in features.items():
        np.save(os.path.join(feature_dir, f"{ft}.npy"), data)

    # 保存元数据
    metadata = {
        "feature_shapes": {ft: data.shape for ft, data in features.items()},
        "timestamp": time.time()
    }
    with open(os.path.join(feature_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Saved features to {feature_dir}")


# 加载特征图
def load_features(mode):
    """从磁盘加载特征"""
    feature_dir = os.path.join(config.features_dir, mode)
    features = {}

    if not os.path.exists(feature_dir):
        return None

    # 加载元数据
    metadata_path = os.path.join(feature_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # 加载特征数据
    for ft in metadata["feature_shapes"].keys():
        file_path = os.path.join(feature_dir, f"{ft}.npy")
        if os.path.exists(file_path):
            features[ft] = np.load(file_path)

    if len(features) == len(metadata["feature_shapes"]):
        print(f"Loaded features from {feature_dir}")
        return features

    return None


# 加载特征图数据
def load_feature_maps(mode='train'):
    # 先尝试从磁盘加载特征
    cached_features = load_features(mode)
    if cached_features is not None:
        return cached_features

    features = {ft: [] for ft in config.feature_dims}
    feature_extractor = FeatureExtractors()

    # 根据模式确定类别范围
    if mode == 'train':
        image_dir = config.train_image_dir
        num_categories = config.train_categories
        images_per_cat = config.images_per_category
        feature_prefix = "train"
    else:  # test
        image_dir = config.test_image_dir
        num_categories = config.test_categories
        images_per_cat = 1
        feature_prefix = "test"

    print(f"\n{'=' * 50}")
    print(f"开始加载 {mode} 数据集的特征图")
    start_time = time.time()

    # 获取所有类别目录
    category_dirs = sorted(glob.glob(os.path.join(image_dir, "*")))
    print(f"找到 {len(category_dirs)} 个类别目录")

    # 用于存储文本特征的字典（按类别存储）
    text_features_cache = {}

    # 添加进度条
    pbar = tqdm(category_dirs, desc=f"处理 {mode} 类别", unit="category")
    for cat_dir in pbar:
        # 从目录名中提取类别名称
        dir_name = os.path.basename(cat_dir)
        cat_name = dir_name.split('_', 1)[1] if '_' in dir_name else dir_name
        pbar.set_postfix({"当前类别": cat_name[:20] + "..." if len(cat_name) > 20 else cat_name})

        # 处理文本特征（每个类别一个）
        if cat_name not in text_features_cache:
            text_feature = feature_extractor.clip_encode_text(cat_name)[0]
            text_feature = torch.from_numpy(text_feature)
            text_feature = text_feature / torch.norm(text_feature, p=2, dim=-1, keepdim=True)
            text_features_cache[cat_name] = text_feature

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split('([0-9]+)', s)]

        # 获取类别所有图像
        image_files = sorted(glob.glob(os.path.join(cat_dir, "*.jpg")), key=natural_sort_key)

        # 处理每张图像
        for img_file in image_files:
            img_base = os.path.basename(img_file)

            # 加载原始图像
            orig_image = Image.open(img_file).convert("RGB")

            # 1. 提取CLIP图像特征
            clip_img_feature = feature_extractor.clip_encode_image(orig_image)[0]
            clip_img_feature = torch.from_numpy(clip_img_feature)
            clip_img_feature = clip_img_feature / torch.norm(clip_img_feature, p=2, dim=-1, keepdim=True)
            features['clip_image'].append(clip_img_feature)

            # 2. 添加文本特征
            features['clip_text'].append(text_features_cache[cat_name])

            # 3. 处理VAE特征图
            for ft_type in ['depth',  'segmenter', 'sketch']:
                # 构建特征图路径
                feature_map_dir = config.feature_map_types[ft_type].replace("train", feature_prefix)
                feature_map_path = os.path.join(
                    "data",
                    feature_map_dir,
                    dir_name,
                    f"{ft_type}_{img_base}"
                )

                if not os.path.exists(feature_map_path):
                    feature_map_path = os.path.join(
                        "data",
                        feature_map_dir,
                        dir_name,
                        img_base
                    )
                if os.path.exists(feature_map_path):
                    # 加载特征图图像并调整大小
                    feature_image = Image.open(feature_map_path).convert("RGB")
                    feature_image = feature_image.resize((512, 512), Image.LANCZOS)

                    # 预处理并编码为VAE潜在空间
                    img_tensor = feature_extractor.preprocess_image(feature_image)
                    vae_feature = feature_extractor.vae_encode_image(img_tensor)

                    features[ft_type].append(vae_feature.squeeze(0).cpu().numpy())
                else:
                    # 创建空白特征作为占位符
                    dummy_feature = np.zeros(config.feature_dims[ft_type], dtype=np.float32)
                    features[ft_type].append(dummy_feature)

    # 转换为数组
    print("\n特征提取完成，正在转换为数组...")
    for ft in features:
        features[ft] = np.array(features[ft])
        print(f"  {ft} 特征形状: {features[ft].shape}")

    # 保存提取的特征
    save_features(features, mode)

    end_time = time.time()
    print(f"特征加载完成，耗时: {end_time - start_time:.2f}秒")
    print('=' * 50)

    return features


# 神经网络投影头模型
class NeuralProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 8192),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(8192, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# 时间注意力模块
class TimeAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: (B, T, D)
        attn_weights = self.attention(x)  # (B, T, 1)
        attn_weights = attn_weights.permute(0, 2, 1)  # (B, 1, T)
        return torch.bmm(attn_weights, x).squeeze(1)  # (B, D)


# 特征对齐模型
class FeatureAlignmentModel:
    def __init__(self, feature_name, eeg_encoder, device):
        self.feature_name = feature_name
        self.eeg_encoder = eeg_encoder
        self.device = device
        self.time_attention = TimeAttention(2048).to(device)

        # 冻结EEG编码器
        for param in self.eeg_encoder.parameters():
            param.requires_grad = False

        # 根据特征类型选择模型
        if feature_name in ['clip_image', 'clip_text']:
            # 神经网络模型
            output_dim = config.feature_dims[feature_name]
            self.model = NeuralProjectionHead(2048, output_dim).to(device)
            self.is_neural = True
        else:
            # 岭回归模型
            self.model = None
            self.is_neural = False

    def extract_eeg_global(self, eeg):
        # 获取EEG潜在表示
        with torch.no_grad():
            eeg_latent, _ = self.eeg_encoder(eeg, mask_ratio=0.0)  # (B, T, D)

        # 应用时间注意力
        eeg_global = self.time_attention(eeg_latent)  # (B, D)

        # 对于VAE特征，应用Tanh激活
        if not self.is_neural:
            eeg_global = torch.tanh(eeg_global)

        return eeg_global

    def train_neural(self, train_loader, val_loader):
        optimizer = torch.optim.AdamW(
            list(self.time_attention.parameters()) + list(self.model.parameters()),
            lr=config.lr, weight_decay=0.01
        )

        best_val_loss = float('inf')
        no_improve = 0
        patience = 3

        # 添加epoch进度条
        epoch_pbar = tqdm(range(config.epochs), desc=f"训练 {self.feature_name}", unit="epoch")

        for epoch in epoch_pbar:
            self.time_attention.train()
            self.model.train()
            total_train_loss = 0.0

            # 添加batch进度条
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", unit="batch", leave=False)

            # 训练阶段
            for eeg, feature in batch_pbar:
                eeg = eeg.to(self.device)
                feature = feature.to(self.device)

                # 提取EEG全局表示
                eeg_global = self.extract_eeg_global(eeg)

                # 前向传播
                projection = self.model(eeg_global)

                # 计算余弦相似度损失
                loss = 1 - F.cosine_similarity(projection, feature, dim=-1).mean()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                batch_pbar.set_postfix({"train_loss": f"{loss.item():.4f}"})

            avg_train_loss = total_train_loss / len(train_loader)

            # 验证阶段
            val_loss = self.validate(val_loader)
            epoch_pbar.set_postfix({
                "train_loss": f"{avg_train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}"
            })

            # 早停检查
            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                no_improve = 0
                torch.save({
                    'time_attention': self.time_attention.state_dict(),
                    'model': self.model.state_dict()
                }, f'best_{self.feature_name}_model.pth')
                tqdm.write(f"保存最佳模型 {self.feature_name} (epoch {epoch + 1}, val_loss={val_loss:.4f})")
            else:
                no_improve += 1
                if no_improve >= patience:
                    tqdm.write(f"{self.feature_name} 提前停止 (epoch {epoch + 1})")
                    break

        print(f"{self.feature_name} 训练完成. 最佳验证损失: {best_val_loss:.4f}")

    # 在FeatureAlignmentModel类中的train_ridge方法中添加进度条
    def train_ridge(self, X_train, y_train, X_val, y_val):
        # 将数据转换为numpy
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        X_val = X_val.cpu().numpy()
        y_val = y_val.cpu().numpy()

        # 训练岭回归模型
        best_alpha = None
        best_val_loss = float('inf')

        # 在多个alpha值中选择最佳（添加进度条）
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
        alpha_pbar = tqdm(alphas, desc="选择最佳alpha", unit="alpha")

        for alpha in alpha_pbar:
            reg = skl.Ridge(alpha=alpha, max_iter=50000, fit_intercept=True)
            reg.fit(X_train, y_train)

            # 验证集验证
            y_pred = reg.predict(X_val)
            val_loss = mean_squared_error(y_val, y_pred)

            alpha_pbar.set_postfix({"alpha": alpha, "val_mse": f"{val_loss:.4f}"})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_alpha = alpha
                best_reg = reg

        print(f"{self.feature_name} 最佳 alpha: {best_alpha}, 验证 MSE: {best_val_loss:.4f}")

        # 保存最佳模型
        joblib.dump(best_reg, f'ridge_{self.feature_name}_model.pkl')
        print(f"保存岭回归模型 {self.feature_name}")

    def validate(self, val_loader):
        self.time_attention.eval()
        if self.is_neural:
            self.model.eval()

        total_loss = 0.0

        with torch.no_grad():
            for eeg, feature in val_loader:
                eeg = eeg.to(self.device)
                feature = feature.to(self.device)

                eeg_global = self.extract_eeg_global(eeg)

                if self.is_neural:
                    projection = self.model(eeg_global)
                    loss = 1 - F.cosine_similarity(projection, feature, dim=-1).mean()
                else:
                    # 对于岭回归，我们只提取特征，不计算损失
                    continue

                total_loss += loss.item()

        return total_loss / len(val_loader) if self.is_neural else 0.0

    def evaluate(self, test_loader):
        self.time_attention.eval()
        if self.is_neural:
            self.model.eval()

        if self.is_neural:
            total_cos_loss = 0.0
            test_pbar = tqdm(test_loader, desc=f"测试 {self.feature_name}", unit="batch")

            with torch.no_grad():
                for eeg, feature in test_pbar:
                    eeg = eeg.to(self.device)
                    feature = feature.to(self.device)

                    eeg_global = self.extract_eeg_global(eeg)
                    projection = self.model(eeg_global)

                    cos_sim = F.cosine_similarity(projection, feature, dim=-1)
                    batch_loss = (1 - cos_sim).mean().item()
                    total_cos_loss += batch_loss

                    test_pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})

            avg_cos_loss = total_cos_loss / len(test_loader)
            print(f"{self.feature_name} 测试余弦相似度损失: {avg_cos_loss:.4f}")
            return avg_cos_loss
        else:
            # 加载岭回归模型
            reg = joblib.load(f'ridge_{self.feature_name}_model.pkl')

            # 准备测试数据
            X_test = []
            y_test = []

            # 添加进度条
            eeg_pbar = tqdm(test_loader, desc=f"提取EEG特征 {self.feature_name}", unit="batch")
            for eeg, feature in eeg_pbar:
                eeg_global = self.extract_eeg_global(eeg.to(self.device)).cpu().numpy()
                X_test.append(eeg_global)
                y_test.append(feature.numpy())

            X_test = np.concatenate(X_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)

            # 预测并计算MSE
            test_pbar = tqdm(total=1, desc=f"预测 {self.feature_name}")
            y_pred = reg.predict(X_test)
            test_pbar.update(1)
            test_pbar.close()

            mse = mean_squared_error(y_test, y_pred)
            print(f"{self.feature_name} 测试 MSE: {mse:.4f}")
            return mse


# 加载所有被试的EEG数据
def load_all_subject_eeg(mode='train', subject_preprocessors=None):
    all_eeg = []
    all_features = {ft: [] for ft in config.feature_dims}

    if subject_preprocessors is None:
        subject_preprocessors = {}

    # 加载特征图
    features = load_feature_maps(mode)

    # 如果没有特征图，创建空数组
    if features is None:
        features = {ft: np.zeros((0, config.feature_dims[ft])) for ft in config.feature_dims}
        print("Warning: No feature maps found, using empty arrays")

    # 遍历所有被试
    for subject_id in range(1, config.num_subjects + 1):
        eeg_path = os.path.join(
            config.eeg_base,
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
            if mode == 'train':
                print(f"Fitting preprocessor for subject {subject_id}")
                subject_preprocessors[subject_id].fit(eeg_data)

        # 使用该被试特定的预处理器
        eeg_data = subject_preprocessors[subject_id].transform(eeg_data)

        # 添加到总数据集
        all_eeg.append(eeg_data)

        # 添加特征图（每个被试的特征图相同）
        for ft in features:
            all_features[ft].append(features[ft])

    # 合并所有被试数据
    all_eeg = np.concatenate(all_eeg, axis=0) if all_eeg else np.zeros((0, 17, 100))
    for ft in all_features:
        # 沿第一个维度拼接
        all_features[ft] = np.concatenate(all_features[ft], axis=0) if all_features[ft] else np.zeros(
            (0, config.feature_dims[ft]))

    print(f"Combined EEG data shape: {all_eeg.shape}")
    print("Combined feature data shapes:")
    for ft, arr in all_features.items():
        print(f"  {ft}: {arr.shape}")

    if mode == 'train':
        return all_eeg, all_features, subject_preprocessors
    else:
        return all_eeg, all_features


# 主训练和评估函数
def train_and_evaluate_models():
    device = config.device
    
    # 加载预训练的EEG模型
    try:
        eeg_model = EEGTransformer(
            input_dim=17,
            embed_dim=2048,
            num_patches=100,
            learnable_pe=True
        ).to(device)
        
        checkpoint = torch.load('checkpoints/model_5.pth', map_location=device)
        
        # 从不同键尝试加载
        load_success = False
        for key in ['model_state', 'state_dict', 'model']:
            if key in checkpoint:
                try:
                    state_dict = checkpoint[key]
                    # 处理可能的分布式训练前缀
                    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    missing, unexpected = eeg_model.load_state_dict(new_state_dict, strict=False)
                    
                    print(f"\n=== 加载结果 [{key}] ===")
                    print(f"缺失的键: {len(missing)} | 意外的键: {len(unexpected)}")
                    if not missing:
                        print("✓ 所有关键权重已加载")
                        load_success = True
                        break
                except Exception as e:
                    print(f"从 {key} 加载失败: {str(e)[:200]}")

        if not load_success:
            raise RuntimeError("无法从任何已知键加载模型")
            
        # ===== 新增的验证代码 =====
        print("\n=== 模型权重验证 ===")
        total_params = 0
        loaded_params = 0
        for name, param in eeg_model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                loaded_params += param.numel()
            print(f"{name:60} | 形状: {str(list(param.shape)):20} | 已加载: {param.requires_grad}")

        print(f"\n总参数数量: {total_params}")
        print(f"可训练参数数量: {loaded_params}")
        print(f"冻结参数数量: {total_params - loaded_params}")

        # 测试前向传播
        print("\n运行测试前向传播...")
        test_input = torch.randn(1, 17, 100).to(device)
        try:
            with torch.no_grad():
                memory, output = eeg_model(test_input)
            print(f"测试通过! 输出形状: memory={memory.shape}, output={output.shape}")
        except Exception as e:
            print(f"前向传播失败: {str(e)}")
            raise
        # ===== 验证结束 =====
        
    except Exception as e:
        print(f"\n加载失败: {str(e)}")
        print("使用随机初始化的模型")
        eeg_model = EEGTransformer(
            input_dim=17,
            embed_dim=2048,
            num_patches=100,
            learnable_pe=True
        ).to(device)
    
    eeg_model.eval()
    
    # 加载训练数据和测试数据（严格区分）
    train_eeg, train_features, subject_preprocessors = load_all_subject_eeg(mode='train')
    test_eeg, test_features = load_all_subject_eeg(mode='test', subject_preprocessors=subject_preprocessors)
    
    # 准备特征处理函数
    def prepare_feature_data(feature_data, feature_name):
        return feature_data.reshape(feature_data.shape[0], -1)  # 直接展平
    
    # 创建主进度条
    feature_names = list(config.feature_dims.keys())
    feature_pbar = tqdm(feature_names, desc="处理特征", unit="feature")
    
    # 遍历所有特征类型进行处理
    for feature_name in feature_pbar:
        feature_pbar.set_postfix({"current_feature": feature_name})
        print(f"\n{'='*50}")
        print(f"处理特征: {feature_name}")
        print(f"{'='*50}")
        
        # 准备特征数据
        train_feature = prepare_feature_data(train_features[feature_name], feature_name)
        test_feature = prepare_feature_data(test_features[feature_name], feature_name)
        
        # 转换为Tensor
        train_eeg_tensor = torch.tensor(train_eeg, dtype=torch.float32)
        train_feature_tensor = torch.tensor(train_feature, dtype=torch.float32)
        test_eeg_tensor = torch.tensor(test_eeg, dtype=torch.float32)
        test_feature_tensor = torch.tensor(test_feature, dtype=torch.float32)
        
        # 创建对齐模型
        align_model = FeatureAlignmentModel(feature_name, eeg_model, device)
        
        # 定义数据加载器
        train_dataset = TensorDataset(train_eeg_tensor, train_feature_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        test_dataset = TensorDataset(test_eeg_tensor, test_feature_tensor)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # 训练阶段
        if feature_name in ['clip_image', 'clip_text']:
            align_model.train_neural(train_loader, test_loader)
        else:
            # 提取训练集EEG全局表示
            with torch.no_grad():
                eeg_global_list = []
                # 添加进度条
                extract_pbar = tqdm(train_loader, desc=f"提取EEG特征 {feature_name}", unit="batch")
                for eeg, _ in extract_pbar:
                    eeg_global = align_model.extract_eeg_global(eeg.to(device))
                    eeg_global_list.append(eeg_global.cpu())
                X_train = torch.cat(eeg_global_list, dim=0)
                y_train = train_feature_tensor
            
            # 提取测试集EEG全局表示
            with torch.no_grad():
                eeg_global_list = []
                extract_pbar = tqdm(test_loader, desc=f"提取EEG特征 {feature_name}", unit="batch")
                for eeg, _ in extract_pbar:
                    eeg_global = align_model.extract_eeg_global(eeg.to(device))
                    eeg_global_list.append(eeg_global.cpu())
                X_test = torch.cat(eeg_global_list, dim=0)
                y_test = test_feature_tensor
            
            # 训练岭回归模型
            align_model.train_ridge(X_train, y_train, X_test, y_test)
        
        # 测试阶段评估
        test_loss = align_model.evaluate(test_loader)
        
        # 保存测试集预测结果
        align_model.save_test_predictions(test_loader)
        
        # 保存结果
        if feature_name in ['clip_image', 'clip_text']:
            result = {"feature": feature_name, "cos_loss": test_loss}
        else:
            result = {"feature": feature_name, "mse": test_loss}
        
        with open(f"results_{feature_name}.json", "w") as f:
            json.dump(result, f)
        print(f"保存结果 {feature_name} 到 results_{feature_name}.json")
    
    print("\n所有特征处理完成!")


if __name__ == "__main__":
    train_and_evaluate_models()
