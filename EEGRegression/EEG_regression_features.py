import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL
from sklearn.preprocessing import StandardScaler
import glob
import re
from tqdm import tqdm
import time
import re
import json

PIPELINE_DIR = "best_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 配置参数
class Config:
    num_subjects = 10
    train_categories = 1654
    test_categories = 200
    images_per_category = 10
    total_train = train_categories * images_per_category  # 16540
    total_test = test_categories  # 200
    eeg_channels = 17
    eeg_timesteps = 100
    grid_size = (64, 64) 
    color_channels = 3
    batch_size = 64
    lr = 1e-4
    epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = "data/pretrained_models"
    features_dir = "data/features"
    
    # 特征图维度配置
    feature_dims = {
        'clip_image': 768,       # CLIP图像特征维度
        'clip_text': 768,        # CLIP文本特征维度
        'color': (3, 64, 64),    # 颜色特征图维度 (C, H, W)
        'depth': (4, 64, 64),    # VAE深度图特征维度
        'intensity': (4, 64, 64), # VAE强度图特征维度
        'segmenter': (4, 64, 64), # VAE分割图特征维度
        'sketch': (4, 64, 64)    # VAE草图特征维度
    }
    
    # 路径配置
    train_image_dir = "data/training_images"
    test_image_dir = "data/test_images"
    eeg_base = "data"
    
    # 特征图类型映射 - 修正为实际路径名
    feature_map_types = {
        'depth': 'train_depth_images',
        'intensity': 'train_intensity_images',
        'segmenter': 'train_segmenter_images',
        'sketch': 'train_sketch_images'
    }
    early_stop_patience = 10  # 验证损失连续不下降的epoch数
    min_delta_absolute = 1e-6    # 绝对最小改进阈值
    min_delta_relative = 0.01    # 相对改进阈值 (1%)

config = Config()

# 残差块定义
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        
        # 如果维度变化，需要shortcut投影
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out + identity

# 卷积精炼块（用于特征图输出）
class ConvRefinementBlock(nn.Module):
    def __init__(self, in_channels, feature_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.norm1 = nn.BatchNorm2d(in_channels * 2)
        self.norm2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.activation(x + identity)

# Transformer-MLP模型
class EEG2FeatureModel(nn.Module):
    def __init__(self, input_dim, output_dim, feature_shape=None):
        super().__init__()
        self.feature_shape = feature_shape
        
        # 增强的输入投影 - 增加维度
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU()
        )
        
        # 更深的Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=512,           # 大幅增加模型容量
            nhead=16,              # 增加注意力头
            dim_feedforward=2048,  # 保持前馈网络维度
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=True        # 添加Pre-LN结构
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers=8)  # 增加层数
        
        # 增强的时间注意力
        self.time_attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        # 改进的MLP解码器 - 更平缓的维度变化
        if feature_shape is None:  # 向量输出
            self.mlp = nn.Sequential(
                nn.Linear(512, 1024),
                nn.GELU(),
                nn.Dropout(0.3),
                # 残差块1
                ResidualBlock(1024, 2048),
                # 残差块2
                ResidualBlock(2048, 4096),
                # 残差块3
                ResidualBlock(4096, 8192),
                # 残差块4
                ResidualBlock(8192, 16384),
                # 最终输出层
                nn.Linear(16384, output_dim)
            )
        else:  # 特征图输出
            # 首先将向量解码为特征图
            self.mlp = nn.Sequential(
                nn.Linear(512, 1024),
                nn.GELU(),
                nn.Dropout(0.3),
                # 残差块1
                ResidualBlock(1024, 2048),
                # 残差块2
                ResidualBlock(2048, 4096),
                # 残差块3
                ResidualBlock(4096, 8192),
                # 残差块4
                ResidualBlock(8192, 16384),
                # 转换为特征图
                nn.Linear(16384, np.prod(feature_shape)),
                nn.Unflatten(1, feature_shape),
                # 卷积精炼网络
                ConvRefinementBlock(feature_shape[0], feature_shape)
            )

    def forward(self, x):
        # 输入形状: (batch, channels, timesteps)
        batch, channels, timesteps = x.shape
        # 调整维度为 (batch, timesteps, channels)
        x = x.permute(0, 2, 1)
        # 投影到更高维度
        x = self.input_proj(x)  # (batch, timesteps, 512)
        # Transformer处理
        x = self.transformer(x)  # (batch, timesteps, 512)
        # 计算时间注意力权重
        attn_weights = self.time_attention(x)  # (batch, timesteps, 1)
        attn_weights = attn_weights.permute(0, 2, 1)  # (batch, 1, timesteps)
        # 加权求和
        x = torch.bmm(attn_weights, x).squeeze(1)  # (batch, 512)
        return self.mlp(x)

# 特征提取模型初始化
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
    
    # 图像预处理函数 - 修改为接受PIL图像
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
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.text_encoder(**inputs)
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def extract_spatial_color_features(img, grid_size=(64, 64)):
    """提取空间颜色特征图"""
    img_array = np.array(img)

    # 创建特征图
    feature_map = np.zeros((grid_size[0], grid_size[1], 3), dtype=np.float32)

    # 计算每个网格单元的大小
    cell_width = img.width // grid_size[1]
    cell_height = img.height // grid_size[0]

    # 处理每个网格单元
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            # 计算当前网格位置
            left = col * cell_width
            upper = row * cell_height
            right = min(left + cell_width, img.width)
            lower = min(upper + cell_height, img.height)

            # 提取网格区域
            cell = img.crop((left, upper, right, lower))
            cell_array = np.array(cell)

            # 计算该区域的平均颜色
            if cell_array.size > 0:
                avg_color = cell_array.mean(axis=(0, 1)) / 255.0
                feature_map[row, col] = avg_color

    # 归一化到[-1, 1]范围并调整维度顺序
    return (feature_map * 2.0 - 1.0).transpose(2, 0, 1)  # 调整为 (C, H, W)

# EEG预处理类
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

    
# 特征图数据集
class FeatureDataset(Dataset):
    def __init__(self, eeg_data, feature_maps, feature_type):
        self.eeg_data = eeg_data
        self.feature_maps = feature_maps[feature_type]
        self.feature_type = feature_type
        
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        feature = torch.tensor(self.feature_maps[idx], dtype=torch.float32)
        
        # 对CLIP特征进行归一化
        if self.feature_type in ['clip_image', 'clip_text']:
            feature = feature / torch.norm(feature, p=2, dim=-1, keepdim=True)
        
        return eeg, feature

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

# 加载特征图数据 - 修正特征图路径处理
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
    
    print(f"\n{'='*50}")
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
            
            # 1. 提取颜色特征
            color_feature = extract_spatial_color_features(orig_image, config.grid_size)
            color_feature = color_feature / torch.norm(color_feature, p=2, dim=-1, keepdim=True)
            features['color'].append(color_feature)
            
            # 2. 提取CLIP图像特征
            clip_img_feature = feature_extractor.clip_encode_image(orig_image)[0]
            features['clip_image'].append(clip_img_feature)
            
            # 3. 添加文本特征
            features['clip_text'].append(text_features_cache[cat_name])
            
            # 4. 处理VAE特征图
            for ft_type in ['depth', 'intensity', 'segmenter', 'sketch']:
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
    print(f"特征加载完成，耗时: {end_time-start_time:.2f}秒")
    print('='*50)
    
    return features

# 加载所有被试的EEG数据
def load_all_subject_eeg(mode='train', subject_preprocessors=None):
    """
    subject_preprocessors: 字典 {subject_id: 预处理器实例}
    """
    all_eeg = []
    all_features = {ft: [] for ft in config.feature_dims}
    
    # 初始化预处理器字典（如果未提供）
    if subject_preprocessors is None:
        subject_preprocessors = {}
    
    # 加载特征图（所有被试共享）
    features = load_feature_maps(mode)
    
    # 遍历所有被试
    for subject_id in range(1, config.num_subjects + 1):
        # 加载EEG数据
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
            
            # 如果是训练模式，拟合该被试的预处理器
            if mode == 'train':
                print(f"Fitting preprocessor for subject {subject_id}")
                subject_preprocessors[subject_id].fit(eeg_data)
        
        # 使用该被试特定的预处理器
        eeg_data = subject_preprocessors[subject_id].transform(eeg_data)
        
        # 添加到总数据集
        all_eeg.append(eeg_data)
        
        # 复制特征图
        for ft in features:
            all_features[ft].append(features[ft])
    
    # 合并所有被试的数据
    all_eeg = np.concatenate(all_eeg, axis=0) if all_eeg else np.array([])
    for ft in all_features:
        all_features[ft] = np.concatenate(all_features[ft], axis=0) if all_features[ft] else np.array([])
    
    print(f"Combined EEG data shape: {all_eeg.shape}")
    print("Combined all_features data shapes:")
    for ft, arr in all_features.items():
        print(f"  {ft}: {arr.shape}")
    if mode == 'train':
        return all_eeg, all_features, subject_preprocessors  # 返回所有预处理器
    else:
        return all_eeg, all_features

# 主训练函数
def train_all_subjects_models():
    print("=== Training models for all subjects ===")
    
    # 加载所有训练数据
    print("Loading training data...")
    train_eeg, train_features, eeg_preprocessor = load_all_subject_eeg('train')
    
    # 加载所有测试数据
    print("Loading test data...")
    test_eeg, test_features = load_all_subject_eeg('test', subject_preprocessors=eeg_preprocessor)
    
    if train_eeg.size == 0 or test_eeg.size == 0:
        print("Error: No EEG data loaded. Exiting.")
        return
    
    # 为每个特征训练模型
    for feature_type in config.feature_dims:
        print(f"\n{'='*50}")
        print(f"Training model for {feature_type} feature...")
        
        # 创建数据集
        train_dataset = FeatureDataset(train_eeg, train_features, feature_type)
        test_dataset = FeatureDataset(test_eeg, test_features, feature_type)
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"Skipping {feature_type} due to empty dataset")
            continue
            
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size,
            num_workers=2,
            pin_memory=True
        )
        
        # 初始化模型
        output_dim = config.feature_dims[feature_type]
        if isinstance(output_dim, tuple):  # 特征图
            model = EEG2FeatureModel(
                input_dim=config.eeg_channels,
                output_dim=np.prod(output_dim),
                feature_shape=output_dim
            )
        else:  # 向量特征
            model = EEG2FeatureModel(
                input_dim=config.eeg_channels,
                output_dim=output_dim
            )
        model.to(config.device)
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.lr,
            weight_decay=1e-4,  # 增加权重衰减
            betas=(0.9, 0.999)
        )
        
        # 使用余弦退火学习率调度
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config.epochs * len(train_loader),  # 按迭代次数而非epoch
            eta_min=1e-6
        )
        
        # 训练循环
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False
        
        for epoch in range(config.epochs):
            if early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
            model.train()
            train_loss = 0.0
            batch_idx = 0
            
            for eeg, feature in train_loader:
                eeg = eeg.to(config.device, non_blocking=True)
                feature = feature.to(config.device, non_blocking=True)
                
                # 前向传播
                outputs = model(eeg)
                loss = criterion(outputs, feature)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()
                
                # 更新学习率
                scheduler.step()
                
                train_loss += loss.item() * eeg.size(0)
                batch_idx += 1
            
            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for eeg, feature in test_loader:
                    eeg = eeg.to(config.device, non_blocking=True)
                    feature = feature.to(config.device, non_blocking=True)
                    outputs = model(eeg)
                    loss = criterion(outputs, feature)
                    val_loss += loss.item() * eeg.size(0)
            
            # 计算平均损失
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(test_loader.dataset)
            
            # 打印统计信息
            print(f"Epoch {epoch+1}/{config.epochs} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # 检查是否是第一个epoch或是否有改进
            if best_val_loss == float('inf'):
                # 第一个epoch，无条件保存模型
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), 
                          f"models/all_subjects_{feature_type}_best_model.pth")
                print(f"Saved best model for {feature_type} with val loss: {val_loss:.6f} (first epoch)")
            else:
                # 计算最小改进阈值（相对和绝对）
                min_delta = max(config.min_delta_absolute, best_val_loss * config.min_delta_relative)
                
                # 保存最佳模型
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), 
                              f"models/all_subjects_{feature_type}_best_model.pth")
                    print(f"Saved best model for {feature_type} with val loss: {val_loss:.6f}")
                else:
                    epochs_no_improve += 1
                    print(f"No improvement for {epochs_no_improve}/{config.early_stop_patience} epochs")
                    
                    # 检查早停条件
                    if epochs_no_improve >= config.early_stop_patience:
                        early_stop = True
        
        # 保存最终模型
        torch.save(model.state_dict(), 
                  f"models/all_subjects_{feature_type}_final_model.pth")
        print(f"Saved final model for {feature_type} feature")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print('='*50)

# 确保模型目录存在
def setup_directories():
    os.makedirs("models", exist_ok=True)
    os.makedirs(config.features_dir, exist_ok=True)
    os.makedirs("data/features/train", exist_ok=True)
    os.makedirs("data/features/test", exist_ok=True)

# 主函数
if __name__ == "__main__":
    setup_directories()
    train_all_subjects_models()
