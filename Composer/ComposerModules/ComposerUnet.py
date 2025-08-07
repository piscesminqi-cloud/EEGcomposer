import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
from PIL import Image
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class ComposerUNet(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            class_labels: Optional[torch.Tensor] = None,
            timestep_cond: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
            down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            mid_block_additional_residual: Optional[torch.Tensor] = None,
            down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # 执行原始UNet前向传播
        return super().forward(sample=sample,
                               timestep=timestep,
                               encoder_hidden_states=encoder_hidden_states,
                               timestep_cond=timestep_cond,
                               return_dict=True)


class ComposerDataset(Dataset):
    def __init__(self,
                 num_samples=123403,
                 unlabeled_dir="unlabeled2017",
                 feature_dir="feature_maps",
                 caption_csv="caption.csv",
                 filenames_npy="filenames.npy",
                 color_dir="color_features",
                 grid_size=(64, 64),
                 color_channels=3):
        self.num_samples = num_samples
        self.unlabeled_dir = unlabeled_dir
        self.feature_dir = feature_dir

        self.color_dir = color_dir

        # 确保color_dir存在
        os.makedirs(self.color_dir, exist_ok=True)

        # 加载颜色数据
        self.filenames = np.load(filenames_npy)[:num_samples]
        self.grid_size = grid_size
        self.color_channels = color_channels

        # 预计算颜色特征图
        print("Precomputing/loading color features...")
        self.color_features = []

        for img_path in tqdm(self.filenames):
            # 获取基础文件名（不带扩展名）
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            feature_path = os.path.join(self.color_dir, f"{base_name}_color.npy")

            if os.path.exists(feature_path):
                # 加载现有特征
                self.color_features.append(np.load(feature_path))
            else:
                # 计算并保存新特征
                full_img_path = os.path.join(self.unlabeled_dir, img_path)
                img = Image.open(full_img_path).convert("RGB")
                color_feature = self.extract_spatial_color_features(img, base_name)
                np.save(feature_path, color_feature)
                self.color_features.append(color_feature)

        # 加载caption数据
        self.caption_df = pd.read_csv(caption_csv)
        self.caption_map = {
            os.path.basename(row.image_path): row.caption
            for _, row in self.caption_df.iterrows()
        }

        # 定义图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.clip_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711])
        ])

        self.feature_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_spatial_color_features(self, img, base_name: str):
        """提取空间颜色特征图"""
        img_array = np.array(img)

        # 创建特征图
        feature_map = np.zeros((self.grid_size[0],
                                self.grid_size[1],
                                self.color_channels),
                               dtype=np.float32)

        # 计算每个网格单元的大小
        cell_width = 512 // self.grid_size[1]
        cell_height = 512 // self.grid_size[0]

        # 处理每个网格单元
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                # 计算当前网格位置
                left = col * cell_width
                upper = row * cell_height
                right = min(left + cell_width, 512)
                lower = min(upper + cell_height, 512)

                # 提取网格区域
                cell = img_array[upper:lower, left:right]

                # 计算该区域的平均颜色
                if cell.size > 0:
                    avg_color = cell.mean(axis=(0, 1)) / 255.0
                    feature_map[row, col] = avg_color

        # 归一化到[-1, 1]范围
        return feature_map * 2.0 - 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 获取基础文件名
        filename = self.filenames[idx]
        base_name = os.path.splitext(filename)[0]

        # 加载原始图像
        image = Image.open(os.path.join(self.unlabeled_dir, filename)).convert("RGB")
        image_512 = self.image_transform(image)
        pixel_values = self.clip_transform(image)

        # 加载特征数据
        def load_feature(feature_type):
            path = os.path.join(
                self.feature_dir,
                feature_type,
                f"{base_name}_{feature_type}.{'jpg'}"
            )
            return self.feature_transform(Image.open(path).convert("RGB"))

        return {
            "image": image_512,
            "pixel_values": pixel_values,
            "prompt": self.caption_map[filename],
            "color": torch.tensor(self.color_features[idx]).permute(2, 0, 1),
            "sketch": load_feature("sketch"),
            "instance": load_feature("instance"),
            "depth": load_feature("depth"),
            "intensity": load_feature("intensity")
        }
