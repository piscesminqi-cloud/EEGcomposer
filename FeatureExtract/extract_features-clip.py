import os
import csv
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection
)
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

# 配置参数
CSV_PATH = "test_captions.csv"
SAVE_DIR = "features_test"
PIPELINE_DIR = "best_model"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

# 加载模型
text_encoder = CLIPTextModel.from_pretrained(PIPELINE_DIR, subfolder="text_encoder").to(DEVICE)
tokenizer = CLIPTokenizer.from_pretrained(PIPELINE_DIR, subfolder="tokenizer")
safety_checker = StableDiffusionSafetyChecker.from_pretrained(PIPELINE_DIR, subfolder="safety_checker").to(DEVICE)
feature_extractor = CLIPImageProcessor.from_pretrained(PIPELINE_DIR, subfolder="feature_extractor")
image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)

# 准备存储所有特征的大数组
all_image_features = []
all_text_features = []
metadata = []

# 读取CSV文件
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# 分批处理
for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="处理特征提取"):
    batch = rows[i:i+BATCH_SIZE]
    
    # 准备图像和文本数据
    images, texts, paths = [], [], []
    for row in batch:
        try:
            img_path = row['image_path']
            caption = row['caption']
            
            # 加载并预处理图像
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            paths.append(img_path)
            texts.append(caption)
            print(f" {row['image_path']}: {str(e)}")
            
        except Exception as e:
            print(f"处理失败 {row['image_path']}: {str(e)}")
            continue

    # 处理图像特征
    if images:
        inputs = feature_extractor(images=images, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            image_embeds = image_encoder(**inputs).image_embeds
        
        # 添加到图像特征列表
        all_image_features.extend([embed.cpu().numpy() for embed in image_embeds])

    # 处理文本特征
    text_inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        text_embeds = text_encoder(**text_inputs).last_hidden_state
    
    # 添加到文本特征列表
    all_text_features.extend([embed.cpu().numpy() for embed in text_embeds])

    # 收集元数据
    for j, row in enumerate(batch):
        try:
            img_path = row['image_path']
            caption = row['caption']
            
            # 记录元数据（包含原始索引）
            metadata.append({
                "index": len(all_image_features) - len(batch) + j,  # 当前批次中的索引
                "image_path": img_path,
                "caption": caption,
            })
            
        except Exception as e:
            print(f"元数据记录失败: {str(e)}")
            continue

# 将特征列表转换为numpy数组
all_image_features = np.array(all_image_features)  # 形状: (n_samples, 768)
all_text_features = np.array(all_text_features)    # 形状: (n_samples, seq_len, 768)

# 保存所有特征
np.save(os.path.join(SAVE_DIR, "all_image_features.npy"), all_image_features)
np.save(os.path.join(SAVE_DIR, "all_text_features.npy"), all_text_features)

# 保存元数据
with open(os.path.join(SAVE_DIR, "metadata.csv"), 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["index", "image_path", "caption"])
    writer.writeheader()
    writer.writerows(metadata)

print(f"处理完成! 共保存 {len(all_image_features)} 个特征")
print(f"图像特征形状: {all_image_features.shape}")
print(f"文本特征形状: {all_text_features.shape}")