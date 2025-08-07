import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 初始化VAE模型
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    subfolder="vae",
    cache_dir="data/pretrained_models/"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = vae.to(device, dtype=torch.float16)
vae.eval()

# 图像预处理函数（匹配ComposerDataset的归一化）
def preprocess_image(image_path, target_size=(512, 512)):
    """加载并预处理图像，应用相同的归一化"""
    image = Image.open(image_path).convert("RGB")
    # 调整尺寸
    image = image.resize(target_size, Image.LANCZOS)
    # 转换为numpy并归一化到[0,1]
    image = np.array(image).astype(np.float32) / 255.0
    # 应用相同的归一化：(x - 0.5) / 0.5 = x*2 - 1
    image = image * 2.0 - 1.0
    # 调整维度顺序 [C, H, W] 并增加批次维度
    image = image.transpose(2, 0, 1)  # [C, H, W]
    return torch.from_numpy(image).unsqueeze(0)  # [1, C, H, W]

# 编码函数
def encode_image(image_tensor):
    """将图像编码为VAE潜在表示"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device, dtype=torch.float16)
        latent = vae.encode(image_tensor).latent_dist.mode()
        return latent * vae.config.scaling_factor

# 主处理函数（适应目录结构）
def process_condition_images(base_dir, output_dir, conditions, splits):
    """
    处理条件图像并保存嵌入
    
    参数:
        base_dir: 包含条件图像的根目录
        output_dir: 保存npy文件的目录
        conditions: 要处理的条件列表 
        splits: 数据集分割列表 ['train', 'test']
    """
    # 映射关系：代码中的条件名称 -> 目录名称
    condition_dir_map = {
        "sketch": "sketch_images",
        "instance": "segmenter_images",  # 注意：目录中是segmenter，代码中是instance
        "depth": "depth_images",
        "intensity": "intensity_images"
    }
    
    # 元数据存储（记录图像路径与嵌入的对应关系）
    metadata = {cond: {split: [] for split in splits} for cond in conditions}
    
    for cond in conditions:
        for split in splits:
            # 获取当前条件类型的目录
            cond_dir_name = condition_dir_map[cond]
            input_dir = os.path.join(base_dir, f"{split}_{cond_dir_name}")
            
            # 创建输出目录
            cond_output_dir = os.path.join(output_dir, cond, split)
            os.makedirs(cond_output_dir, exist_ok=True)
            
            # 遍历所有类别目录
            class_dirs = [d for d in os.listdir(input_dir) 
                         if os.path.isdir(os.path.join(input_dir, d))]
            
            for class_dir in tqdm(class_dirs, desc=f"Processing {split} {cond}"):
                class_path = os.path.join(input_dir, class_dir)
                
                # 处理当前类别下的所有图像
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        
                        try:
                            # 预处理图像（应用归一化）
                            img_tensor = preprocess_image(img_path)
                            
                            # 编码图像
                            latent = encode_image(img_tensor)
                            
                            # 保存嵌入文件
                            base_name = os.path.splitext(img_file)[0]
                            output_path = os.path.join(cond_output_dir, f"{base_name}.npy")
                            np.save(output_path, latent[0].cpu().numpy())  # 移除批次维度
                            
                            # 记录元数据
                            metadata[cond][split].append({
                                "class": class_dir,
                                "image": img_file,
                                "embedding": os.path.relpath(output_path, output_dir)
                            })
                        except Exception as e:
                            print(f"Error processing {img_path}: {str(e)}")
    
    # 保存元数据
    metadata_path = os.path.join(output_dir, "embeddings_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Embeddings and metadata saved to {output_dir}")

# 配置参数
if __name__ == "__main__":
    BASE_IMAGE_DIR = "data"  # 包含训练和测试目录的根目录
    OUTPUT_DIR = "condition_embeddingss"
    CONDITIONS = ['sketch', 'instance', 'depth', 'intensity']
    SPLITS = ['train', 'test']  # 注意：目录中是training/test，代码中是train/test
    
    process_condition_images(
        base_dir=BASE_IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        conditions=CONDITIONS,
        splits=SPLITS
    )