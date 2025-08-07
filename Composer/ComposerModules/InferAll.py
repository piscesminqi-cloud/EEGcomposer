import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

# 导入您的自定义模型和管道
from ComposerUnet import ComposerUNet
from new_composer import ComposerStableDiffusionPipeline, LocalConditionProj

def main():
    # 1. 配置参数
    subject = "01"  # 被试编号
    n_samples = 200  # 样本数量
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = f"generated_images/sub-{subject}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载预训练管道
    print("加载预训练管道...")
    pipe = ComposerStableDiffusionPipeline.load_custom_pretrained(
        "best_model"  # 替换为您的管道保存路径
    )
    pipe = pipe.to(device)
    
    # 3. 加载所有预测的特征
    feature_dir = f"cache/predicted_embeddings/sub-{subject}/"
    print(f"从 {feature_dir} 加载预测的特征...")
    
    # 加载所有特征文件
    features = {
        "clip_image": np.load(os.path.join(feature_dir, f"pred_clip_image_sub-{subject}.npy")),
        "clip_text": np.load(os.path.join(feature_dir, f"pred_clip_text_sub-{subject}.npy")),
        "color": np.load(os.path.join(feature_dir, f"pred_color_sub-{subject}.npy")),
        "vae_sketch": np.load(os.path.join(feature_dir, f"pred_vae_sketch_sub-{subject}.npy")),
        "vae_instance": np.load(os.path.join(feature_dir, f"pred_vae_instance_sub-{subject}.npy")),
        "vae_depth": np.load(os.path.join(feature_dir, f"pred_vae_depth_sub-{subject}.npy")),
        "vae_intensity": np.load(os.path.join(feature_dir, f"pred_vae_intensity_sub-{subject}.npy")),
    }
    
    # 4. 为每个样本生成图像
    print(f"为 {n_samples} 个样本生成图像...")
    for i in tqdm(range(n_samples)):
        # 4.1 准备CLIP图像嵌入
        clip_image_emb = torch.tensor(features["clip_image"][i], dtype=torch.float32)
        clip_image_emb = clip_image_emb.unsqueeze(0)  # (1, 768)
        
        # 4.2 准备文本嵌入
        clip_text_emb = torch.tensor(features["clip_text"][i], dtype=torch.float32)
        clip_text_emb = clip_text_emb.unsqueeze(0)  # (1, 77, 768)
        
        # 4.3 处理颜色特征 (64x64x3)
        color_feat = features["color"][i].reshape(64, 64, 3)
        color_tensor = torch.tensor(color_feat, dtype=torch.float32)
        color_tensor = color_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 64, 64)
        
        # 4.4 处理VAE特征 (4x64x64)
        def prepare_vae_feature(feature_array, feature_name):
            feature = feature_array[i].reshape(4, 64, 64)
            tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)  # (1, 4, 64, 64)
            return tensor
        
        sketch = prepare_vae_feature(features["vae_sketch"], "sketch")
        instance = prepare_vae_feature(features["vae_instance"], "instance")
        depth = prepare_vae_feature(features["vae_depth"], "depth")
        intensity = prepare_vae_feature(features["vae_intensity"], "intensity")

        # 随机数据测试
        # clip_image_emb=torch.randn((1,768))
        # clip_text_emb=torch.randn((1,77,768))
        # color=torch.randn((1,4,64,64))
        # sketch=torch.randn((1,4,64,64))
        # instance=torch.randn((1,4,64,64))
        # depth=torch.randn((1,4,64,64))
        # intensity=torch.randn((1,4,64,64))
        clip_text_emb=torch.zeros((1,77,768))
        # color=torch.zeros((1,4,64,64))
        # sketch=torch.zeros((1,4,64,64))
        # instance=torch.zeros((1,4,64,64))
        # depth=torch.zeros((1,4,64,64))
        # intensity=torch.zeros((1,4,64,64))
        
        # 4.5 生成图像
        result = pipe(
            clip_image_embeds=clip_image_emb,
            text_embeddings=clip_text_emb,
            color=color_tensor,
            sketch=sketch,
            instance=instance,
            depth=depth,
            intensity=intensity,
            guidance_scale=7.5,
            num_inference_steps=50
        )
        
        # 4.6 保存生成的图像
        generated_image = result["images"][0]
        generated_image.save(os.path.join(output_dir, f"sample_{i:04d}.png"))
    
    print(f"所有图像已生成并保存到 {output_dir}")

if __name__ == "__main__":
    main()