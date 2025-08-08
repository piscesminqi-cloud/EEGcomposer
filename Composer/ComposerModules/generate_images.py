import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from ComposerUnet import ComposerUNet
from new_composer import ComposerStableDiffusionPipeline, LocalConditionProj

# 配置参数
device = "cuda" if torch.cuda.is_available() else "cpu"
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)

# 1. 加载预训练管道
print("加载预训练管道...")
pipe = ComposerStableDiffusionPipeline.load_custom_pretrained("best_model")
pipe = pipe.to(device)

# 2. 加载所有预测的特征
feature_dir = "intermediate_features"
print(f"从 {feature_dir} 加载预测的特征...")

# 加载所有特征文件
features = {
    "clip_image": np.load(os.path.join(feature_dir, "eeg_global_clip_image.npy")),
    "clip_text": np.load(os.path.join(feature_dir, "eeg_global_clip_text.npy")),
    "depth": np.load(os.path.join(feature_dir, "eeg_global_depth.npy")),
    "segmenter": np.load(os.path.join(feature_dir, "eeg_global_segmenter.npy")),
    "sketch": np.load(os.path.join(feature_dir, "eeg_global_sketch.npy"))
}

# 3. 为每个样本生成图像
n_samples = features["clip_image"].shape[0]
print(f"为 {n_samples} 个样本生成图像...")

for i in tqdm(range(n_samples)):
    # 准备CLIP图像嵌入
    clip_image_emb = torch.tensor(features["clip_image"][i], dtype=torch.float32).unsqueeze(0).to(device)

    # 准备文本嵌入
    clip_text_emb = torch.tensor(features["clip_text"][i], dtype=torch.float32).unsqueeze(0).to(device)


    # 准备VAE特征
    def prepare_vae_feature(feature_array, feature_name):
        # 对于VAE特征，需要重塑为原始维度
        if feature_name in ['depth', 'segmenter', 'sketch']:
            # 重塑为 (1, 4, 64, 64)
            feature = feature_array[i].reshape(4, 64, 64)
            return torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        return torch.tensor(feature_array[i], dtype=torch.float32).unsqueeze(0).to(device)


    depth = prepare_vae_feature(features["depth"], "depth")
    segmenter = prepare_vae_feature(features["segmenter"], "segmenter")
    sketch = prepare_vae_feature(features["sketch"], "sketch")

    # 生成图像
    result = pipe(
        clip_image_embeds=clip_image_emb,
        text_embeddings=clip_text_emb,
        sketch=sketch,
        instance=segmenter,  # 使用segmenter作为instance
        depth=depth,
        guidance_scale=7.5,
        num_inference_steps=50
    )

    # 保存生成的图像
    generated_image = result["images"][0]
    generated_image.save(os.path.join(output_dir, f"sample_{i:04d}.png"))

print(f"所有图像已生成并保存到 {output_dir}")