import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ComposerPipeline import ComposerStableDiffusionPipeline

# 定义图像预处理转换
preprocess_512 = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

preprocess_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

if __name__ == "__main__":
    loaded_pipe = ComposerStableDiffusionPipeline.load_custom_pretrained(
        "./logs/composer_sd_v2_20250627_115440/best_model").to("cuda")
    print("Custom pipeline created successfully!")
    loaded_pipe.unet.to("cuda")
    loaded_pipe.clip_image_time_proj.to("cuda")
    loaded_pipe.clip_text_time_proj.to("cuda")
    loaded_pipe.local_condition_proj.to("cuda")


    # 加载并预处理图像
    def load_and_preprocess_image(image_path, size=512):
        """加载图像并进行预处理"""
        img = Image.open(image_path).convert("RGB")
        if size == 512:
            return preprocess_512(img)
        elif size == 224:
            return preprocess_224(img)
        else:
            raise ValueError(f"Unsupported size: {size}")


    # 从文件加载图像
    reference_image = load_and_preprocess_image("data/unlabeled2017/000000003524.jpg", size=512).unsqueeze(0).to("cuda")
    pixel_values = load_and_preprocess_image("data/unlabeled2017/000000003524.jpg", size=224).unsqueeze(0).to("cuda")

    sketch_image = load_and_preprocess_image("data/feature_maps/sketch/000000003524_sketch.jpg", size=512).unsqueeze(
        0).to("cuda")
    instance_image = load_and_preprocess_image("data/feature_maps/instance/000000003524_instance.jpg",
                                               size=512).unsqueeze(0).to("cuda")
    depth_image = load_and_preprocess_image("data/feature_maps/depth/000000003524_depth.jpg", size=512).unsqueeze(0).to(
        "cuda")
    intensity_image = load_and_preprocess_image("data/feature_maps/intensity/000000003524_intensity.jpg",
                                                size=512).unsqueeze(0).to("cuda")
    color = np.load("data/color_features/000000003524_color.npy").astype(np.float32)
    color = torch.tensor(color).permute(2, 0, 1).unsqueeze(0).to("cuda")  # 转换为Tensor并添加batch维度

    # 创建测试数据
    test_data = {
        "image": reference_image,  # 参考图像
        "pixel_values": pixel_values,  # CLIP输入图像
        "prompt": "a group of people walking down a street next to a river",
        "color": color,
        "sketch": sketch_image,  # 草图图像
        "instance": instance_image,  # 实例分割图像
        "depth": depth_image,  # 深度图
        "intensity": intensity_image,  # 强度图
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
    }

    # 执行推理
    output = loaded_pipe(
        image=test_data["image"],
        pixel_values=test_data["pixel_values"],
        prompt=test_data["prompt"],
        color=test_data["color"],
        sketch=test_data["sketch"],
        instance=test_data["instance"],
        depth=test_data["depth"],
        intensity=test_data["intensity"],
        guidance_scale=test_data["guidance_scale"],
        num_inference_steps=test_data["num_inference_steps"],
    )

    # 保存结果
    image = output["images"][0]
    image.save("result.png")
    print("Image saved as result.png!")
    print("Custom model loaded successfully!")
