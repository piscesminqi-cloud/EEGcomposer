import cv2
import numpy as np
import random
import os
from tqdm import tqdm

class IntensityGenerator:
    def __init__(self):
        self.weights_set = [
            [0.299, 0.587, 0.114],  # 传统RGB权重
            [0.2126, 0.7152, 0.0722],  # sRGB标准
            [0.333, 0.333, 0.334]  # 均等权重
        ]
    
    def get_intensity(self, image_np, output_size=(512,512)):
        """输入RGB numpy数组 (0-255 uint8), 输出强度图"""
        # 随机选择一组权重
        weights = random.choice(self.weights_set)  # 关键修复：从self.weights_set获取
        
        # 计算强度
        resized = cv2.resize(image_np, output_size)
        intensity = np.dot(resized[..., :3], weights).astype(np.uint8)
        return intensity

def process_images(input_dir, output_dir=None):
    """
    处理输入目录中的所有图像，生成强度图
    
    参数:
        input_dir: 输入图像目录路径
        output_dir: 可选，输出目录路径。如果为None则不保存
    """
    # 初始化强度生成器
    intensity_gen = IntensityGenerator()
    
    # 确保输出目录存在
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历所有子目录和图像
    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # 构建完整文件路径
                img_path = os.path.join(root, file)
                
                try:
                    # 读取图像
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"警告: 无法读取图像 {img_path}")
                        continue
                    
                    # 生成强度图
                    intensity = intensity_gen.get_intensity(img)
                    if intensity.ndim == 2:
                        intensity = np.repeat(intensity[:, :, np.newaxis], 3, axis=2)
                    
                    # 如果需要保存结果
                    if output_dir is not None:
                        # 保持原始目录结构
                        rel_path = os.path.relpath(root, input_dir)
                        output_subdir = os.path.join(output_dir, rel_path)
                        
                        if not os.path.exists(output_subdir):
                            os.makedirs(output_subdir)
                        
                        # 保存强度图
                        output_path = os.path.join(output_subdir, f"intensity_{file}")
                        print(intensity.shape)
                        cv2.imwrite(output_path, intensity)
                        print(f"已处理并保存: {img_path} -> {output_path}")
                    else:
                        print(f"已处理: {img_path}")
                        
                except Exception as e:
                    print(f"处理图像 {img_path} 时出错: {str(e)}")

if __name__ == "__main__":
    input_dir = "data/training_images"
    output_dir = "data/training_intensity_images"  # 设置为None则不保存
    
    process_images(input_dir, output_dir)