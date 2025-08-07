import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

class ImageFeatureExtractor:
    def __init__(self, grid_size, color_channels=3):
        """初始化图像特征提取器"""
        self.grid_size = grid_size
        self.color_channels = color_channels
        
    def extract_spatial_color_features(self, img, base_name: str):
        """提取空间颜色特征图"""
        img_array = np.array(img)
        feature_map = np.zeros((self.grid_size[0], self.grid_size[1], self.color_channels), dtype=np.float32)
        cell_width = 512 // self.grid_size[1]
        cell_height = 512 // self.grid_size[0]
        
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                left = col * cell_width
                upper = row * cell_height
                right = min(left + cell_width, 512)
                lower = min(upper + cell_height, 512)
                cell = img_array[upper:lower, left:right]
                if cell.size > 0:
                    avg_color = cell.mean(axis=(0, 1)) / 255.0
                    feature_map[row, col] = avg_color
        return feature_map * 2.0 - 1.0

def process_images_and_save_color_features(image_dir, output_dir, grid_size):
    """
    递归处理图像目录中的所有图像，排除检查点文件，提取颜色特征并保存为.npy文件
    """
    if not os.path.exists(image_dir):
        print(f"错误: 图像目录 '{image_dir}' 不存在")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"开始处理图像目录: {image_dir}")
    
    # 支持的图像扩展名
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    # 递归遍历所有子目录，排除检查点目录和文件
    for root_dir, dirs, files in os.walk(image_dir):
        # 排除.ipynb_checkpoints目录
        if '.ipynb_checkpoints' in root_dir:
            continue
        
        # 过滤子目录，排除检查点目录
        dirs[:] = [d for d in dirs if d != '.ipynb_checkpoints']
        
        for filename in files:
            # 排除检查点文件
            if filename.startswith('.') or '.ipynb_checkpoint' in filename:
                continue
            
            # 检查图像扩展名
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_path = os.path.join(root_dir, filename)
                image_files.append(image_path)
    
    print(f"找到 {len(image_files)} 个有效图像文件")
    if not image_files:
        print(f"错误: 在目录 '{image_dir}' 中没有找到有效图像文件")
        return
    
    # 统计各类别图像数量
    category_counts = {}
    for file in image_files:
        category = os.path.basename(os.path.dirname(file))
        category_counts[category] = category_counts.get(category, 0) + 1
    
    print("各类别图像数量统计:")
    for category, count in category_counts.items():
        print(f"{category}: {count}张")
    print(f"总计: {sum(category_counts.values())}张")
    
    extractor = ImageFeatureExtractor(grid_size=grid_size)
    processed_count = 0
    error_count = 0
    
    for image_path in tqdm(image_files, desc="处理图像"):
        try:
            # 计算相对路径作为image_id（保留子目录结构）
            image_id = os.path.relpath(image_path, image_dir)
            base_name = os.path.splitext(os.path.basename(image_id))[0]
            
            # 构建输出路径，保留子目录结构
            subdir = os.path.dirname(image_id)
            output_subdir = os.path.join(output_dir, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, f"{base_name}.npy")
            
            with Image.open(image_path) as img:
                img = img.resize((512, 512))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                color_features = extractor.extract_spatial_color_features(img, base_name)
                np.save(output_path, color_features)
                processed_count += 1
                
        except Exception as e:
            error_count += 1
            error_msg = f"处理失败: {str(e)}"
            with open("color_processing_errors.log", "a") as log_file:
                log_file.write(f"{image_path}\t{error_msg}\n")
    
    print(f"处理完成! 成功: {processed_count}, 失败: {error_count}")
    print(f"颜色特征已保存到目录: {output_dir}")

if __name__ == "__main__":
    # 配置参数
    IMAGE_DIR = "data/training_images"  # 图像根目录
    OUTPUT_DIR = "color_features_train"  # 输出目录
    GRID_SIZE = (64, 64)  # 网格大小
    
    print(f"当前工作目录: {os.getcwd()}")
    process_images_and_save_color_features(IMAGE_DIR, OUTPUT_DIR, GRID_SIZE)