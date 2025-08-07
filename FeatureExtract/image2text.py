import os
import csv
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 初始化模型
try:
    processor = BlipProcessor.from_pretrained("salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("salesforce/blip-image-captioning-large").to("cuda" if torch.cuda.is_available() else "cpu")
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit(1)

# 路径设置 - 使用绝对路径更可靠
BASE_DIR = "/root/autodl-tmp"  # 根据您的实际路径调整
IMAGE_DIR = os.path.join(BASE_DIR, "data/training_images")
OUTPUT_CSV = os.path.join(BASE_DIR, "train_captions.csv")

print(f"图像目录: {IMAGE_DIR}")
print(f"输出文件: {OUTPUT_CSV}")

# 检查图像目录是否存在
if not os.path.exists(IMAGE_DIR):
    print(f"错误: 图像目录不存在 - {IMAGE_DIR}")
    exit(1)

# 递归查找所有图像文件
def find_image_files(root_dir):
    image_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    image_files = []
    for dirpath, dirs, filenames in os.walk(root_dir):
         # 排除.ipynb_checkpoints目录
        if '.ipynb_checkpoints' in root_dir:
            continue
        
        # 过滤子目录，排除检查点目录
        dirs[:] = [d for d in dirs if d != '.ipynb_checkpoints']
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                image_files.append(os.path.join(dirpath, filename))
    return image_files

# 获取所有图像文件
image_files = find_image_files(IMAGE_DIR)

if not image_files:
    print(f"警告: 在 {IMAGE_DIR} 中未找到任何图像文件")
    exit(0)

print(f"找到 {len(image_files)} 张图像")

# 处理图像并生成描述
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'image_path', 'caption'])
    
    processed_count = 0
    error_count = 0
    
    for image_path in tqdm(image_files, desc="处理图像"):
        try:
            # 获取相对路径作为image_id
            image_id = os.path.relpath(image_path, IMAGE_DIR)
            
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                inputs = processor(img, return_tensors="pt").to(device)
                output = model.generate(**inputs)
                caption = processor.decode(output[0], skip_special_tokens=True)
                
                writer.writerow([
                    image_id,
                    image_path,
                    caption.replace('\n', ' ').replace('\r', '')
                ])
                processed_count += 1
                
        except Exception as e:
            error_count += 1
            error_msg = f"处理失败: {str(e)}"
            writer.writerow([image_path, image_path, error_msg])
            with open("processing_errors.log", "a") as log_file:
                log_file.write(f"{image_path}\t{error_msg}\n")

print("\n处理结果:")
print(f"成功处理: {processed_count}")
print(f"失败数量: {error_count}")
print(f"结果已保存到: {OUTPUT_CSV}")