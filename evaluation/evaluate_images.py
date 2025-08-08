import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from alexnet_metrics import calculate_alexnet_similarity
from fid_score import calculate_fid


def evaluate_generated_images(real_dir, gen_dir):
    # 收集图像路径
    real_images = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith('.jpg')])
    gen_images = sorted([os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith('.png')])

    # 确保数量匹配
    n = min(len(real_images), len(gen_images))
    real_images = real_images[:n]
    gen_images = gen_images[:n]

    print(f"Evaluating {n} image pairs")

    # 计算SSIM
    ssim_values = []
    for real_path, gen_path in tqdm(zip(real_images, gen_images), desc="Calculating SSIM"):
        img_real = np.array(Image.open(real_path).convert("RGB"))
        img_gen = np.array(Image.open(gen_path).convert("RGB"))

        # 调整大小
        if img_real.shape != img_gen.shape:
            img_gen = np.array(Image.fromarray(img_gen).resize(
                (img_real.shape[1], img_real.shape[0]),
                Image.LANCZOS
            )

            # 计算SSIM
            ssim_val = ssim(
                img_real, img_gen,
                channel_axis=2,
                data_range=255,
                win_size=11,
                gaussian_weights=True
            )
            ssim_values.append(ssim_val)

            # 计算FID
            fid_value = calculate_fid(real_dir, gen_dir)

            # 计算AlexNet相似度
            alexnet2_sim, alexnet5_sim = calculate_alexnet_similarity(real_images, gen_images)

            # 汇总结果
            results = {
                "n_images": n,
                "ssim_mean": np.mean(ssim_values),
                "ssim_std": np.std(ssim_values),
                "fid": fid_value,
                "alexnet2_similarity": alexnet2_sim,
                "alexnet5_similarity": alexnet5_sim
            }

            # 打印结果
            print("\nEvaluation Results:")
            print("=" * 50)
            print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
            print(f"FID: {results['fid']:.2f}")
            print(f"AlexNet2 Similarity: {results['alexnet2_similarity']:.4f}")
            print(f"AlexNet5 Similarity: {results['alexnet5_similarity']:.4f}")
            print("=" * 50)

            # 保存结果
            with open("evaluation_results.json", "w") as f:
                json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    real_dir = "data/test_images"  # 真实测试图像
    gen_dir = "results/generated_images/sub-01"  # 生成的图像
    evaluate_generated_images(real_dir, gen_dir)