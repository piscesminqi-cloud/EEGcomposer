import sklearn.linear_model as skl
from scipy.spatial.distance import correlation
import argparse

# 设置参数解析
parser = argparse.ArgumentParser(description='EEG to Multiple Latents Mapping')
parser.add_argument("-sub", "--sub", help="Subject Number (sub-01 to sub-10)", default='01')
parser.add_argument('-alpha', '--alpha', help='Alpha for Ridge Regression', default=1000, type=int)
args = parser.parse_args()
sub = args.sub
alpha = args.alpha

# -------------------- 1. 加载 EEG 数据 --------------------
eeg_train = np.load(f'data/sub-{int(sub):02d}/train_thingseeg2_avg.npy')
eeg_test = np.load(f'data/sub-{int(sub):02d}/test_thingseeg2_avg.npy')

# EEG 数据展平，适配回归模型输入（样本数×特征数）
eeg_train_flat = eeg_train.reshape(eeg_train.shape[0], -1)
eeg_test_flat = eeg_test.reshape(eeg_test.shape[0], -1)

# 计算 EEG 训练集的均值和标准差，用于标准化
norm_mean_train = np.mean(eeg_train_flat, axis=0)
norm_scale_train = np.std(eeg_train_flat, axis=0, ddof=1)
eeg_train_norm = (eeg_train_flat - norm_mean_train) / norm_scale_train
eeg_test_norm = (eeg_test_flat - norm_mean_train) / norm_scale_train

# -------------------- 2. 定义加载各类 latent 数据的函数 --------------------
def load_latent_data(latent_type, is_train=True):
    """
    加载不同类型 latent 数据，根据训练/测试集以及类型适配路径和形状
    :param latent_type: 区分 latent 类型，如 'clip_image'/'clip_text'/'color'/'vae_instance' 等
    :param is_train: 是否加载训练集数据
    :return: 加载并整理好的 latent 数据
    """
    if latent_type == 'clip_image':
        if is_train:
            path = f'features_train/all_image_features.npy'
            data = np.load(path)  # 形状(16540, 768)
        else:
            path = f'features_test/all_image_features.npy'
            data = np.load(path)  # 形状(200, 768)
        return data
    
    elif latent_type == 'clip_text':
        if is_train:
            path = f'features_train/all_text_features.npy'
            data = np.load(path)  # 形状(16540, 77, 768)
        else:
            path = f'features_test/all_text_features.npy'
            data = np.load(path)  # 形状(200, 77, 768)
        return data
    
    elif latent_type == 'color':
        if is_train:
            base_dir = 'color_features_train'
            num_samples = 16540
        else:
            base_dir = 'color_features_test'
            num_samples = 200
        
        data_list = []
        # 遍历所有类别目录（按数字顺序）
        category_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        for category_dir in category_dirs:
            category_path = os.path.join(base_dir, category_dir)
            # 获取该类别下所有npy文件并按文件名排序
            npy_files = sorted([f for f in os.listdir(category_path) if f.endswith('.npy')])
            for file_name in npy_files:
                file_path = os.path.join(category_path, file_name)
                data = np.load(file_path)
                data_flat = data.flatten()
                data_list.append(data_flat)
                
                # 达到所需样本数时停止
                if len(data_list) >= num_samples:
                    return np.array(data_list)
        
        return np.array(data_list)
    
    elif latent_type.startswith('vae_'):
        vae_subtype = latent_type.split('_')[1]  # 如 'instance'/'intensity'/'sketch'/'depth'
        if is_train:
            base_dir = f'condition_embeddings/train/{vae_subtype}'
            num_samples = 16540
        else:
            base_dir = f'condition_embeddings/test/{vae_subtype}'
            num_samples = 200
        
        data_list = []
        # 遍历所有类别目录（按数字顺序）
        category_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        for category_dir in category_dirs:
            category_path = os.path.join(base_dir, category_dir)
            # 获取该类别下所有npy文件并按文件名排序
            npy_files = sorted([f for f in os.listdir(category_path) if f.endswith('.npy')])
            for file_name in npy_files:
                file_path = os.path.join(category_path, file_name)
                data = np.load(file_path)
                data_list.append(data)
                
                # 达到所需样本数时停止
                if len(data_list) >= num_samples:
                    return np.array(data_list)
        
        return np.array(data_list)
    
    else:
        raise ValueError(f"不支持的 latent 类型: {latent_type}")

# -------------------- 3. 定义训练和保存模型的函数 --------------------
def train_and_save(latent_type, eeg_train, eeg_test, alpha, save_dir):
    """
    训练 EEG 到单个 latent 的回归模型并保存权重，同时计算评估指标
    :param latent_type: latent 类型标识
    :param eeg_train: 标准化后的训练集 EEG 数据
    :param eeg_test: 标准化后的测试集 EEG 数据
    :param alpha: Ridge 回归的正则化系数
    :param save_dir: 权重和结果保存目录
    """
    # 加载 latent 训练集和测试集数据
    train_latents = load_latent_data(latent_type, is_train=True)
    test_latents = load_latent_data(latent_type, is_train=False)
    
    print(f"\nLatent 类型: {latent_type}")
    print(f"训练集 latent 形状: {train_latents.shape}, 测试集 latent 形状: {test_latents.shape}")
    
    if latent_type == 'clip_text':
        num_samples, num_token, num_dim = train_latents.shape
        reg_w = np.zeros((num_token, num_dim, eeg_train.shape[1])).astype(np.float32)
        reg_b = np.zeros((num_token, num_dim)).astype(np.float32)
        pred_latents = np.zeros_like(test_latents)
        for i in range(num_token):
            reg = skl.Ridge(alpha=alpha, max_iter=50000, fit_intercept=True)
            reg.fit(eeg_train, train_latents[:, i])
            reg_w[i] = reg.coef_
            reg_b[i] = reg.intercept_
            
            pred_test_latent = reg.predict(eeg_test)
            std_norm_test_latent = (pred_test_latent - np.mean(pred_test_latent, axis=0)) / np.std(pred_test_latent, axis=0)
            pred_latents[:, i] = std_norm_test_latent * np.std(train_latents[:, i], axis=0) + np.mean(train_latents[:, i], axis=0)

            # 计算评估指标
            # 计算 R² 分数
            r2_score = reg.score(eeg_test, test_latents[:, i])
            
            # 计算平均欧氏距离
            euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_latents[:, i], test_latents[:, i])])
            avg_euclidean = euclidean_distances.mean()
            
            # 计算平均相关性（基于 correlation 距离转换）
            correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_latents[:, i], test_latents[:, i])])
            avg_correlation = (1 - correlation_distances).mean()
            
            print(f"Token {i} - R² 分数: {r2_score:.4f}, 平均欧氏距离: {avg_euclidean:.4f}, 平均相关性: {avg_correlation:.4f}")
    else:
        # 训练 Ridge 回归模型
        reg = skl.Ridge(alpha=alpha, max_iter=50000, fit_intercept=True)
        reg.fit(eeg_train, train_latents)
        print(f"{latent_type} 模型训练完成")
        
        # 预测并计算评估指标
        pred_latents = reg.predict(eeg_test)
        # std_norm_latent = (pred_latents - np.mean(pred_latents, axis=0)) / np.std(pred_latents, axis=0)
        # pred_latents = std_norm_latent * np.std(train_latents, axis=0) + np.mean(train_latents, axis=0)
        
        # 计算 R² 分数
        r2_score = reg.score(eeg_test, test_latents)
        
        # 计算平均欧氏距离
        euclidean_distances = np.array([np.linalg.norm(u - v) for u, v in zip(pred_latents, test_latents)])
        avg_euclidean = euclidean_distances.mean()
        
        # 计算平均相关性（基于 correlation 距离转换）
        correlation_distances = np.array([correlation(u, v) for u, v in zip(pred_latents, test_latents)])
        avg_correlation = (1 - correlation_distances).mean()
        
        print(f"R² 分数: {r2_score:.4f}, 平均欧氏距离: {avg_euclidean:.4f}, 平均相关性: {avg_correlation:.4f}")

    # 保存模型权重
    if latent_type == 'clip_text':
        weights_dict = {
            'weight': reg_w,
            'bias': reg_b,
            'latent_type': latent_type
        }
    else:
        weights_dict = {
            'weight': reg.coef_,
            'bias': reg.intercept_,
            'latent_type': latent_type
        }
    os.makedirs(save_dir, exist_ok=True)
    weight_save_path = f"{save_dir}/regress_{latent_type}_weights_sub-{sub}.pkl"
    with open(weight_save_path, "wb") as f:
        pickle.dump(weights_dict, f)
    print(f"{latent_type} 模型权重已保存至: {weight_save_path}")

    # 保存预测结果
    save_dir_pred = f'cache/predicted_embeddings/sub-{sub}/'
    os.makedirs(save_dir_pred, exist_ok=True)
    pred_save_path = f"{save_dir_pred}/pred_{latent_type}_sub-{sub}.npy"
    np.save(pred_save_path, pred_latents)
    print(f"{latent_type} 预测结果已保存至: {pred_save_path}")

# -------------------- 4. 执行各类 latent 的训练与保存 --------------------
save_base_dir = f'cache/regression_weights/sub-{sub}/'
# 依次处理 7 类 latent
latent_types = [
    'clip_image',
    'clip_text',
    'color',
    'vae_instance',
    'vae_intensity',
    'vae_sketch',
    'vae_depth'
]

for lt in latent_types:
    train_and_save(lt, eeg_train_norm, eeg_test_norm, alpha, save_base_dir)