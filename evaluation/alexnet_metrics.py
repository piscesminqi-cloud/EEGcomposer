import torch
import torch.nn as nn
from torchvision.models import alexnet
from torchvision import transforms
from tqdm import tqdm


class AlexNetFeatureExtractor(nn.Module):
    def __init__(self, layer_index):
        super().__init__()
        self.alexnet = alexnet(pretrained=True).features
        self.layer_index = layer_index
        self.features = []

        # 注册钩子获取指定层输出
        for i, layer in enumerate(self.alexnet):
            if i == layer_index:
                layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features.append(output)

    def forward(self, x):
        self.features = []
        _ = self.alexnet(x)
        return self.features[0]


def calculate_alexnet_similarity(real_paths, gen_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建特征提取器
    alexnet2 = AlexNetFeatureExtractor(layer_index=2).to(device).eval()
    alexnet5 = AlexNetFeatureExtractor(layer_index=5).to(device).eval()

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 提取特征
    features_real2, features_gen2 = [], []
    features_real5, features_gen5 = [], []

    for real_path, gen_path in tqdm(zip(real_paths, gen_paths), desc="Extracting AlexNet features"):
        # 处理真实图像
        img_real = Image.open(real_path).convert("RGB")
        img_real = preprocess(img_real).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_real2 = alexnet2(img_real).flatten(1)
            feat_real5 = alexnet5(img_real).flatten(1)

        # 处理生成图像
        img_gen = Image.open(gen_path).convert("RGB")
        img_gen = preprocess(img_gen).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_gen2 = alexnet2(img_gen).flatten(1)
            feat_gen5 = alexnet5(img_gen).flatten(1)

        # 保存特征
        features_real2.append(feat_real2)
        features_gen2.append(feat_gen2)
        features_real5.append(feat_real5)
        features_gen5.append(feat_gen5)

    # 计算相似度（余弦相似度）
    sim2 = cosine_similarity(
        torch.cat(features_real2),
        torch.cat(features_gen2)
    )

    sim5 = cosine_similarity(
        torch.cat(features_real5),
        torch.cat(features_gen5)
    )

    return sim2.mean().item(), sim5.mean().item()


def cosine_similarity(a, b):
    # a和b形状: (n_samples, feature_dim)
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)).diag()