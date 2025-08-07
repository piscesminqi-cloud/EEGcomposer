import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import logging
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
import numpy as np 
from ComposerUnet import ComposerUNet


class LocalConditionProj(nn.Module):
    def __init__(self):
        super().__init__()
        # 多条件视觉特征提取
        self.condition_convs = nn.ModuleDict({
            'sketch': nn.Sequential(nn.Identity()),
            'instance': nn.Sequential(nn.Identity()),
            'depth': nn.Sequential(nn.Identity()),
            'intensity': nn.Sequential(nn.Identity())
        })

    def dropout_conditions(self, sketch, instance, depth, intensity):
        """
        对四个条件张量应用自定义的 Dropout 策略，并返回新的张量列表 [sketch', instance', depth', intensity']。
        - sketch、instance、depth 独立以 0.5 概率丢弃
        - intensity 以 0.7 概率丢弃
        - 0.1 概率丢弃所有条件，0.1 概率保留所有条件
        """
        device = sketch.device  # 确保在相同设备上生成随机数
        rand = torch.rand(1, device=device).item()  # 随机判断是否全丢弃或全保留

        if rand < 0.1:
            # 以0.1概率丢弃所有条件
            return [
                torch.zeros_like(sketch),
                torch.zeros_like(instance),
                torch.zeros_like(depth),
                torch.zeros_like(intensity)
            ]
        elif rand < 0.2:
            # 以0.1概率保留所有条件
            return [
                sketch.clone(),
                instance.clone(),
                depth.clone(),
                intensity.clone()
            ]
        else:
            # 其他情况下独立决策是否丢弃每个条件
            drop_sketch = torch.rand(1, device=device).item() < 0.5
            drop_instance = torch.rand(1, device=device).item() < 0.5
            drop_depth = torch.rand(1, device=device).item() < 0.5
            drop_intensity = torch.rand(1, device=device).item() < 0.7

            # 如果标志为 True 则置零，否则返回原张量的克隆
            new_sketch = torch.zeros_like(sketch) if drop_sketch else sketch.clone()
            new_instance = torch.zeros_like(instance) if drop_instance else instance.clone()
            new_depth = torch.zeros_like(depth) if drop_depth else depth.clone()
            new_intensity = torch.zeros_like(intensity) if drop_intensity else intensity.clone()

            return [new_sketch, new_instance, new_depth, new_intensity]

    def forward(self, sketch, instance, depth, intensity):
        # 处理各条件特征
        condition_features = []
        for name in ['sketch', 'instance', 'depth', 'intensity']:
            conv = self.condition_convs[name]
            input_tensor = locals()[name]  # 获取对应名称的输入张量
            feat = conv(input_tensor)
            condition_features.append(feat)

        new_condition_features = self.dropout_conditions(*condition_features)
        return new_condition_features


class ComposerStableDiffusionPipeline(StableDiffusionPipeline):
    """
    Custom Stable Diffusion pipeline that integrates a ConditionUNet combining CLIP text features
    with additional local conditions (color, sketch, instance, depth, intensity).
    """

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: ComposerUNet,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPImageProcessor,
            image_encoder: CLIPVisionModelWithProjection,  # 确保使用正确的类型
            requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker
        )
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder
        )
        self.logger = logging.get_logger(__name__)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

        self.clip_image_time_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.SiLU(),
            nn.Linear(256, 320)
        )
        self.clip_text_time_proj = nn.Sequential(
            nn.Conv1d(77, 64, 3, padding=1, stride=1),
            nn.Conv1d(64, 32, 3, padding=1, stride=1),
            nn.Conv1d(32, 16, 3, padding=1, stride=1),
            nn.Conv1d(16, 1, 3, padding=1, stride=1),
            nn.Linear(768, 256),
            nn.SiLU(),
            nn.Linear(256, 320)
        )

        # 视觉条件处理
        self.local_condition_proj = LocalConditionProj()
    
    @torch.no_grad()
    def __call__(
            self,
            clip_image_embeds: np.ndarray,  # (batch, 768)
            text_embeddings: np.ndarray,    # (batch, 768)
            color: np.ndarray,              # (batch, 3, 64, 64)
            sketch: np.ndarray,             # (batch, 4, 64, 64)
            depth: np.ndarray,             # (batch, 4, 64, 64)
            intensity: np.ndarray,         # (batch, 4, 64, 64)
            instance: np.ndarray,         # (batch, 4, 64, 64)
            guidance_scale: float = 7.5,
            num_inference_steps: int = 50,
            height: int = 512,  # 添加高度参数
            width: int = 512    # 添加宽度参数
    ):
        """
        使用预计算特征生成图像:
        - clip_image_embeds: 预计算的CLIP图像嵌入 (batch_size, 768)
        - text_embeddings: 预计算的文本嵌入 (batch_size, 77, 768),但实际传入的是（batch_size, 768），那么现在需要填充到（batch_size, 77, 768）后面的76个768都填充0
        - color: 颜色张量 (batch_size, 3, 64, 64)
        - sketch, instance, depth, intensity: 视觉条件张量 (batch_size, 4, 64, 64)
        """
        # 获取设备信息
        device = self.device
        dtype = self.text_encoder.dtype
        self.local_condition_proj.eval()

        # 确定batch size
        batch_size = clip_image_embeds.shape[0]

        # 准备初始噪声
        latents = torch.randn(
            (batch_size, 4, height // 8, width // 8),
            device=device,
            dtype=dtype,
        )
        latents = latents * self.scheduler.init_noise_sigma

        # 设置时间步
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # 确保所有张量在正确设备和数据类型上
        clip_image_embeds = torch.from_numpy(clip_image_embeds).to(device=device, dtype=dtype)
        clip_image_embeds = torch.unsqueeze(clip_image_embeds, 1)
        text_embeddings = torch.from_numpy(text_embeddings).to(device=device, dtype=dtype)
        color = torch.from_numpy(color).to(device=device, dtype=dtype)
        sketch = torch.from_numpy(sketch).to(device=device, dtype=dtype)
        instance = torch.from_numpy(instance).to(device=device, dtype=dtype)
        depth = torch.from_numpy(depth).to(device=device, dtype=dtype)
        intensity = torch.from_numpy(intensity).to(device=device, dtype=dtype)
        self.clip_image_time_proj = self.clip_image_time_proj.to(device=device, dtype=dtype)
        self.clip_text_time_proj = self.clip_text_time_proj.to(device=device, dtype=dtype)
        if text_embeddings.dim() == 2 and text_embeddings.shape[1] == 768:
            text_embeddings_padded = torch.zeros((batch_size, 77, 768), device=text_embeddings.device, dtype=text_embeddings.dtype)
            text_embeddings_padded[:, 0, :] = text_embeddings
            text_embeddings = text_embeddings_padded


        # 扩散过程
        for t in self.scheduler.timesteps:
            # 扩展latents用于CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # 准备时间条件
            time_cond_pixel_values = self.clip_image_time_proj(clip_image_embeds).view(batch_size, 320)
            time_cond_prompt = self.clip_text_time_proj(text_embeddings).view(batch_size, 320)
            time_cond = time_cond_pixel_values + time_cond_prompt
            time_cond = torch.cat([time_cond] * 2, dim=0)  # 扩展时间条件以匹配latents

            # 准备CFG条件
            cond_pixel_values = torch.cat([clip_image_embeds] * 2)
            cond_prompt = torch.cat([text_embeddings] * 2)
            cond_color = torch.cat([color] * 2)

            # 准备视觉条件（应用dropout）
            cond_local_conditions = self.local_condition_proj(
                sketch=torch.cat([sketch] * 2),
                instance=torch.cat([instance] * 2),
                depth=torch.cat([depth] * 2),
                intensity=torch.cat([intensity] * 2)
            )

            # 合并全局条件
            cond_encoder_hidden_states = torch.cat([cond_pixel_values, cond_prompt], dim=1)

            # 合并局部条件
            local_conditions = torch.zeros_like(latent_model_input)
            for cond in cond_local_conditions:
                if cond is not None:
                    local_conditions += 0.25 * cond

            # 将视觉条件与latents合并
            latent_model_input = torch.cat([latent_model_input, local_conditions, cond_color], dim=1)

            # 预测噪声
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=cond_encoder_hidden_states,
                timestep_cond=time_cond
            ).sample

            # 应用分类器自由引导(CFG)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 计算前一个噪声样本 x_{t-1}
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 解码latents为图像
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample

        # 后处理
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        images = (image * 255).round().astype("uint8")
        images = [Image.fromarray(img) for img in images]

        return {"images": images}

    def save_custom_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)

        # 保存 diffusers 标准组件
        super().save_pretrained(save_directory)

        # 保存自定义 UNet 的权重
        torch.save(self.unet.state_dict(), os.path.join(save_directory, "composer_unet.pth"))

        # 保存其他自定义模块
        custom_modules = {
            "clip_image_time_proj": self.clip_image_time_proj.state_dict(),
            "clip_text_time_proj": self.clip_text_time_proj.state_dict(),
            "local_condition_proj": self.local_condition_proj.state_dict()
        }
        torch.save(custom_modules, os.path.join(save_directory, "custom_modules.pth"))

        print(f"[保存成功] Pipeline 与自定义模块保存到 {save_directory}")

    @classmethod
    def load_custom_pretrained(cls, load_directory: str or None,
                               base_unet_model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        自定义加载函数：加载 Pipeline + 注入 UNet state_dict。
        参数：
            - load_directory: 保存路径（包含 pipe + composer_unet.pth）
            - base_unet_model_id: 用于获取原始 SD1.5 的 unet config
        """

        # 加载其他组件
        if load_directory is None:
            load_directory = base_unet_model_id

        vae = AutoencoderKL.from_pretrained(load_directory, subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained(load_directory, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(load_directory, subfolder="tokenizer")
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(load_directory, subfolder="safety_checker")
        feature_extractor = CLIPImageProcessor.from_pretrained(load_directory, subfolder="feature_extractor")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        scheduler = DDPMScheduler.from_pretrained(load_directory, subfolder="scheduler")
        # 构造并加载自定义 UNet
        base_unet = UNet2DConditionModel.from_pretrained(load_directory, subfolder="unet")


        # 修改配置以匹配自定义UNet
        unet_config = base_unet.config
        unet_config["time_cond_proj_dim"] = 320  # 添加时间条件投影维度

        custom_unet = ComposerUNet(**unet_config)

        if load_directory == base_unet_model_id:
            # 修改输入通道数（原始为4，现在需要11）
            with torch.no_grad():
                custom_unet.conv_in = nn.Conv2d(4 + 4 + 3, unet_config["block_out_channels"][0], kernel_size=3,
                                                padding=1)
            # 复制原始权重（前4个通道）
            with torch.no_grad():
                original_weights = base_unet.conv_in.weight.data.clone()
                custom_unet.conv_in.weight[:, :4] = original_weights

            # 初始化新增通道的权重
            torch.nn.init.normal_(custom_unet.conv_in.weight[:, 4:], std=0.02)

            # 加载基础UNet权重（跳过conv_in）
            unet_state_dict = base_unet.state_dict()
            for name, param in unet_state_dict.items():
                if name not in ["conv_in.weight", "conv_in.bias"]:
                    if name in custom_unet.state_dict():
                        custom_unet.state_dict()[name].copy_(param)

        # 如果指定了自定义目录，加载额外权重
        if load_directory != base_unet_model_id:
            # 加载自定义UNet权重
            unet_path = os.path.join(load_directory, "composer_unet.pth")
            if os.path.exists(unet_path):
                custom_unet.load_state_dict(torch.load(unet_path, map_location="cpu"))

            # 加载其他自定义模块
            modules_path = os.path.join(load_directory, "custom_modules.pth")
            if os.path.exists(modules_path):
                custom_modules = torch.load(modules_path, map_location="cpu")
                # 将在管道创建后加载这些模块

        # 构造 Pipeline
        pipe = cls(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=custom_unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=True
        )

        # 加载其他自定义模块的权重
        if load_directory != base_unet_model_id and os.path.exists(modules_path):
            pipe.clip_image_time_proj.load_state_dict(custom_modules["clip_image_time_proj"])
            pipe.clip_text_time_proj.load_state_dict(custom_modules["clip_text_time_proj"])
            pipe.local_condition_proj.load_state_dict(custom_modules["local_condition_proj"])

        print(f"[加载成功] Pipeline 从 {load_directory} 加载完成")
        return pipe
