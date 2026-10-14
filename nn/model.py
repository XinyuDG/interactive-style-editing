import os
from pathlib import Path

try:
    from config import HF_HOME, HF_HUB_CACHE
except ModuleNotFoundError:
    from .config import HF_HOME, HF_HUB_CACHE

# ============================================================
# 缓存目录：必须放在 torchvision 导入之前
# ============================================================
CACHE_ROOT = Path(os.getenv("TORCH_HOME", HF_HOME)).resolve()
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
Path(HF_HUB_CACHE).mkdir(parents=True, exist_ok=True)

# TorchVision / PyTorch Hub 预训练权重缓存
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT))

# 可选：顺手统一 HF 根目录
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights
import torchvision.transforms as transforms
from PIL import Image


class NeuralStyleTransfer:
    def __init__(self):
        self.content_layers = ["conv_4"]
        self.style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.device = torch.device("cpu")

        self.model, self.content_layer_indices, self.style_layer_indices = (
            self._build_model()
        )
        self.model.to(self.device)

    def _build_model(self):
        # 新写法：不要再用 pretrained=True
        vgg = (
            models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(self.device).eval()
        )

        for param in vgg.parameters():
            param.requires_grad = False

        content_layer_indices = []
        style_layer_indices = []
        model = nn.Sequential()

        i = 0
        for layer in vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f"conv_{i}"
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{i}"
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{i}"
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"bn_{i}"
            else:
                raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

            model.add_module(name, layer)

            if name in self.content_layers:
                content_layer_indices.append(len(model) - 1)
            if name in self.style_layers:
                style_layer_indices.append(len(model) - 1)

        return model, content_layer_indices, style_layer_indices

    def load_img(self, path_or_img):
        max_dim = 512

        if isinstance(path_or_img, str):
            image = Image.open(path_or_img).convert("RGB")
        elif isinstance(path_or_img, Image.Image):
            image = path_or_img.convert("RGB")
        else:
            raise TypeError("load_img 期望输入是文件路径 (str) 或 PIL Image 对象")

        transform = transforms.Compose(
            [transforms.Resize(max_dim), transforms.ToTensor()]
        )
        image = transform(image).unsqueeze(0)
        return image.to(self.device)

    def gram_matrix(self, input_tensor):
        b, c, h, w = input_tensor.size()
        features = input_tensor.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G / (c * h * w)

    def compute_loss(
        self,
        style_targets,
        style_outputs,
        content_targets,
        content_outputs,
        style_weight,
        content_weight,
    ):
        style_loss = 0
        content_loss = 0

        for st, so in zip(style_targets, style_outputs):
            style_loss += torch.mean((self.gram_matrix(so) - self.gram_matrix(st)) ** 2)

        for ct, co in zip(content_targets, content_outputs):
            content_loss += torch.mean((co - ct) ** 2)

        style_loss *= style_weight / self.num_style_layers
        content_loss *= content_weight / self.num_content_layers
        total_loss = style_loss + content_loss
        return total_loss

    def extract_features(self, x):
        style_features = []
        content_features = []
        for idx, layer in enumerate(self.model):
            x = layer(x)
            if idx in self.style_layer_indices:
                style_features.append(x)
            if idx in self.content_layer_indices:
                content_features.append(x)
        return style_features, content_features

    def run_style_transfer(
        self,
        content_path,
        style_path,
        num_iterations=30,
        style_weight=1e6,
        content_weight=1,
        progress_callback=None,
    ):
        content_image = self.load_img(content_path)
        style_image = self.load_img(style_path)

        with torch.no_grad():
            style_features, _ = self.extract_features(style_image)
            _, content_features = self.extract_features(content_image)

        style_targets = [sf.clone() for sf in style_features]
        content_targets = [cf.clone() for cf in content_features]

        generated_image = content_image.clone().requires_grad_(True).to(self.device)
        optimizer = torch.optim.Adam([generated_image], lr=0.02)

        for i in range(num_iterations):
            optimizer.zero_grad()
            style_outputs, content_outputs = self.extract_features(generated_image)

            loss = self.compute_loss(
                style_targets,
                style_outputs,
                content_targets,
                content_outputs,
                style_weight,
                content_weight,
            )

            style_loss = 0
            content_loss = 0

            for st, so in zip(style_targets, style_outputs):
                style_loss += torch.mean(
                    (self.gram_matrix(so) - self.gram_matrix(st)) ** 2
                )

            for ct, co in zip(content_targets, content_outputs):
                content_loss += torch.mean((co - ct) ** 2)

            style_loss *= style_weight / self.num_style_layers
            content_loss *= content_weight / self.num_content_layers
            total_loss = style_loss + content_loss

            loss.backward()
            optimizer.step()
            generated_image.data.clamp_(0, 1)

            if progress_callback:
                current_progress_ratio = (i + 1) / num_iterations
                progress_callback(current_progress_ratio)

            print(f"Iteration {i + 1}/{num_iterations}:")
            print(f"  Content Loss: {content_loss.item()}")
            print(f"  Style Loss: {style_loss.item()}")
            print(f"  Total Loss: {total_loss.item()}")

        return generated_image.detach()
