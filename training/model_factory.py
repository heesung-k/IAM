"""Factory helpers for constructing models used in the project."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision.models import resnet18, resnet50
import timm

from model import WideResNet


def _replace_first_conv(module: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if module.in_channels == in_channels:
        return module
    return nn.Conv2d(
        in_channels,
        module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        bias=module.bias is not None,
    )


def _replace_classifier(module: nn.Module, num_classes: int) -> nn.Module:
    if isinstance(module, nn.Linear) and module.out_features != num_classes:
        return nn.Linear(module.in_features, num_classes)
    return module


def build_model(args, num_classes: int, input_channels: int, device) -> nn.Module:
    model_name = args.model

    if model_name == "resnet-18":
        model = resnet18(weights=None)
        model.conv1 = _replace_first_conv(model.conv1, input_channels)
        model.fc = _replace_classifier(model.fc, num_classes)
    elif model_name == "resnet-50":
        model = resnet50(weights=None)
        model.conv1 = _replace_first_conv(model.conv1, input_channels)
        model.fc = _replace_classifier(model.fc, num_classes)
    elif model_name == "WideResNet":
        depth = getattr(args, "depth", 16)
        width_factor = getattr(args, "width_factor", 8)
        model = WideResNet(
            depth=depth,
            width_factor=width_factor,
            dropout=getattr(args, "dropout", 0.0),
            in_channels=input_channels,
            labels=num_classes,
        )
    elif model_name == "vit-s":
        # ViT-S/32 configuration
        if args.dataset in ["cifar10", "cifar100", "mnist", "fashion_mnist"]:
            image_size = 32
            model = timm.create_model(
                'vit_small_patch16_32', 
                pretrained=False, 
                num_classes=num_classes,
                img_size=image_size,
                drop_path_rate=0.1,  # Stochastic Depth 적용
                in_chans=input_channels # 입력 채널 변경을 여기서 바로 지원합니다!
                )
        else:
            image_size = 224
            
            model = timm.create_model(
                'vit_small_patch32_224', 
                pretrained=False, 
                num_classes=num_classes,
                img_size=image_size,
                drop_path_rate=0.1,
                in_chans=input_channels 
                )
    elif model_name == "vit-t":
        # ViT-T/32 configuration
        if args.dataset in ["cifar10", "cifar100", "mnist", "fashion_mnist"]:
            image_size = 32
            model = timm.create_model(
                'vit_tiny_patch16_224',  # 베이스 모델 이름
                pretrained=False,        # From scratch 학습
                num_classes=num_classes,          # CIFAR-10 클래스 수
                img_size=image_size,             # 이미지 크기 강제 설정 (32x32)
                patch_size=4,            # 패치 크기 변경 (16 -> 4)
                embed_layer=timm.layers.PatchEmbed, 
                dynamic_img_size=True,
                drop_path_rate=0.1, 
                in_chans=input_channels 
                )
        else:
            raise ValueError(f"Unsupported dataset for ViT-T: {args.dataset}")
    elif model_name == "mlp_mixer-t":
        # ViT-T/32 configuration
        if args.dataset in ["cifar10", "cifar100", "mnist", "fashion_mnist"]:
            image_size = 32
            model = timm.models.mlp_mixer.MlpMixer(
                num_classes=num_classes,          # CIFAR-10 클래스 수
                img_size=image_size,             # 이미지 크기 강제 설정 (32x32)
                patch_size=4,            # 패치 크기 변경 (16 -> 4)
                # 구조 설정
                embed_dim=256,
                num_blocks=10,
                mlp_ratio=(0.5, 4.0),
                in_chans=input_channels,
                drop_path_rate=0.1, 
                )
        else:
            raise ValueError(f"Unsupported dataset for ViT-T: {args.dataset}")
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    return model.to(device)
