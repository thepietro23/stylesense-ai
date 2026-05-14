"""ResNet50-based image feature extractor.

Loads a pretrained ResNet50, removes the classification head, and exposes a
function that returns a 2048-dimensional embedding for a given image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_preprocess() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def build_augmentations() -> dict[str, transforms.Compose]:
    """Return a dict of named augmentation transforms used during training.

    Each transform maps a PIL image to a tensor ready for the ResNet50 forward
    pass. These transforms are applied only to training images to enlarge the
    effective dataset without leaking the test split.
    """
    norm = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return {
        "flip": transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), norm,
        ]),
        "jitter": transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                     saturation=0.3, hue=0.05),
            transforms.ToTensor(), norm,
        ]),
        "rotate": transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.RandomRotation(15),
            transforms.ToTensor(), norm,
        ]),
        "rrc": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(), norm,
        ]),
        "flip_jitter": transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(), norm,
        ]),
    }


@torch.inference_mode()
def extract_with_transform(image_paths, transform, model, device="cpu", batch_size=16):
    """Embed images using a specific transform (used for augmented training data)."""
    out = []
    batch = []
    for path in image_paths:
        with Image.open(path) as im:
            im = im.convert("RGB")
            t = transform(im)
        batch.append(t)
        if len(batch) == batch_size:
            out.append(model(torch.stack(batch).to(device)).cpu().numpy())
            batch = []
    if batch:
        out.append(model(torch.stack(batch).to(device)).cpu().numpy())
    return np.vstack(out)


def load_resnet50_extractor(device: str | torch.device = "cpu") -> nn.Module:
    """Load a pretrained ResNet50 with the final FC layer replaced by identity.

    The returned module produces a 2048-dimensional embedding per image.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


@torch.inference_mode()
def extract_embeddings(
    image_paths: Iterable[str | Path],
    model: nn.Module | None = None,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """Compute 2048-d embeddings for the given image paths.

    Returns an array of shape (N, 2048) in the order of the input paths.
    """
    if model is None:
        model = load_resnet50_extractor(device=device)

    preprocess = _build_preprocess()
    paths = [Path(p) for p in image_paths]
    embeddings: list[np.ndarray] = []

    batch: list[torch.Tensor] = []
    for path in paths:
        with Image.open(path) as im:
            im = im.convert("RGB")
            tensor = preprocess(im)
        batch.append(tensor)

        if len(batch) == batch_size:
            stacked = torch.stack(batch).to(device)
            output = model(stacked).cpu().numpy()
            embeddings.append(output)
            batch = []

    if batch:
        stacked = torch.stack(batch).to(device)
        output = model(stacked).cpu().numpy()
        embeddings.append(output)

    return np.vstack(embeddings)
