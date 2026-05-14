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
