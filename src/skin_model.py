"""
Skin Disease Detection Model Builder (PyTorch)
===============================================
Builds an EfficientNet-B0-based classifier for 7 skin disease classes
(matching the HAM10000 dermoscopy dataset categories) and saves it
as 'skin_disease_model.pth'.

Classes:
  0 — Actinic Keratosis (akiec)
  1 — Basal Cell Carcinoma (bcc)
  2 — Benign Keratosis (bkl)
  3 — Dermatofibroma (df)
  4 — Melanoma (mel)
  5 — Melanocytic Nevi (nv)
  6 — Vascular Lesions (vasc)

Usage:
  python skin_model.py          # Creates skin_disease_model.pth in this directory
"""

import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ── Disease class metadata ──────────────────────────────────────────────── #
SKIN_CLASSES = [
    "Actinic Keratosis",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions",
]

NUM_CLASSES = len(SKIN_CLASSES)
INPUT_SIZE = 224  # EfficientNet-B0 default


def build_model() -> nn.Module:
    """
    Build EfficientNet-B0 with a custom classification head for skin diseases.

    The backbone is pre-trained on ImageNet. We replace the final classifier
    with a new head for 7-class skin disease classification.
    """
    # Load EfficientNet-B0 with ImageNet weights
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    # Freeze backbone for transfer learning
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES),
    )

    return model


def save_model(model: nn.Module, path: str):
    """Save the full model state dict + metadata."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": SKIN_CLASSES,
            "num_classes": NUM_CLASSES,
            "input_size": INPUT_SIZE,
            "backbone": "efficientnet_b0",
        },
        path,
    )


def load_skin_model(path: str) -> nn.Module:
    """Load a saved skin disease model from disk."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_and_save_model() -> str:
    """Build, save, and return the model path."""
    print("🔬 Building Skin Disease Detection Model...")
    print("   Backbone:  EfficientNet-B0 (ImageNet pretrained)")
    print("   Framework: PyTorch")
    print(f"   Input:     {INPUT_SIZE} × {INPUT_SIZE} × 3")
    print(f"   Classes:   {NUM_CLASSES} (HAM10000)")
    print()

    model = build_model()

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable params: {trainable:,}")
    print(f"   Frozen params:    {total_params - trainable:,}")

    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "skin_disease_model.pth",
    )
    save_model(model, save_path)

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n✅ Model saved to: {save_path}")
    print(f"   File size: {size_mb:.1f} MB")
    print()
    print("⚠️  NOTE: This model has a randomly initialized classification head.")
    print("   For accurate predictions, train it on the HAM10000 dataset.")

    return save_path


if __name__ == "__main__":
    build_and_save_model()
