import torch
import torch.nn as nn
from torchvision import models

def build_medical_model(backbone: str = "vgg16", num_classes: int = 4, pretrained: bool = False):
    """
    Constructs transfer learning diagnostic models with custom dense classification heads.
    Supports Brain Tumor (4 classes) and Skin Disease (7 classes).
    """
    if backbone == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    elif backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model
