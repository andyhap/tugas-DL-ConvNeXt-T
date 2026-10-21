import torch.nn as nn
from torchvision.models import convnext_tiny

def build_model(num_classes: int = 10) -> nn.Module:
    """
    ConvNeXt-Tiny untuk klasifikasi Imagenette 10 kelas.

    Input : (B, 3, 160, 160)
    Output: (B, 10)
    """
    model = convnext_tiny(weights=None)

    # Mengganti classifier bawaan ImageNet menjadi 10 kelas.
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    return model
