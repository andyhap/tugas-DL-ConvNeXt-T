"""Model ConvNeXt-Tiny untuk Imagenette 160 px, kompatibel dengan Colab."""
import torch.nn as nn
from torchvision.models import convnext_tiny

def build_model(num_classes: int = 10) -> nn.Module:
    """Input (B, 3, 160, 160), output (B, num_classes)."""
    model = convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model
