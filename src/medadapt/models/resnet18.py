"""ResNet-18 classifier for binary BreakHis experiments."""
from __future__ import annotations


def build_resnet18(
    pretrained: bool = True,
    dropout: float = 0.2,
    num_classes: int = 2,
):
    """Create a TorchVision ResNet-18 with a replaceable classification head.

    Pretrained weights are ImageNet-1K weights from TorchVision and may need
    to be downloaded on first use. Set pretrained=False for offline smoke tests.
    """
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in the interval [0, 1).")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    try:
        from torch import nn
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:  # pragma: no cover - depends on training extra
        raise ImportError(
            "PyTorch and TorchVision are required. "
            "Install them with: pip install -e '.[training]'"
        ) from exc

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def count_parameters(model) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable)}
