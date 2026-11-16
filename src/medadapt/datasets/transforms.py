"""Model-independent image transforms for BreakHis experiments."""
from __future__ import annotations

try:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
except ImportError as exc:  # pragma: no cover - depends on optional training extra
    raise ImportError(
        "TorchVision is required for BreakHis transforms. "
        "Install it with: pip install -e '.[training]'"
    ) from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PadToSquare:
    """Pad an RGB PIL image to a square without cropping or stretching.

    The fill colour defaults to the ImageNet mean, which becomes approximately
    zero after the corresponding normalization.
    """

    def __init__(self, fill: tuple[int, int, int] = (124, 116, 104)) -> None:
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        if width == height:
            return image

        if width > height:
            difference = width - height
            top = difference // 2
            bottom = difference - top
            padding = (0, top, 0, bottom)
        else:
            difference = height - width
            left = difference // 2
            right = difference - left
            padding = (left, 0, right, 0)

        return TF.pad(image, padding, fill=self.fill)


def build_transforms(
    image_size: int = 224,
    train: bool = False,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
):
    """Build deterministic evaluation or training-only augmentation transforms.

    The shorter image edge is resized to image_size. The longer edge is then
    padded to a square using the normalization mean, preserving the full field
    of view without aspect-ratio distortion or a centre crop.

    Training augmentation uses horizontal/vertical flips and right-angle
    rotations. No colour jitter is applied by default so that stain colour
    variation is not artificially changed in the initial baseline.
    """
    if image_size <= 0:
        raise ValueError("image_size must be a positive integer.")
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("mean and std must contain three valid channel values.")

    fill = tuple(round(channel * 255) for channel in mean)
    operations = [
        transforms.Resize(image_size, interpolation=InterpolationMode.BILINEAR),
        PadToSquare(fill=fill),
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BILINEAR,
        ),
    ]

    if train:
        rotations = [
            transforms.RandomRotation(
                (angle, angle),
                interpolation=InterpolationMode.BILINEAR,
                fill=fill,
            )
            for angle in (0, 90, 180, 270)
        ]
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomChoice(rotations),
            ]
        )

    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transforms.Compose(operations)
