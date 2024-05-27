import torch
from PIL import Image
from torchvision import transforms


class TestTransforms:
    def __init__(self, image_size: int, in_chas: int):
        if in_chas == 3:
            self.data_transform = transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        elif in_chas == 1:
            self.data_transform = transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

    def __call__(self, img: Image) -> torch.Tensor:
        return self.data_transform(img)


__all__ = ["TestTransforms"]
