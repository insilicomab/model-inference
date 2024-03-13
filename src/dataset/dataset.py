import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from src.dataset.transformation import TestTransforms


class InferenceImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_path_list: list,
        transform: Compose,
    ) -> None:
        self.root = root
        self.image_path_list = image_path_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_path_list)

    def __getitem__(self, index) -> tuple[torch.Tensor, str, str]:
        image = Image.open(os.path.join(self.root, self.image_path_list[index]))
        image = self.transform(image)

        file_path = self.image_path_list[index]

        return image, file_path


def get_inference_dataloader(
    root: str, image_path_list: list, image_size: int
) -> DataLoader:

    # test dataset
    test_dataset = InferenceImageDataset(
        root=root,
        image_path_list=image_path_list,
        transform=TestTransforms(image_size=image_size),
    )

    # dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return test_dataloader
