from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Compose
from tqdm import tqdm

from src.dataset.dataset import InferenceImageDataset
from src.model.evidences import calculate_uncertainty
from src.model.model import get_model

CUDA_IS_AVAILABLE = torch.cuda.is_available()


def load_model_weights(
    config: DictConfig, download_root: str, model_path: str
) -> torch.nn.Module:
    _model_state_dict = torch.load(Path(download_root) / Path(model_path))["state_dict"]
    model_state_dict = {
        k.replace("model.", ""): v for k, v in _model_state_dict.items()
    }
    model = get_model(config)
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    return model


def inference(
    dataloader: DataLoader, model: torch.nn.Module, int_to_label: dict
) -> pd.DataFrame:
    if CUDA_IS_AVAILABLE:
        model.cuda()
    paths, preds, uncertainties = [], [], []
    with torch.no_grad():
        for image, path in tqdm(dataloader):
            if CUDA_IS_AVAILABLE:
                image = image.cuda()
            logits = model(image)
            pred = logits.argmax(dim=1)
            pred = pred.cpu().detach().numpy()
            pred = [int_to_label[i] for i in pred]
            uncertainty = calculate_uncertainty(logits, num_classes=len(int_to_label))
            paths.extend(path)
            preds.extend(pred)
            uncertainties.extend(uncertainty)
    df = pd.DataFrame(
        {
            "image_path": paths,
            "pred": preds,
            "uncertainty": uncertainties,
        }
    )
    return df


class TestTimeAugmentationInference:
    def __init__(self) -> None:
        pass

    def setup_tta_transforms(self, image_size: int) -> list[Compose]:
        tta_transforms = [
            transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.functional.hflip,
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            transforms.Compose(
                [
                    transforms.CenterCrop((image_size, image_size)),
                    transforms.functional.vflip,
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        ]
        return tta_transforms

    def _get_inference_dataloader(
        self,
        root: str,
        image_path_list: list,
        transform: Compose,
    ) -> DataLoader:
        # test dataset
        test_dataset = InferenceImageDataset(
            root=root,
            image_path_list=image_path_list,
            transform=transform,
        )
        # dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        return test_dataloader

    def _inference(
        self, dataloader: DataLoader, model: torch.nn.Module, int_to_label: dict
    ) -> tuple[np.ndarray, list]:
        if CUDA_IS_AVAILABLE:
            model.cuda()
        paths, preds, uncertainties = [], [], []
        with torch.no_grad():
            for image, path in tqdm(dataloader):
                if CUDA_IS_AVAILABLE:
                    image = image.cuda()
                logits = model(image)
                preds_proba = F.softmax(logits, dim=1)
                uncertainty = calculate_uncertainty(
                    logits, num_classes=len(int_to_label)
                )
                preds.append(preds_proba.cpu().numpy())
                paths.extend(path)
                uncertainties.extend(uncertainty)
        preds = np.concatenate(preds)
        return preds, paths, uncertainties

    def inference_tta(
        self,
        root: str,
        image_path_list: list,
        model: torch.nn.Module,
        int_to_label: dict,
        tta_transforms: list[Compose],
    ) -> pd.DataFrame:
        with torch.no_grad():
            preds_for_tta = []
            uncertainties_for_tta = []
            for transform in tqdm(tta_transforms):
                test_dataloader = self._get_inference_dataloader(
                    root=root, image_path_list=image_path_list, transform=transform
                )
                preds, paths, uncertainties = self._inference(
                    dataloader=test_dataloader, model=model, int_to_label=int_to_label
                )
                preds_for_tta.append(preds)
                uncertainties_for_tta.append(uncertainties)

            tta_preds_mean = np.mean(preds_for_tta, axis=0)
            tta_preds = np.argmax(tta_preds_mean, axis=1)
            tta_preds_label = [int_to_label[i] for i in tta_preds]

            tta_uncertainties_mean = np.mean(uncertainties_for_tta, axis=0)

        df = pd.DataFrame(
            {
                "image_path": paths,
                "pred": tta_preds_label,
                "uncertainty": tta_uncertainties_mean,
            }
        )

        return df
