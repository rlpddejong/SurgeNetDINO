# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Adapted from EoMT (https://github.com/tue-mps/eomt).
# Modified by Ronald de Jong, 2025.
# ---------------------------------------------------------------


import zipfile
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms


CLASS_NAMES = {
    0: "Background",
    1: "Hook",
    2: "Forceps",
    3: "Suction & irrigation",
    4: "Vessel sealer",
    5: "Right lung",
    6: "Azygos vein & Vena cava",
    7: "Aorta",
    8: "Pericardium",
    9: "Airways",
    10: "Vagal nerves",
    11: "Recurrent laryngeal nerves",
    12: "Thoracic duct",
}

CLASS_MAPPING = {
    0: 0,   # Background
    1: 1,   # Hook
    2: 2,   # Forceps
    3: 3,   # Suction & irrigation
    4: 4,   # Vessel sealer
    5: 5,   # Right lung
    6: 6,   # Azygos vein
    7: 6,   # Vena cava
    8: 7,   # Aorta
    9: 8,   # Pericardium
    10: 9,  # Airways
    11: 10, # Vagal nerves
    12: 11, # Recurrent laryngeal nerves
    13: 12, # Thoracic duct incl. fat
    14: 0,  # Thoracic duct
    15: 0,  # Left pleura
    16: 0,  # Esophagus
    17: 0,  # Clip applier
    18: 0,  # Needle driver
    19: 0,  # Scissors
    20: 0,  # Stapler
}

COLOR_PALETTE = {
    0: (0, 0, 0),
    1: (100, 80, 0),
    2: (128, 0, 0),
    3: (0, 128, 0),
    4: (0, 255, 255),
    5: (160, 100, 160),
    6: (0, 0, 255),
    7: (255, 0, 0),
    8: (255, 0, 157),
    9: (255, 255, 255),
    10: (255, 255, 0),
    11: (139, 128, 0),
    12: (0, 255, 0),
}


class RAMIEDataset(torch.utils.data.Dataset):
    """One sample per clip: the alphabetically-last (= last/annotated) frame in the zip."""

    def __init__(
        self,
        video_dir: Path,
        mask_dir: Path,
        identifiers: set[str],
        target_parser: Callable,
        transforms: Optional[Callable] = None,
    ):
        self.mask_dir = mask_dir
        self.target_parser = target_parser
        self.transforms = transforms
        self.clips = [
            clip
            for clip in sorted(video_dir.glob("*.zip"))
            if clip.stem.split("_")[0] in identifiers
        ]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index: int):
        clip = self.clips[index]

        with zipfile.ZipFile(clip) as img_zip:
            frame = max(n for n in img_zip.namelist() if n.endswith(".png"))
            with img_zip.open(frame) as file:
                img = tv_tensors.Image(Image.open(file).convert("RGB"))
        with zipfile.ZipFile(self.mask_dir / clip.name) as target_zip:
            with target_zip.open(frame) as file:
                target = tv_tensors.Mask(Image.open(file), dtype=torch.long)

        if img.shape[-2:] != target.shape[-2:]:
            target = F.resize(
                target, list(img.shape[-2:]), interpolation=F.InterpolationMode.NEAREST
            )

        masks, labels, is_crowd = self.target_parser(target=target)

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels),
            "is_crowd": torch.tensor(is_crowd),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class RAMIESemantic(LightningDataModule):
    def __init__(
        self,
        path,
        split: int = 1,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (336, 336),
        num_classes: int = len(CLASS_NAMES),
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
        num_predict_frames: int = 25,
        num_video_clips: int = 5,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.split = split
        # Inference only (see inference_eomt.py); unused
        # during training, overridable via the inference CLI arguments.
        # num_predict_frames: how many of the last frames of each clip to predict.
        # num_video_clips: for the first k clips, predict *all* frames and write
        # an overlay video; the remaining clips only get num_predict_frames.
        self.num_predict_frames = num_predict_frames
        self.num_video_clips = num_video_clips
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, **kwargs):
        mapped = torch.zeros_like(target[0])
        for src, dst in CLASS_MAPPING.items():
            mapped[target[0] == src] = dst

        masks, labels = [], []
        for cls_id in mapped.unique().tolist():
            masks.append(mapped == cls_id)
            labels.append(cls_id)

        return masks, labels, [False for _ in range(len(masks))]

    def _split_file(self, stage: str) -> Path:
        return Path(self.path, "MTL_split", f"{stage}.split{self.split}.bundle")

    def _identifiers(self, stage: str) -> set[str]:
        return set(self._split_file(stage).read_text().split())

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        video_dir = Path(self.path, "seg_videos")
        mask_dir = Path(self.path, "seg_video_masks")

        self.train_dataset = RAMIEDataset(
            video_dir,
            mask_dir,
            self._identifiers("train"),
            target_parser=self.target_parser,
            transforms=self.transforms,
        )
        self.val_dataset = RAMIEDataset(
            video_dir,
            mask_dir,
            self._identifiers("val"),
            target_parser=self.target_parser,
        )
        # Test split is used by the inference scripts; only built when present so
        # training works for splits that have no test bundle.
        self.test_dataset = (
            RAMIEDataset(
                video_dir,
                mask_dir,
                self._identifiers("test"),
                target_parser=self.target_parser,
            )
            if self._split_file("test").exists()
            else None
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
