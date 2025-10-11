#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from pathlib import Path
import random
from typing import Callable, Union

import cv2
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)
    
    def augment_transform(self, save_debug_dir=None):
    ### online stochastic data augmentation ###
    # transformations are relatively subtle to mimic real
    # ct scan machine noise & variation and respect anatomy
        aug = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.8),
            A.ElasticTransform(alpha=20, sigma=5, alpha_affine=5, border_mode=0, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(var_limit=(5.0, 10.0), p=0.2),
        ])
        
        if save_debug_dir is not None:
            os.makedirs(os.path.join(save_debug_dir, "img"), exist_ok=True)
            os.makedirs(os.path.join(save_debug_dir, "mask"), exist_ok=True)

        def _apply(image, mask):
            if random.random() < 0.2:
                # 20% chance to skip all augmentation entirely
                return image, mask
            out = aug(image=image, mask=mask)
            
            if save_debug_dir is not None and random.random() < 0.05:
                # save 5%
                uid = f"{random.randint(0, 999999):06d}"
                cv2.imwrite(os.path.join(save_debug_dir, "img", f"{uid}.png"), out["image"])
                cv2.imwrite(os.path.join(save_debug_dir, "mask", f"{uid}.png"), out["mask"])

            
            return out["image"], out["mask"]
        
        return _apply

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]
        img = Image.open(img_path)
        if not self.test_mode:
            gt = Image.open(gt_path)
            
        if not self.test_mode and self.augment:
            # added data augmentation
            transformed = self.augment_transform(image=img, mask=gt, save_debug_dir="debug/")  # online augmentation
            img = Image.fromarray(transformed["image"])
            gt = Image.fromarray(transformed["mask"])

        img: Tensor = self.img_transform(img)  # base transform

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(gt) # base transform

            # sanity check
            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict
