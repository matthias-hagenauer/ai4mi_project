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

# new imports
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

PID_RX = re.compile(r"(Patient_\d+)_\d{4}\.png")  # patient ID regex


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ["train", "val", "test"]

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / "img"
    full_path = root / subset / "gt"

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != "test":
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(
        self,
        subset,
        root_dir,
        img_transform=None,
        gt_transform=None,
        augment=False,
        equalize=False,
        debug=False,
        radius=2,
        pad_mode="edge",
    ):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize

        self.test_mode: bool = subset == "test"

        # added for 2.5D
        self.radius: int = radius

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        # again added for 2.5D
        self.pid_to_indices: Dict[str, List[int]] = {}  # patient ID to list of indices of slices
        self.index_to_pid_pos: List[Tuple[str, int]] = []  # index to (patient ID, position in patient stack)

        for i, (img_path, _) in enumerate(self.files):  # iterate over all slices
            m = PID_RX.search(img_path.name)  # patient ID
            if not m:
                raise ValueError(f"Cannot parse patient ID from {img_path.name}")
            pid = m.group(1)  # extract patient ID
            self.pid_to_indices.setdefault(pid, []).append(i)
            self.index_to_pid_pos.append((pid, -1))  # fill later

        # ensure per-patient slices are sorted by filename
        for pid, idxs in self.pid_to_indices.items():  # for each patient
            idxs.sort(key=lambda k: self.files[k][0].name)  # sort indices by filename
            # update position mapping
            for pos, k in enumerate(idxs):
                self.index_to_pid_pos[k] = (pid, pos)

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def _neighbor_indices_(self, pid: str, pos: int) -> List[int]:  # core of 2.5D
        idxs = self.pid_to_indices[pid]  # all slice indexes for this patient
        n = len(idxs)
        ii = []  # neighbor indexes
        for off in range(-self.radius, self.radius + 1):  # loop around the current position
            j = pos + off  # neighbor position e.g. 10 + -2 = 8
            j = 0 if j < 0 else (n - 1 if j >= n else j)  # edge cases
            ii.append(idxs[j])  # global index
        return ii  # list of neighbors

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]
        pid, pos = self.index_to_pid_pos[index]
        neigh_global_idxs = self._neighbor_indices_(pid, pos)  # get neighbor indexes

        imgs = []
        for gi in neigh_global_idxs:  # for each neighbor index
            if gi == -1:  # zero padding
                arr = np.zeros_like(np.array(Image.open(img_path)))  # black image
            else:
                p, _ = self.files[gi]  # get path to neighbor slice
                arr = np.array(Image.open(p))  # load neighbor slice
            imgs.append(arr)

        # (C,H,W)
        vol = np.stack(imgs, axis=0)
        vol = vol.astype(np.float32) / 255.0  # normalize to [0,1]
        m = vol.mean()
        s = vol.std()
        vol = (vol - m) / (s + 1e-8)  # normalization to zero mean and unit variance
        img: Tensor = torch.from_numpy(vol)  # to tensor

        data_dict = {"images": img, "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))  # for center slice only

            data_dict["gts"] = gt

        return data_dict
