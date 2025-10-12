#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

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

import argparse
import random
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from preprocess import Denoise2D, ROIBackgroundStrip
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy)
from functools import partial
import json, csv

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
optimizer_name = None

def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def write_run_summary(args, e_best, log_dice_val_epoch):
    K = log_dice_val_epoch.shape[1]
    fg = slice(1, K) if K > 1 else slice(0, K)
    per_img_macro = log_dice_val_epoch[:, fg].mean(axis=1)
    macro_mean = float(per_img_macro.mean())
    macro_std = float(per_img_macro.std())
    per_class_means = [float(log_dice_val_epoch[:, k].mean()) for k in range(K)]

    js = {
        "dataset": args.dataset,
        "mode": args.mode,
        "seed": getattr(args, "seed", None),
        "use_roi": bool(getattr(args, "use_roi", False)),
        "roi_method": getattr(args, "roi_method", None),
        "roi_thresh": getattr(args, "roi_thresh", None),
        "denoise": bool(getattr(args, "denoise", False)),
        "denoise_method": getattr(args, "denoise_method", None),
        "denoise_ksize": getattr(args, "denoise_ksize", None),
        "denoise_sigma": getattr(args, "denoise_sigma", None),
        "optimizer": optimizer_name,
        "best_epoch": int(e_best),
        "macro_dice_mean": macro_mean,
        "macro_dice_std": macro_std,
        "per_class_dice_means": per_class_means,
    }
    metrics_json = _ensure_dir(args.dest / "metrics.json")
    with open(metrics_json, "w") as f:
        json.dump(js, f, indent=2)

    csv_path = _ensure_dir(args.dest.parent / "results.csv")
    header = ["dataset","mode","seed","use_roi","roi_method","roi_thresh","denoise","denoise_method","denoise_ksize","denoise_sigma","optimizer","best_epoch","macro_dice_mean","macro_dice_std"] + [f"dice_class_{k}" for k in range(K)]
    row = [js[h] if h in js else None for h in header[:14]] + per_class_means
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists: writer.writerow(header)
        writer.writerow(row)

    # aggregate
    df = pd.read_csv(csv_path)

    metric_cols = ["macro_dice_mean"] + [f"dice_class_{k}" for k in range(K)]
    for c in metric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    group_cols = [
        "dataset", "mode", "use_roi", "roi_method", "roi_thresh",
        "denoise", "denoise_method", "denoise_ksize", "denoise_sigma", "optimizer"
    ]

    grouped = df.groupby(group_cols, dropna=False)[metric_cols].agg(['mean', 'std']).reset_index()

    def flat(col):
        if not isinstance(col, tuple):
            return col
        base, stat = col
        return base if stat == '' else f"{base}_{stat}"

    grouped.columns = [flat(c) for c in grouped.columns]

    agg_path = args.dest.parent / "results_aggregated.csv"
    grouped.to_csv(agg_path, index=False)
    print(f">> Updated aggregated results at {agg_path}")


def gt_transform(K, img):
    img = np.array(img)[...]
    # The idea is that the classes are mapped to {0, 255} for binary cases
    # {0, 85, 170, 255} for 4 classes
    # {0, 51, 102, 153, 204, 255} for 6 classes
    # Very sketchy but that works here and that simplifies visualization
    img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
    img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
    img = class2one_hot(img, K=K)
    return img[0]


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif args.gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f">> Picked {device} to run experiments")

    if getattr(args, "seed", None) is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
        print(f">> Set all seeds to {args.seed}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    g = None
    if getattr(args, "seed", None) is not None:
        g = torch.Generator()
        g.manual_seed(args.seed)

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset

    # Picklable image transform (no lambdas, all top-level ops)
    base_img_transforms = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),  # -> [C,H,W] float32 in [0,1]
    ]
    if getattr(args, "denoise", False):
        base_img_transforms.append(
            Denoise2D(
                method=args.denoise_method,
                ksize=args.denoise_ksize,
                sigma=args.denoise_sigma
            )
        )

    if getattr(args, "use_roi", False):
        base_img_transforms.append(
            ROIBackgroundStrip(method=args.roi_method, thresh=args.roi_thresh)
        )
    img_transform = transforms.Compose(base_img_transforms)

    train_set = SliceDataset('train',
                             root_dir,
                             img_transform=img_transform,
                             gt_transform= partial(gt_transform, K),
                             debug=args.debug)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True,generator=g)

    val_set = SliceDataset('val',
                           root_dir,
                           img_transform=img_transform,
                           gt_transform=partial(gt_transform, K),
                           debug=args.debug)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False,generator=g)


    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")
            log_dice_epoch = log_dice_val[e].cpu().numpy()  # [N_images, K]
            write_run_summary(args, e, log_dice_epoch)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results_2 (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--use_roi', action='store_true',
                        help="Enable simple ROI preprocessing that suppresses background intensities (no size change).")
    parser.add_argument('--roi_method', default='otsu', choices=['otsu', 'fixed'],
                        help="ROI thresholding method: 'otsu' (default) or 'fixed'.")
    parser.add_argument('--roi_thresh', default=0.1, type=float,
                        help="Fixed threshold in [0,1] if --roi_method fixed (ignored for 'otsu').")

    parser.add_argument('--denoise', action='store_true',
                        help="Apply denoising to input images (after ToTensor).")

    parser.add_argument('--denoise_method', default='gaussian', choices=['gaussian', 'median'],
                        help="Denoising method: 'gaussian' (separable blur) or 'median'.")

    parser.add_argument('--denoise_ksize', default=5, type=int,
                        help="Kernel size for denoising (odd integer).")

    parser.add_argument('--denoise_sigma', default=1.0, type=float,
                        help="Sigma for gaussian denoising (ignored by median).")

    parser.add_argument('--seeds', type=str, default=None,
                        help="Comma-separated seeds; if set, overrides --seed and loops internally")

    args = parser.parse_args()

    pprint(args)

    if args.seeds:
        for s in [int(x) for x in args.seeds.split(',')]:
            sub = argparse.Namespace(**vars(args))
            sub.seed = s
            sub.dest = args.dest / f"seed{s}"
            runTraining(sub)
    else:
        runTraining(args)


if __name__ == '__main__':
    main()
