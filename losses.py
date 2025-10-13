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


from torch import einsum

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


#### from here on newly implemented losses ####

class DiceLoss():
    def __init__(self, idk=None, smooth=1e-6):
        self.idk = idk  # List of classes to supervise
        self.smooth = smooth

    def __call__(self, pred_softmax, target):
        """
        pred_softmax: (B, C, H, W) probabilities
        target: one-hot encoded (B, C, H, W)
        """
        if self.idk is not None:
            pred = pred_softmax[:, self.idk, ...]
            target = target[:, self.idk, ...]
        else:
            pred = pred_softmax
            target = target

        # Flatten per batch
        pred_flat = pred.contiguous().view(pred.shape[0], -1)
        target_flat = target.contiguous().view(target.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        return loss.mean()


class ComboLoss():
    def __init__(self, alpha=0.5, idk=None):
        self.alpha = alpha
        self.dice = DiceLoss(idk=idk)
        self.ce = CrossEntropy(idk=idk)

    def __call__(self, pred_softmax, target):

        dice_loss = self.dice(pred_softmax, target)
        ce_loss = self.ce(pred_softmax, target)
        return self.alpha * ce_loss + (1 - self.alpha) * dice_loss



class FocalLoss():
    """
    Focal Loss (Lin et al. 2017)
    Focuses on hard examples by down-weighting easy ones.
    Works for multi-class segmentation (with softmax inputs).
    """
    def __init__(self, alpha=1.0, gamma=2.0, idk=None):
        self.alpha = alpha
        self.gamma = gamma
        self.idk = idk

    def __call__(self, pred_softmax, target):
        if self.idk is not None:
            pred_softmax = pred_softmax[:, self.idk, ...]
            target = target[:, self.idk, ...]

        eps = 1e-10
        pred_softmax = torch.clamp(pred_softmax, eps, 1.0 - eps)

        loss = -self.alpha * target * (1 - pred_softmax) ** self.gamma * pred_softmax.log()
        return loss.mean()



class ComboFocalCrossEntropy():
    """
    Combination of Focal Loss and Cross Entropy Loss.
    alpha controls the weighting between the two losses.
    """
    def __init__(self, alpha=0.5, idk=None, focal_alpha=1.0, gamma=2.0):
        self.alpha = alpha
        self.focal = FocalLoss(alpha=focal_alpha, gamma=gamma, idk=idk)
        self.ce = CrossEntropy(idk=idk)

    def __call__(self, pred_softmax, target):
        focal_loss = self.focal(pred_softmax, target)
        ce_loss = self.ce(pred_softmax, target)
        return self.alpha * ce_loss + (1 - self.alpha) * focal_loss


class ComboFocalDice():
    """
    Combination of Focal Loss and Dice Loss.
    alpha controls the weighting between the two losses.
    """
    def __init__(self, alpha=0.5, idk=None, focal_alpha=1.0, gamma=2.0):
        self.alpha = alpha
        self.focal = FocalLoss(alpha=focal_alpha, gamma=gamma, idk=idk)
        self.dice = DiceLoss(idk=idk)

    def __call__(self, pred_softmax, target):
        focal_loss = self.focal(pred_softmax, target)
        dice_loss = self.dice(pred_softmax, target)
        return self.alpha * dice_loss + (1 - self.alpha) * focal_loss

