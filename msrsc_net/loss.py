"""The full fixed MSRSC-Net training objective (no loss ablation switches)."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class MSRSCLoss(nn.Module):
    """BCE + 0.25 Dice + 0.45 IoU + 0.15 Boundary + 0.5 scheduled UAL.

    BCE foreground pixels receive weight 1.05. `progress` is the completed
    training fraction in [0, 1], used for the cosine UAL ramp.
    """

    @staticmethod
    def _boundary(x):
        maximum = F.max_pool2d(x, 5, stride=1, padding=2)
        minimum = -F.max_pool2d(-x, 5, stride=1, padding=2)
        return (maximum - minimum).clamp(0, 1)

    def forward(self, logits, mask, progress):
        if not 0 <= progress <= 1:
            raise ValueError("progress must be a training fraction in [0, 1].")
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("logits must have shape [N, 1, H, W].")
        if mask.ndim != 4 or mask.shape[:2] != logits.shape[:2]:
            raise ValueError(
                "mask must have shape [N, 1, H, W], with the same batch size."
            )
        mask = mask.to(device=logits.device, dtype=logits.dtype)
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(
                logits, size=mask.shape[-2:], mode="bilinear", align_corners=False
            )
        prob = logits.sigmoid()
        bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
        bce = (bce * torch.where(mask > 0.5, 1.05, 1.0)).mean()
        intersection = (prob * mask).sum((1, 2, 3))
        total = prob.sum((1, 2, 3)) + mask.sum((1, 2, 3))
        dice = (1 - (2 * intersection + 1) / (total + 1)).mean()
        iou = (1 - (intersection + 1) / (total - intersection + 1)).mean()
        boundary = F.l1_loss(self._boundary(prob), self._boundary(mask))
        ramp = (1 - math.cos(progress * math.pi)) / 2
        ual = ramp * (1 - (2 * prob - 1).abs().pow(2)).mean()
        return bce + 0.25 * dice + 0.45 * iou + 0.15 * boundary + 0.5 * ual
