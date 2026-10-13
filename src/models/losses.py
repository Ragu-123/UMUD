import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def compute_dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    """Computes binary Dice coefficient for evaluation."""
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > threshold).float()

    intersection = (preds_bin * targets_bin).sum().item()
    union = preds_bin.sum().item() + targets_bin.sum().item()
    return (2.0 * intersection + smooth) / (union + smooth)


def compute_iou_score(preds, targets, threshold=0.5, smooth=1e-6):
    """Computes binary IoU (Jaccard index) for evaluation."""
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > threshold).float()

    intersection = (preds_bin * targets_bin).sum().item()
    union = (preds_bin + targets_bin).clamp(0, 1).sum().item()
    return (intersection + smooth) / (union + smooth)
