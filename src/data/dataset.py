import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from .preprocessing import clean_mask_polarity_and_size


class AponeurosisDataset(Dataset):
    """
    PyTorch Dataset for ultrasound aponeurosis segmentation.
    Handles dimension matching, polarity correction, and data augmentation.
    """
    def __init__(self, pairs, target_size=(512, 512), is_train=True):
        self.pairs = pairs
        self.target_size = target_size
        self.is_train = is_train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Cannot load mask: {mask_path}")

        # Correct polarity and dimension mismatch
        mask = clean_mask_polarity_and_size(mask, target_shape=img.shape)

        # Data augmentations
        if self.is_train:
            # Horizontal flip
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

            # Contrast and brightness jitter
            if np.random.rand() > 0.5:
                alpha = 1.0 + np.random.uniform(-0.15, 0.15)
                beta = np.random.uniform(-15, 15)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # Resize to network dimensions
        th, tw = self.target_size
        if img.shape[:2] != (th, tw):
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Normalize and convert to tensors
        img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()

        return img_t, mask_t


def make_kfold_splits(pairs, num_folds=5, seed=42):
    """Generates deterministic K-fold train/validation splits."""
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in kf.split(pairs):
        train_pairs = [pairs[i] for i in train_idx]
        val_pairs = [pairs[i] for i in val_idx]
        splits.append((train_pairs, val_pairs))
    return splits
