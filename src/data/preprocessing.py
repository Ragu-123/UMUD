import os
import glob
import cv2
import numpy as np


def filter_valid_image_files(file_list):
    """Filters out hidden files and OS metadata caches such as Thumbs.db."""
    valid_exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    return [
        f for f in file_list
        if f.lower().endswith(valid_exts) and not os.path.basename(f).startswith('.') and 'thumbs.db' not in f.lower()
    ]


def clean_mask_polarity_and_size(mask_gray, target_shape=None):
    """
    Corrects inverted binary masks and resamples to match the target image dimensions.
    """
    if target_shape is not None:
        target_h, target_w = target_shape[:2]
        if mask_gray.shape[:2] != (target_h, target_w):
            mask_gray = cv2.resize(mask_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Detect polarity inversion by inspecting boundary pixels
    border = np.concatenate([
        mask_gray[0, :], mask_gray[-1, :],
        mask_gray[:, 0], mask_gray[:, -1]
    ])
    if np.mean(border) > 127.0:
        mask_gray = 255 - mask_gray

    return (mask_gray > 127).astype(np.float32)


def discover_training_pairs(data_dir):
    """
    Scans for aponeurosis training image and mask pairs, resolving directory hierarchies.
    """
    img_candidates = [
        os.path.join(data_dir, 'apo_imgs_v1', 'apo_images_new_model_v1'),
        os.path.join(data_dir, 'apo_imgs_v1'),
    ] + glob.glob(os.path.join(data_dir, '**', 'apo_images*'), recursive=True)

    mask_candidates = [
        os.path.join(data_dir, 'apo_masks_v1', 'apo_masks_new_model_v1'),
        os.path.join(data_dir, 'apo_masks_v1'),
    ] + glob.glob(os.path.join(data_dir, '**', 'apo_masks*'), recursive=True)

    img_dir = next((d for d in img_candidates if os.path.isdir(d)), None)
    mask_dir = next((d for d in mask_candidates if os.path.isdir(d)), None)

    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(f"Could not locate aponeurosis datasets in {data_dir}")

    raw_files = filter_valid_image_files(os.listdir(img_dir))
    pairs = []
    for f in sorted(raw_files):
        img_p = os.path.join(img_dir, f)
        mask_p = os.path.join(mask_dir, f)
        if os.path.exists(mask_p):
            pairs.append((img_p, mask_p))

    return pairs
