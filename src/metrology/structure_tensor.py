import cv2
import numpy as np


class StructureTensorMetrology:
    """
    Continuous 2D Gaussian Structure Tensor for muscle fiber orientation.
    Suppresses acoustic speckle and extracts acute pennation angle and coherency.
    """
    def __init__(self, sigma=3.0, min_angle=5.0, max_angle=45.0, min_coherency=0.05):
        self.sigma = sigma
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_coherency = min_coherency

    def estimate_orientation(self, gray_img, roi_mask=None):
        img = gray_img.astype(np.float32)

        # 1. Spatial gradients via Sobel
        Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

        # 2. Gradient outer products
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # 3. Local 2D Gaussian tensor integration (crucial for rank-2 tensor & coherency)
        ksize = int(6 * self.sigma + 1)
        if ksize % 2 == 0:
            ksize += 1
        Jxx = cv2.GaussianBlur(Ixx, (ksize, ksize), self.sigma)
        Jyy = cv2.GaussianBlur(Iyy, (ksize, ksize), self.sigma)
        Jxy = cv2.GaussianBlur(Ixy, (ksize, ksize), self.sigma)

        # 4. Continuous orientation angle
        angle = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) + (np.pi / 2.0)
        angle_deg = (angle * 180.0 / np.pi) % 180.0

        # Acute angle mapping (prevents coordinate quadrant flip)
        acute_deg = np.minimum(angle_deg, 180.0 - angle_deg)

        # 5. Anisotropic coherency metric
        diff = Jxx - Jyy
        trace = Jxx + Jyy + 1e-6
        coherency = np.sqrt(diff * diff + 4.0 * Jxy * Jxy) / trace

        # Filter within valid muscle belly ROI
        if roi_mask is not None:
            valid = (roi_mask > 0) & (acute_deg >= self.min_angle) & (acute_deg <= self.max_angle) & (coherency >= self.min_coherency)
        else:
            valid = (acute_deg >= self.min_angle) & (acute_deg <= self.max_angle) & (coherency >= self.min_coherency)

        if np.sum(valid) > 50:
            weights = coherency[valid]
            angles = acute_deg[valid]
            # Coherency-weighted 68th percentile to isolate true fascicle orientation above horizontal noise floor
            idx = np.argsort(angles)
            sorted_angles = angles[idx]
            sorted_weights = weights[idx]
            cum_weights = np.cumsum(sorted_weights)
            cutoff = 0.68 * cum_weights[-1]
            q_idx = np.searchsorted(cum_weights, cutoff)
            dominant_angle = float(sorted_angles[min(q_idx, len(sorted_angles) - 1)])
            mean_coherency = float(np.mean(weights))
        else:
            dominant_angle = 15.0  # Normal physiological resting average
            mean_coherency = 0.0

        return dominant_angle, mean_coherency
