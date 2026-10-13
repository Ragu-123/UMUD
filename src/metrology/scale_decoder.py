import cv2
import numpy as np
from scipy.signal import find_peaks

KNOWN_HARDWARE_PRESETS = {
    (1200, 800): 14.80,   # Siemens / GE medical linear transducer
    (1088, 644): 9.80,    # Telemed Echo Blaster
    (1069, 853): 13.40,   # Esaote MyLab
    (959, 853): 13.40,    # Esaote cropped
    (465, 513): 9.00,     # Aloka / Sonosite portable
    (461, 513): 9.00,
    (463, 513): 9.00,
    (464, 513): 9.00,
    (465, 512): 9.00,
    (466, 513): 9.00,
}


def analyze_margin_ticks(strip_gray, min_distance=20):
    """
    Computes horizontal Sobel gradient and identifies periodic graticule peaks.
    """
    grad_y = cv2.Sobel(strip_gray, cv2.CV_64F, 0, 1, ksize=3)
    profile = np.mean(np.abs(grad_y), axis=1)

    # Detrend
    window_len = min(25, len(profile) // 2 * 2 + 1)
    if window_len > 3:
        profile = profile - np.convolve(profile, np.ones(window_len)/window_len, mode='same')
    profile = np.clip(profile, 0, None)

    if np.max(profile) < 1e-4:
        return None

    norm_profile = profile / np.max(profile)
    peaks, _ = find_peaks(norm_profile, height=0.25, distance=min_distance)

    if len(peaks) >= 3:
        diffs = np.diff(peaks)
        median_spacing = np.median(diffs)
        if np.std(diffs) < 0.20 * median_spacing:
            return float(median_spacing)
    return None


class ScaleDecoder:
    """
    Extracts spatial scale (pixels per millimeter) from ultrasound displays.
    Scans both right and left margins for graticules, falling back to hardware presets.
    """
    def __init__(self, fallback_px_per_mm=9.80):
        self.fallback = fallback_px_per_mm

    def decode(self, img_np):
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np

        h, w = gray.shape
        shape_key = (w, h)

        # 1. Look up known hardware presets
        preset = KNOWN_HARDWARE_PRESETS.get(shape_key, None)

        # 2. Adaptive margin search
        detected_spacing = None
        if w > 100:
            # Check right margin
            strip_r = gray[:, max(0, w-60):w-10]
            detected_spacing = analyze_margin_ticks(strip_r)

            # Check left margin if right fails
            if detected_spacing is None:
                strip_l = gray[:, 15:min(w, 65)]
                detected_spacing = analyze_margin_ticks(strip_l)

        if detected_spacing is not None:
            px_per_mm = detected_spacing / 10.0  # Major tick = 10 mm (1 cm)
            if 5.0 <= px_per_mm <= 25.0:
                return float(px_per_mm)

        return float(preset if preset is not None else self.fallback)
