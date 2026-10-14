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
    Extracts spatial scale (pixels per millimeter) and active acoustic bounds
    from ultrasound displays by parsing vertical graticule ticks per scanner model.
    Reverse-engineered from hardware displays by Grandmaster AmbrosM.
    """
    def __init__(self, fallback_px_per_mm=10.0):
        self.fallback = fallback_px_per_mm

    def decode(self, img_np, filename=""):
        a = img_np
        height, width = a.shape[:2]
        filetype = filename.split('.')[-1].lower() if '.' in filename else 'tif'

        if filetype == 'png':
            first_tick = np.argmax(a[:, 6].mean(axis=-1) > 50)
            second_minor_tick = 150 + np.argmax(a[150:, 6].mean(axis=-1) > 50)
            second_major_tick = 150 + np.argmax(a[150:, 9].mean(axis=-1) > 50)
            last_tick = len(a) - 1 - np.argmax(a[:, 6].mean(axis=-1)[::-1] > 50)
            if (second_major_tick - first_tick) < 3 * (second_minor_tick - first_tick):
                px_per_cm = second_major_tick - first_tick
            else:
                px_per_cm = second_minor_tick - first_tick
            hw = a.shape[1] // 2
            w2 = np.argmin(a[:, hw:].sum(axis=(0, 2)))
            l, t, r, b = hw - w2, first_tick, hw + w2, last_tick
        elif a.shape[:2] == (800, 1200):
            if len(a.shape) == 3 and (a[87, 1147:1157] == 175).all():
                first_tick = np.argmax(a[:, 1150].mean(axis=-1) > 50)
                second_major_tick = first_tick + 20 + np.argmax(a[first_tick + 20:, 1150].mean(axis=-1) > 50)
                last_tick = len(a) - 1 - np.argmax(a[:, 1150].mean(axis=-1)[::-1] > 50)
                n_ticks = round((last_tick - first_tick) / (second_major_tick - first_tick))
                if n_ticks <= 14:
                    px_per_cm = (last_tick - first_tick) / n_ticks * 2
                    offsets = {7: (142, 1058), 8: (163, 1037), 9: (211, 989), 10: (249, 951),
                               11: (282, 918), 12: (308, 892), 13: (331, 869), 14: (349, 851)}
                    l, r = offsets.get(n_ticks, (142, 1058))
                    t, b = 91, last_tick
                elif n_ticks == 15:
                    px_per_cm = (last_tick - first_tick) / 3
                    l, t, r, b = 142, 91, 1058, last_tick
                else:
                    px_per_cm = 148.0
                    l, t, r, b = 142, 91, 1058, 700
            elif len(a.shape) == 3 and (a[42, 67:74, 0] > 115).all():
                px_per_cm = (783 - 42) / 5
                l, t, r, b = 171, 42, 1029, 798
            else:
                px_per_cm = 148.0
                l, t, r, b = 0, 0, width, height
        elif a.shape[:2] == (644, 1088):
            px_per_cm = 630.5 / 5
            l, t, r, b = 140, 0, 947, 643
        elif a.shape[0] in [512, 513]:
            px_per_cm = (442 - 53) / 5
            l, t, r, b = 0, 0, width, height - 10
        elif a.shape[0] == 853:
            if len(a.shape) == 2 and a[-5, 100] == 170 and a[-5, 934] == 170:
                px_per_cm = (934 - 100) / 5
            elif len(a.shape) == 2 and a[-5, 44] == 170 and a[-5, 879] == 170:
                px_per_cm = (879 - 44) / 5
            else:
                px_per_cm = 167.0
            l, t, r, b = 0, 0, width, height
        else:
            px_per_cm = 100.0
            l, t, r, b = 0, 0, width, height

        px_per_mm = px_per_cm / 10.0
        return px_per_mm, (l, t, r, b)
