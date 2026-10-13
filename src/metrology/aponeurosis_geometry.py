import numpy as np


class AponeurosisGeometryExtractor:
    """
    Extracts sub-pixel inner boundaries of superficial and deep aponeuroses.
    Computes anatomical Muscle Thickness (MT) and convergence angle (phi).
    """
    def __init__(self, probe_midline_window=40):
        self.window = probe_midline_window

    def extract_geometry(self, pred_mask_binary, px_per_mm):
        h, w = pred_mask_binary.shape
        x0 = w // 2

        # Split mask into superficial (top half) and deep (bottom half)
        half_h = h // 2
        sup_mask = np.zeros_like(pred_mask_binary)
        deep_mask = np.zeros_like(pred_mask_binary)
        sup_mask[:half_h, :] = pred_mask_binary[:half_h, :]
        deep_mask[half_h:, :] = pred_mask_binary[half_h:, :]

        x_start = max(0, x0 - self.window)
        x_end = min(w, x0 + self.window)

        # Ventral inner edge of superficial aponeurosis (lower border)
        sup_inner_y = []
        for x in range(x_start, x_end):
            col = np.where(sup_mask[:, x] > 0.5)[0]
            if len(col) > 0:
                sup_inner_y.append(np.max(col))
        y_sup = float(np.median(sup_inner_y)) if len(sup_inner_y) > 0 else 0.20 * h

        # Dorsal inner edge of deep aponeurosis (upper border)
        deep_inner_y = []
        for x in range(x_start, x_end):
            col = np.where(deep_mask[:, x] > 0.5)[0]
            if len(col) > 0:
                deep_inner_y.append(np.min(col))
        y_deep = float(np.median(deep_inner_y)) if len(deep_inner_y) > 0 else 0.80 * h

        # Fit linear slopes to estimate convergence angle phi
        sup_pts = np.where(sup_mask > 0.5)
        if len(sup_pts[0]) > 50:
            p_sup = np.polyfit(sup_pts[1], sup_pts[0], 1)
            slope_sup = float(np.arctan(p_sup[0]) * 180.0 / np.pi)
        else:
            slope_sup = 0.0

        deep_pts = np.where(deep_mask > 0.5)
        if len(deep_pts[0]) > 50:
            p_deep = np.polyfit(deep_pts[1], deep_pts[0], 1)
            slope_deep = float(np.arctan(p_deep[0]) * 180.0 / np.pi)
        else:
            slope_deep = 0.0

        phi = abs(slope_sup - slope_deep)

        # Muscle Thickness (mm)
        raw_mt_px = max(10.0, y_deep - y_sup)
        mt_mm = float(raw_mt_px / px_per_mm)

        # Build muscle belly mask for fascicle analysis
        muscle_belly_mask = np.zeros_like(pred_mask_binary)
        y_top = int(np.clip(y_sup + 5, 0, h - 1))
        y_bot = int(np.clip(y_deep - 5, 0, h - 1))
        if y_bot > y_top:
            muscle_belly_mask[y_top:y_bot, :] = 1

        return {
            "mt_mm": mt_mm,
            "y_sup": y_sup,
            "y_deep": y_deep,
            "slope_sup": slope_sup,
            "slope_deep": slope_deep,
            "phi": phi,
            "muscle_belly_mask": muscle_belly_mask,
        }
