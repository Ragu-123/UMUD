import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class BiomechanicalMetrology:
    """
    Advanced Biomechanical Metrology Engine for Muscle Architecture.
    Combines:
    1. 1D sub-pixel autocorrelation & Wiener-Khinchin FFT scale calibration.
    2. Sub-pixel 0.5 level-set crossing & robust Huber line fitting for inner aponeuroses.
    3. Two-scale 2D structure tensor with von Mises circular mode estimation for Pennation Angle.
    4. Non-parallel aponeurosis convergence law: FL = MT / sin(PA - phi/2).
    """
    def __init__(
        self,
        device=None,
        default_px_per_mm=10.8,
        prob_threshold=0.5,
        perpendicular_mt=False,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.default_px_per_mm = float(default_px_per_mm)
        self.prob_threshold = float(prob_threshold)
        self.perpendicular_mt = bool(perpendicular_mt)

    def _to_grayscale_uint8(self, image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        img = np.asarray(image)
        if img.ndim == 3:
            if img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            if img.shape[2] == 3:
                img = cv2.cvtColor(img.astype(np.uint8) if img.max() > 1.0 else (img * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
            else:
                img = img[:, :, 0]
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def decode_graticule_scale(self, gray_img):
        H, W = gray_img.shape[:2]
        y0, y1 = int(0.08 * H), int(0.92 * H)
        margin_w = min(75, max(25, int(0.12 * W)))
        strips = [
            gray_img[y0:y1, 4:margin_w],
            gray_img[y0:y1, max(0, W - margin_w):max(1, W - 4)],
        ]

        best_score = -1.0
        best_rho = self.default_px_per_mm

        for strip in strips:
            if strip.size == 0 or strip.shape[0] < 64 or strip.shape[1] < 4:
                continue
            s_float = strip.astype(np.float64)
            gy = np.abs(s_float[2:, 1:-1] - s_float[:-2, 1:-1])
            gx = np.abs(s_float[1:-1, 2:] - s_float[1:-1, :-2])
            tick_resp = np.maximum(0.0, gy - 0.5 * gx)

            col_max = np.percentile(tick_resp, 96, axis=0)
            col_med = np.median(tick_resp, axis=0) + 1e-3
            col_snr = col_max / col_med
            k_cols = min(5, tick_resp.shape[1])
            top_idx = np.argsort(col_snr)[-k_cols:]
            if np.mean(col_max[top_idx]) < 8.0:
                continue

            prof = np.mean(tick_resp[:, top_idx], axis=1)
            kernel_len = max(15, int(0.08 * len(prof)) | 1)
            smooth_base = cv2.GaussianBlur(prof.reshape(-1, 1), (1, kernel_len), 0).ravel()
            sig = np.maximum(0.0, prof - smooth_base)
            norm_val = np.linalg.norm(sig)
            if norm_val < 1e-6:
                continue
            sig = sig / norm_val

            n = len(sig)
            n_fft = 1 << int(math.ceil(math.log2(2 * n)))
            spec = np.fft.rfft(sig, n=n_fft)
            acf = np.fft.irfft(spec * np.conj(spec), n=n_fft)[:n]
            unbiased = np.maximum(1.0, np.arange(n, 0, -1, dtype=np.float64))
            acf = (acf / unbiased) / (acf[0] + 1e-12)

            min_lag = 12
            max_lag = min(n // 2, 190)
            if max_lag <= min_lag + 4:
                continue

            for lag in range(min_lag, max_lag - 1):
                if acf[lag] > acf[lag - 1] and acf[lag] >= acf[lag + 1] and acf[lag] > 0.12:
                    denom = acf[lag - 1] - 2.0 * acf[lag] + acf[lag + 1]
                    offset = 0.5 * (acf[lag - 1] - acf[lag + 1]) / denom if abs(denom) > 1e-9 else 0.0
                    sub_lag = float(lag) + float(np.clip(offset, -0.5, 0.5))

                    harm_sum = acf[lag]
                    harm_count = 1
                    for mult in (2, 3):
                        h_idx = int(round(mult * sub_lag))
                        if h_idx < n:
                            harm_sum += float(np.max(acf[max(0, h_idx - 2):min(n, h_idx + 3)]))
                            harm_count += 1
                    score = harm_sum / harm_count

                    for div in (10.0, 5.0, 2.0, 1.0, 0.5):
                        cand_rho = sub_lag / div
                        if 6.5 <= cand_rho <= 16.5:
                            if score > best_score:
                                best_score = score
                                best_rho = cand_rho
                            break

        return float(best_rho)

    def _prepare_aponeurosis_masks(self, apo_mask, target_h, target_w):
        if isinstance(apo_mask, (tuple, list)) and len(apo_mask) == 2:
            m_sup = self._normalize_single_mask(apo_mask[0], target_h, target_w)
            m_deep = self._normalize_single_mask(apo_mask[1], target_h, target_w)
            return m_sup, m_deep

        if isinstance(apo_mask, torch.Tensor):
            apo_mask = apo_mask.detach().cpu().numpy()
        arr = np.asarray(apo_mask)

        if arr.ndim == 3:
            if arr.shape[0] == 2 and arr.shape[2] != 2:
                m_sup = self._normalize_single_mask(arr[0], target_h, target_w)
                m_deep = self._normalize_single_mask(arr[1], target_h, target_w)
                return m_sup, m_deep
            if arr.shape[2] == 2:
                m_sup = self._normalize_single_mask(arr[:, :, 0], target_h, target_w)
                m_deep = self._normalize_single_mask(arr[:, :, 1], target_h, target_w)
                return m_sup, m_deep
            arr = arr[:, :, 0] if arr.shape[2] in (1, 3) else arr[0]

        prob = self._normalize_single_mask(arr, target_h, target_w)
        binary = (prob >= self.prob_threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        if num_labels < 3:
            mid_y = target_h // 2
            m_sup = prob.copy()
            m_sup[mid_y:, :] = 0.0
            m_deep = prob.copy()
            m_deep[:mid_y, :] = 0.0
            return m_sup, m_deep

        min_area = 0.001 * target_h * target_w
        valid = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        if len(valid) < 2:
            valid = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:2]
        else:
            valid = sorted(valid, key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:2]

        valid = sorted(valid, key=lambda i: centroids[i][1])
        sup_id, deep_id = valid[0], valid[1]

        m_sup = np.where(labels == sup_id, prob, 0.0).astype(np.float32)
        m_deep = np.where(labels == deep_id, prob, 0.0).astype(np.float32)
        return m_sup, m_deep

    def _normalize_single_mask(self, mask, target_h, target_w):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        m = np.asarray(mask, dtype=np.float32)
        if m.max() > 1.0:
            m = m / 255.0
        m = np.clip(m, 0.0, 1.0)

        border_vals = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]])
        if np.mean(border_vals > 0.5) > 0.5:
            m = 1.0 - m

        if m.shape[0] != target_h or m.shape[1] != target_w:
            m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return m

    def _fit_huber_line(self, xs, ys, iterations=5):
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        w = np.ones_like(xs)
        m, b = 0.0, float(np.median(ys))
        for _ in range(iterations):
            W_sum = np.sum(w) + 1e-12
            x_bar = np.sum(w * xs) / W_sum
            y_bar = np.sum(w * ys) / W_sum
            dx = xs - x_bar
            dy = ys - y_bar
            denom = np.sum(w * dx * dx) + 1e-12
            m = float(np.sum(w * dx * dy) / denom)
            b = float(y_bar - m * x_bar)
            res = np.abs(ys - (m * xs + b))
            mad = float(np.median(res)) + 1.0
            delta = 1.345 * mad
            w = np.where(res <= delta, 1.0, delta / (res + 1e-9))
        return m, b

    def extract_aponeurosis_geometry(self, apo_mask, orig_h, orig_w, px_per_mm):
        m_sup, m_deep = self._prepare_aponeurosis_masks(apo_mask, orig_h, orig_w)
        x_start, x_end = int(0.25 * orig_w), int(0.75 * orig_w)
        thr = self.prob_threshold

        xs_sup, ys_sup = [], []
        xs_deep, ys_deep = [], []

        for x in range(x_start, x_end):
            col_s = m_sup[:, x]
            idx_s = np.where(col_s >= thr)[0]
            if len(idx_s) > 0:
                yk = int(np.max(idx_s))
                if yk + 1 < orig_h and col_s[yk] > col_s[yk + 1]:
                    sub_y = yk + (col_s[yk] - thr) / (col_s[yk] - col_s[yk + 1] + 1e-8)
                else:
                    sub_y = float(yk)
                xs_sup.append(float(x))
                ys_sup.append(float(sub_y))

            col_d = m_deep[:, x]
            idx_d = np.where(col_d >= thr)[0]
            if len(idx_d) > 0:
                yk = int(np.min(idx_d))
                if yk - 1 >= 0 and col_d[yk] > col_d[yk - 1]:
                    sub_y = yk - (col_d[yk] - thr) / (col_d[yk] - col_d[yk - 1] + 1e-8)
                else:
                    sub_y = float(yk)
                xs_deep.append(float(x))
                ys_deep.append(float(sub_y))

        if len(xs_sup) < 15 or len(xs_deep) < 15:
            y_sup_c = 0.25 * orig_h
            y_deep_c = 0.70 * orig_h
            mt_mm = (y_deep_c - y_sup_c) / px_per_mm
            roi = np.zeros((orig_h, orig_w), dtype=np.uint8)
            roi[int(y_sup_c) + 10:int(y_deep_c) - 10, x_start:x_end] = 1
            return {
                "y_sup_inner": float(y_sup_c),
                "y_deep_inner": float(y_deep_c),
                "theta_sup": 0.0,
                "theta_deep": 0.0,
                "phi": 0.0,
                "mt_mm": float(mt_mm),
                "roi_mask": roi,
            }

        slope_sup, inter_sup = self._fit_huber_line(xs_sup, ys_sup)
        slope_deep, inter_deep = self._fit_huber_line(xs_deep, ys_deep)

        x0 = 0.5 * orig_w
        y_sup_inner = slope_sup * x0 + inter_sup
        y_deep_inner = slope_deep * x0 + inter_deep

        theta_sup = math.degrees(math.atan(slope_sup))
        theta_deep = math.degrees(math.atan(slope_deep))
        phi = abs(theta_sup - theta_deep)

        chord_px = max(1.0, y_deep_inner - y_sup_inner)
        if self.perpendicular_mt:
            mean_rad = 0.5 * (math.atan(slope_sup) + math.atan(slope_deep))
            mt_mm = (chord_px * math.cos(mean_rad)) / px_per_mm
        else:
            mt_mm = chord_px / px_per_mm

        roi_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        margin_px = max(6, int(round(1.2 * px_per_mm)))
        rx0, rx1 = int(0.20 * orig_w), int(0.80 * orig_w)
        for x in range(rx0, rx1):
            yt = int(math.ceil(slope_sup * x + inter_sup)) + margin_px
            yb = int(math.floor(slope_deep * x + inter_deep)) - margin_px
            if yb > yt + 8:
                roi_mask[max(0, yt):min(orig_h, yb), x] = 1

        return {
            "y_sup_inner": float(y_sup_inner),
            "y_deep_inner": float(y_deep_inner),
            "theta_sup": float(theta_sup),
            "theta_deep": float(theta_deep),
            "phi": float(phi),
            "mt_mm": float(np.clip(mt_mm, 8.0, 55.0)),
            "roi_mask": roi_mask,
        }

    def _gaussian_kernel_2d(self, sigma):
        radius = max(2, int(math.ceil(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=self.device)
        g1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
        g1d = g1d / g1d.sum()
        g2d = torch.outer(g1d, g1d)
        return g2d.view(1, 1, 2 * radius + 1, 2 * radius + 1), radius

    def compute_structure_tensor_pa(self, gray_img, roi_mask, theta_deep, sigma_d=1.2, sigma_i=3.0):
        img_np = gray_img.astype(np.float32) / 255.0
        log_np = np.log1p(img_np * 8.0)

        t_img = torch.from_numpy(log_np).to(self.device).view(1, 1, *log_np.shape)
        t_roi = torch.from_numpy(roi_mask.astype(np.bool_)).to(self.device)

        kd, rd = self._gaussian_kernel_2d(sigma_d)
        t_smooth = F.conv2d(F.pad(t_img, (rd, rd, rd, rd), mode="reflect"), kd)

        kx = torch.tensor([[[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]], dtype=torch.float32, device=self.device).unsqueeze(0) / 32.0
        ky = kx.transpose(2, 3)

        Ix = F.conv2d(F.pad(t_smooth, (1, 1, 1, 1), mode="reflect"), kx)
        Iy = F.conv2d(F.pad(t_smooth, (1, 1, 1, 1), mode="reflect"), ky)

        Jxx = Ix * Ix
        Jyy = Iy * Iy
        Jxy = Ix * Iy

        ki, ri = self._gaussian_kernel_2d(sigma_i)
        J_stack = torch.cat([Jxx, Jyy, Jxy], dim=0)
        J_bar = F.conv2d(F.pad(J_stack, (ri, ri, ri, ri), mode="reflect"), ki)

        Jxx_b = J_bar[0, 0]
        Jyy_b = J_bar[1, 0]
        Jxy_b = J_bar[2, 0]

        trace = Jxx_b + Jyy_b
        diff = Jxx_b - Jyy_b
        disc = torch.sqrt(diff * diff + 4.0 * Jxy_b * Jxy_b + 1e-12)

        lambda1 = 0.5 * (trace + disc)
        lambda2 = 0.5 * (trace - disc)
        coherency = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8)

        theta_rad = (0.5 * torch.atan2(2.0 * Jxy_b, diff) + 0.5 * math.pi) % math.pi
        theta_deg = torch.rad2deg(theta_rad)
        theta_acute = torch.minimum(theta_deg, 180.0 - theta_deg)
        slope_sign = torch.where(theta_deg <= 90.0, torch.ones_like(theta_deg), -torch.ones_like(theta_deg))

        roi_coh = coherency[t_roi]
        if roi_coh.numel() < 50:
            return 15.0

        coh_thresh = torch.quantile(roi_coh, 0.65)
        valid_mask = t_roi & (theta_acute >= 8.0) & (theta_acute <= 38.0) & (coherency >= coh_thresh)

        if valid_mask.sum().item() < 30:
            valid_mask = t_roi & (theta_acute >= 6.0) & (theta_acute <= 42.0)
            if valid_mask.sum().item() < 10:
                return 15.0

        v_acute = theta_acute[valid_mask]
        v_coh = coherency[valid_mask]
        v_energy = torch.sqrt(trace[valid_mask] + 1e-8)
        v_sign = slope_sign[valid_mask]

        notch = 1.0 - torch.exp(-((v_acute - 7.0) ** 2) / 6.0)
        weights = (v_coh ** 2) * v_energy * notch
        weights = weights / (weights.sum() + 1e-8)

        grid = torch.linspace(8.0, 38.0, 301, device=self.device)
        delta_rad = torch.deg2rad(grid.unsqueeze(1) - v_acute.unsqueeze(0))
        vm_scores = torch.sum(weights.unsqueeze(0) * torch.exp(30.0 * torch.cos(2.0 * delta_rad)), dim=1)
        fiber_angle = float(grid[torch.argmax(vm_scores)].item())

        dom_sign = 1.0 if float(torch.sum(weights * v_sign).item()) >= 0.0 else -1.0
        if dom_sign * theta_deep < 0.0:
            pa_deg = fiber_angle + abs(theta_deep)
        else:
            pa_deg = abs(fiber_angle - abs(theta_deep))

        return float(np.clip(pa_deg, 7.0, 40.0))

    def solve_convergent_fascicle_length(self, mt_mm, pa_deg, phi_deg, curvature_k=0.0):
        eff_angle_deg = max(6.5, pa_deg - 0.5 * phi_deg)
        fl_linear = mt_mm / math.sin(math.radians(eff_angle_deg))
        if curvature_k > 0.0:
            u = 0.5 * curvature_k * fl_linear
            sinc_corr = 1.0 + (u * u) / 6.0
            fl_mm = fl_linear * sinc_corr
        else:
            fl_mm = fl_linear
        return float(np.clip(fl_mm, 30.0, 195.0))

    def __call__(self, image, apo_mask, override_px_per_mm=None):
        gray = self._to_grayscale_uint8(image)
        H, W = gray.shape[:2]

        rho = float(override_px_per_mm) if override_px_per_mm is not None else self.decode_graticule_scale(gray)
        geom = self.extract_aponeurosis_geometry(apo_mask, H, W, rho)

        mt_mm = geom["mt_mm"]
        theta_sup = geom["theta_sup"]
        theta_deep = geom["theta_deep"]
        phi_deg = geom["phi"]

        sigma_i = float(np.clip(0.25 * rho, 2.5, 4.5))
        pa_deg = self.compute_structure_tensor_pa(
            gray_img=gray,
            roi_mask=geom["roi_mask"],
            theta_deep=theta_deep,
            sigma_d=1.2,
            sigma_i=sigma_i,
        )

        fl_mm = self.solve_convergent_fascicle_length(mt_mm=mt_mm, pa_deg=pa_deg, phi_deg=phi_deg)

        return {
            "px_per_mm": round(rho, 4),
            "theta_sup_deg": round(theta_sup, 4),
            "theta_deep_deg": round(theta_deep, 4),
            "phi_deg": round(phi_deg, 4),
            "mt_mm": round(mt_mm, 4),
            "pa_deg": round(pa_deg, 4),
            "fl_mm": round(fl_mm, 4),
        }
