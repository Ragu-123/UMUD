import numpy as np


class BiomechanicalMetrologyEngine:
    """
    Unites aponeurosis geometry, structure tensor orientation, and the
    generalized planar convergence identity to estimate (PA, FL, MT).
    """
    def __init__(self, pa_bounds=(5.0, 45.0), fl_bounds=(30.0, 200.0), mt_bounds=(10.0, 50.0)):
        self.pa_bounds = pa_bounds
        self.fl_bounds = fl_bounds
        self.mt_bounds = mt_bounds

    def compute_parameters(self, mt_mm, raw_fiber_angle, slope_deep, phi):
        """
        Computes pennation angle (relative to deep aponeurosis) and fascicle length
        via the generalized planar triangle identity: FL = MT / sin(PA - phi/2).
        """
        # 1. Pennation angle against deep aponeurosis
        pa_deg = float(abs(raw_fiber_angle - slope_deep))

        # 2. Fascicle length via planar convergence identity
        effective_angle_deg = max(4.0, pa_deg - (phi / 2.0))
        effective_angle_rad = np.radians(effective_angle_deg)
        sin_val = np.sin(effective_angle_rad)

        fl_mm = float(mt_mm / max(0.08, sin_val))

        # 3. Clip within physiological human musculoskeletal limits
        pa_deg = float(np.clip(pa_deg, self.pa_bounds[0], self.pa_bounds[1]))
        fl_mm = float(np.clip(fl_mm, self.fl_bounds[0], self.fl_bounds[1]))
        mt_mm = float(np.clip(mt_mm, self.mt_bounds[0], self.mt_bounds[1]))

        return pa_deg, fl_mm, mt_mm
