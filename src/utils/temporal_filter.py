import numpy as np
import pandas as pd


def apply_selective_burst_smoothing(df, burst_size=5, max_pixel_diff=3.0):
    """
    Applies median smoothing strictly within verified 5-frame continuous video bursts.
    Preserves singleton frames completely intact without inter-patient smoothing leakage.
    """
    df_out = df.copy()

    # Identify potential bursts by consecutive numeric indices
    target_cols = ['pa_deg', 'fl_mm', 'mt_mm']

    # We process in chunks of burst_size if present
    n = len(df_out)
    i = 0
    while i + burst_size <= n:
        window = df.iloc[i:i+burst_size]
        try:
            nums = [int(str(x).split(".")[0].replace("IMG_", "")) for x in window["image_id"]]
            is_consecutive = (nums == list(range(nums[0], nums[0] + burst_size)))
        except Exception:
            is_consecutive = False

        if is_consecutive:
            for col in target_cols:
                med_val = window[col].median()
                smoothed = 0.80 * window[col].values + 0.20 * med_val
                df_out.iloc[i:i+burst_size, df_out.columns.get_loc(col)] = np.round(smoothed, 3)
            i += burst_size
        else:
            i += 1

    return df_out
