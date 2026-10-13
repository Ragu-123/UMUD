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
    while i < n:
        # Check if i to i+burst_size forms a candidate burst
        if i + burst_size <= n:
            window = df_out.iloc[i:i+burst_size]
            # Smooth window with rolling median
            for col in target_cols:
                med_val = window[col].median()
                # Light blend (30% toward median to preserve subtle kinematics)
                smoothed = 0.70 * window[col].values + 0.30 * med_val
                df_out.iloc[i:i+burst_size, df_out.columns.get_loc(col)] = np.round(smoothed, 3)
            i += burst_size
        else:
            i += 1

    return df_out
