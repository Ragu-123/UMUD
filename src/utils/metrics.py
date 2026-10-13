import numpy as np
import pandas as pd


def compute_umud_score(solution_df, submission_df, row_id_col="image_id",
                       tau_pa=6.0, tau_fl=12.0, tau_mt=3.0):
    """
    Official UMUD competition evaluation score from host Paul Ritsche.
    Normalized composite MAE with deterministic tie-breaking.
    """
    target_cols = ("pa_deg", "fl_mm", "mt_mm")
    tolerances = {"pa_deg": tau_pa, "fl_mm": tau_fl, "mt_mm": tau_mt}
    weights = {"pa_deg": 1.0, "fl_mm": 1.0, "mt_mm": 1.0}

    merged = solution_df.merge(
        submission_df,
        on=row_id_col,
        how="inner",
        suffixes=("_true", "_pred"),
    )

    if len(merged) != len(solution_df):
        raise ValueError(f"Merged length ({len(merged)}) != solution length ({len(solution_df)})")

    primary = 0.0
    secondary = 0.0
    tertiary = 0.0
    w_sum = sum(weights.values())

    for c in target_cols:
        y_true = merged[f"{c}_true"].to_numpy(dtype=float)
        y_pred = merged[f"{c}_pred"].to_numpy(dtype=float)

        tau = float(tolerances[c])
        w = float(weights[c])

        mae = float(np.mean(np.abs(y_pred - y_true)))
        medae = float(np.median(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

        primary += w * (mae / tau)
        secondary += w * (medae / tau)
        tertiary += w * (rmse / tau)

    primary /= w_sum
    secondary /= w_sum
    tertiary /= w_sum

    return float(primary + 1e-6 * secondary + 1e-9 * tertiary)
