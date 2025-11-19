import numpy as np
from numpy.ma.core import left_shift
from pandas import Series
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from numpy import ndarray
import pandas as pd
from scipy import stats
from typing import Any, Dict, List, Tuple, Optional, Sequence, Union
# =============================================================================
# =============================================================================
# =============================================================================
# Custom scoring and threshold utilities for Youden's J statistic
# =============================================================================
from sklearn.metrics import roc_curve
def youdens_j_statistic(y_true, y_prob):
    # Calculate Youden's J statistic and return (J, threshold)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    return j_scores[idx], thresholds[idx]



def youdens_j_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Compute Youden's J statistic and the corresponding optimal threshold (tau)
    from continuous prediction scores.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_score : np.ndarray
        Predicted continuous scores (e.g., probabilities for the positive class).

    Returns
    -------
    J_opt : float
        Maximum Youden's J statistic (sensitivity - false positive rate).
    tau_opt : float
        Threshold corresponding to J_opt.
    """
    fpr, tpr, thr = roc_curve(y_true, y_score)
    J = tpr - fpr
    ix = int(np.argmax(J))
    return float(J[ix]), float(thr[ix])


def metrics_at_threshold(y_true: ndarray,
                         y_score: ndarray,
                         tau: float) -> Dict[str, float]:
    """
    Return dict of sensitivity, specificity, Youden's J statistic,
    confusion matrix, AUC score, and PRC (average precision) score
    at threshold tau.
    """
    # Predictions at given threshold
    y_pred = (y_score >= tau).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity & specificity (proportions)
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    # Youden's J statistic (proportion scale)
    youden_j = sens + spec - 1

    # AUC score (threshold-independent)
    auc_score = roc_auc_score(y_true, y_score)

    # PRC score (average precision)
    prc_score = average_precision_score(y_true, y_score)

    return {
        "thr": float(tau),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "sensitivity": round(sens * 100, 3),   # percentage
        "specificity": round(spec * 100, 3),   # percentage
        "youden_j": round(youden_j, 3),        # proportion
        "auc_score": round(auc_score, 5),
        "prc_score": round(prc_score, 5)
    }


def generate_ci(
    df_metrics: pd.DataFrame,
    confidence: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute mean ± CI for selected metrics for each (model_type, optimization) pair
    and merge them into df_metrics. Handles metrics that may be in [0,1] or [0,100].

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Metrics of each fold and each model.
        Must contain 'model_type' and 'optimization' columns.
    confidence : float, optional
        Confidence level for the CI (default 0.95).

    Returns
    -------
    df_metrics_with_ci : pd.DataFrame
        Original df_metrics with CI columns merged.
    df_ci : pd.DataFrame
        Table of mean ± CI strings for each (model_type, optimization) pair.
    """
    from scipy import stats
    import numpy as np

    def _mean_ci_str(series: pd.Series) -> str:
        data = series.dropna().astype(float)
        n = len(data)
        if n <= 1:
            return np.nan

        # Detect scale: assume percentage if mean > 1.5 (arbitrary cutoff)
        scale = 100.0 if data.mean() > 1.5 else 1.0
        data_scaled = data / scale

        mean = data_scaled.mean()
        sem = stats.sem(data_scaled)
        h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)

        low, high = mean - h, mean + h
        # Clip bounds to [0,1]
        low = max(0.0, low)
        high = min(1.0, high)

        # Format back in original scale
        mean_out, low_out, high_out = mean * scale, low * scale, high * scale

        if scale == 1.0:
            return f"{mean_out:.3f} ({low_out:.3f}, {high_out:.3f})"
        else:  # percentage
            return f"{mean_out:.1f}% ({low_out:.1f}%, {high_out:.1f}%)"

    # Select metric columns of interest
    metrics_ci = [
        col for col in df_metrics.columns
        if any(key in col for key in ("sensitivity", "specificity", "auc", "prc"))
    ]

    # Compute CI table for all (model_type, optimization) pairs
    df_ci = (
        df_metrics
        .groupby(['model_type', 'optimization'])[metrics_ci]
        .agg(lambda x: _mean_ci_str(x))
        .rename(columns={col: f"{col}_ci" for col in metrics_ci})
        .reset_index()
    )

    # Merge CI table back into original metrics
    df_metrics_with_ci = df_metrics.merge(
        df_ci,
        on=["model_type", "optimization"],
        how="left"
    )

    return df_metrics_with_ci, df_ci


def compute_ci_from_folds_average(
        df_metrics_per_fold: pd.DataFrame,
        group_cols: List[str] = None,
        col_metrics: List[str] = None,
        suffix: str = "_ci"
) -> pd.DataFrame:
    """
    Compute mean ± 95% CI across folds for each metric, then compress each metric
    into a single formatted column: '{mean} ({ci_lower}, {ci_upper})'.

    Auto-detects scale per metric *and per group*:
      - If group mean > 1.5, treat as percentage and print like '88.2% (82.3%, 97.4%)'.
      - Else treat as fraction and print like '0.889 (0.832, 0.947)'.

    If a group has n <= 1, prints 'mean (NA, NA)'.
    """
    if group_cols is None:
        group_cols = ["model_type", "optimization", "threshold"]
    if col_metrics is None:
        col_metrics = ["auc_score", "prc_score", "sensitivity", "specificity"]

    df = df_metrics_per_fold.copy()

    # Ensure numeric metrics
    for c in col_metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    grp = df.groupby(group_cols, dropna=False)[col_metrics]

    means = grp.mean()
    counts = grp.count()
    stds = grp.std(ddof=1)

    # 95% CI half-width (two-sided)
    tcrit = stats.t.ppf(0.975, df=counts - 1)
    sem = stds / np.sqrt(counts)
    half_width = sem * tcrit

    ci_lower = means - half_width
    ci_upper = means + half_width

    # Helper: clamp to [0, 1] or [0, 100] depending on scale
    def _clamp(v: float, scale: float) -> float:
        if pd.isna(v):
            return v
        lo, hi = (0.0, 100.0) if scale == 100.0 else (0.0, 1.0)
        return min(max(v, lo), hi)

    # Helper: format a value according to scale
    def _fmt_val(v: float, scale: float) -> str:
        if pd.isna(v):
            return "NA"
        if scale == 100.0:
            return f"{v:.1f}%"
        return f"{v:.3f}"

    import re

    def _extract_first_numeric(series: pd.Series) -> pd.Series:
        return series.apply(
            lambda s: re.search(r'[-+]?\d*\.\d+|\d+', s).group() if isinstance(s, str) and re.search(
                r'[-+]?\d*\.\d+|\d+', s) else None
        )


    # Build output with group columns
    out = means.reset_index()[group_cols].copy()
    idx_iter = means.index  # MultiIndex over group_cols

    # Compose one '{mean} ({low}, {high})' column per metric
    for m in col_metrics:
        formatted = []
        for idx in idx_iter:
            mean_v = float(means.loc[idx, m])
            low_v = float(ci_lower.loc[idx, m]) if not pd.isna(ci_lower.loc[idx, m]) else np.nan
            high_v = float(ci_upper.loc[idx, m]) if not pd.isna(ci_upper.loc[idx, m]) else np.nan
            n_v = counts.loc[idx, m]

            # Per-group scale detection (your heuristic)
            scale = 100.0 if (not pd.isna(mean_v) and mean_v > 1.5) else 1.0

            if pd.isna(n_v) or n_v <= 1:
                s = f"{_fmt_val(mean_v, scale)} (NA, NA)"
            else:
                low_v = _clamp(low_v, scale)
                high_v = _clamp(high_v, scale)
                s = f"{_fmt_val(mean_v, scale)} ({_fmt_val(low_v, scale)}, {_fmt_val(high_v, scale)})"

            formatted.append(s)

        out[f"{m}{suffix}"] = formatted

    # make the average column
    for m in col_metrics:
        out[m] = _extract_first_numeric(out[m+suffix]).astype(float)

    return out


# %% Post-training selection
def select_best_model_type_per_config(
    df_metrics: pd.DataFrame,
    min_spec: float = 98.0,
    min_sens: float = 70.0,
    prefer_thresholds: Tuple[str, ...] = ("spec_max", "sens_max", "youden", "0p5"),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns exactly ONE row per config (the chosen model_type + its best threshold row).
    Selection within each config:
      1) Highest specificity
      2) Highest sensitivity
      3) Highest AUC (auc_score)
      4) Highest PRC (prc_score)
      5) Preferred threshold type order (spec_max > sens_max > youden > 0p5)

    Constraints (with fallbacks):
      - First try rows meeting BOTH min_spec & min_sens
      - else rows meeting min_spec only
      - else rows meeting min_sens only
      - else any row

    Accepts % or fraction inputs for metrics and mins.
    """
    needed = {"config", "model_type", "specificity", "sensitivity"}
    missing = needed - set(df_metrics.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df_metrics.copy()

    # normalize mins (allow 98.0 or 0.98)
    min_spec_fr = min_spec / 100.0 if min_spec > 1 else min_spec
    min_sens_fr = min_sens / 100.0 if min_sens > 1 else min_sens

    def _as_frac(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        return s / 100.0 if s.dropna().max() > 1.0 else s

    df["_spec"] = round(_as_frac(df["specificity"]), 4)
    df["_sens"] = round(_as_frac(df["sensitivity"]), 4)
    df["_auc"]  = pd.to_numeric(df.get("auc_score", np.nan), errors="coerce")
    df["_prc"]  = pd.to_numeric(df.get("prc_score", np.nan), errors="coerce")

    # filter metrics that uses the thresholds we want
    thr_rank = {t: i for i, t in enumerate(prefer_thresholds)}
    df["_thr_rank"] = df.get("threshold", "zzz").map(thr_rank).fillna(99).astype(int)

    # Step 1: choose best row within each (config, model_type)
    def _best_row_in_model(g: pd.DataFrame) -> pd.DataFrame:
        c1 = g[(g["_spec"] >= min_spec_fr) & (g["_sens"] >= min_sens_fr) ]
        c2 = g[(g["_spec"] >= min_spec_fr)]
        c3 = g[(g["_sens"] >= min_sens_fr)]
        cand = c1 if not c1.empty else (c2 if not c2.empty else (c3 if not c3.empty else g))
        return cand.sort_values(
            ["_spec", "_sens",], # "_auc", "_prc", "_thr_rank"],
            ascending=[False, False] #, False, False, True]
        ).head(1)

    df_per_model_best = (
        df.groupby(["config", "model_type"], dropna=False, group_keys=False)
          .apply(_best_row_in_model)
          .reset_index(drop=True)
    )

    # Step 2: pick ONE model_type per config
    def _pick_for_config(g: pd.DataFrame) -> pd.DataFrame:
        # same ranking at the config level
        win = g.sort_values(
            ["_spec", "_sens"], #"_auc", "_prc", "_thr_rank"],
            ascending=[False, False] # False, False, True]
        ).head(1).copy()
        return win

    df_selected = (
        df_per_model_best.groupby("config", dropna=False, group_keys=False)
                      .apply(_pick_for_config)
                      .reset_index(drop=True)
    )

    # cleanup helpers
    df_selected.drop(columns=[c for c in df_selected.columns if c.startswith("_")], inplace=True, errors="ignore")

    # sanity: exactly one per config
    assert df_selected["config"].is_unique, "More than one row per config selected."
    return df_selected.copy(), df_per_model_best.copy()

