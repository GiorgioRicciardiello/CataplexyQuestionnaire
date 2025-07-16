"""threshold_pipeline_veto.py
================================
End‑to‑end pipeline for selecting probability thresholds **with and without** the
HLA‑based veto rule and evaluating model performance across CV folds.


All intermediate CSVs are saved to *--outdir* so they can be reused by other
notebooks or plotting scripts.
"""

from config.config import config
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from roc_curve_plots import (
    compute_best_thresholds,
    compute_metrics_across_folds_with_best_thresholds_and_veto,
    _compute_metrics,
)


# def collect_best_thresholds_for_models(
#         df: pd.DataFrame,
#         alpha_values: np.ndarray,
#         prevalence: float = 30 / 100000,
#         dataset_type: str = 'full',
#         sensitivity_threshold: float = 0.90,
#         specificity_threshold: float = 0.98,
#         veto:bool=False
# ) -> pd.DataFrame:
#     """
#     For each (config, model_name), returns the best alpha and corresponding threshold
#     that satisfies the input sensitivity and specificity requirements.
#     If none meets both, the pair is skipped.
#
#     Using the PFR criteria and not the Youden index
#     Returns:
#         DataFrame with columns:
#         - config
#         - model_name
#         - best_alpha
#         - best_threshold
#         - sensitivity
#         - specificity
#     """
#     records = []
#     configs = df['config'].unique()
#     model_names = df['model_name'].unique()
#
#     for config in configs:
#         df_config = df[df['config'] == config]
#
#         for model in model_names:
#             df_model = df_config[df_config['model_name'] == model]
#             y_true = df_model['true_label'].to_numpy()
#             y_prob = df_model['predicted_prob'].to_numpy()
#
#             if len(y_true) == 0 or pd.isna(y_prob).all():
#                 continue
#
#             for alpha in alpha_values:
#                 df_thresholds = compute_best_thresholds(
#                     df=df_model,
#                     target_fpr=alpha,
#                     prevalence=prevalence,
#                     dataset_type=dataset_type,
#                     apply_veto=veto,
#                 )
#
#                 if df_thresholds.empty:
#                     continue
#
#                 specificity = df_thresholds['specificity_fpr'].iloc[0]
#                 sensitivity = df_thresholds['sensitivity_fpr'].iloc[0]
#                 best_threshold = df_thresholds['best_threshold_fpr'].iloc[0]
#                 auc = df_thresholds['auc'].iloc[0]
#
#                 if specificity >= specificity_threshold and sensitivity >= sensitivity_threshold:
#                     records.append({
#                         'config': config,
#                         'model_name': model,
#                         'best_alpha': alpha,
#                         'best_threshold': best_threshold,
#                         'sensitivity': sensitivity,
#                         'specificity': specificity,
#                         'auc':auc,
#                     })
#                     break  # Stop after the first valid alpha
#     df_records = pd.DataFrame(records)
#     df_records = df_records.sort_values(by=['config', 'model_name']).reset_index(drop=True)
#     return df_records


def collect_best_thresholds_for_models(
    df: pd.DataFrame,
    alpha_values: np.ndarray,
    prevalence: float = 30 / 100_000,
    dataset_type: str = "full",
    veto: bool = False,
    collect_by:str='specificity',
    spec_ref:float=0.99
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For every (config, model_name) × α (FPR cap), call `compute_best_thresholds`
    to obtain the threshold τ* that maximises sensitivity while keeping
    post-veto FPR ≤ α.  **All** (α, τ*) pairs are recorded.

    Selection rule (per config, model):
        1. Keep the row with the highest **Youden’s J**
           (J = sensitivity + specificity − 1).
        2. If J ties, prefer the row with higher **specificity**.
        3. If J and specificity tie, prefer higher **sensitivity**.

    Returns
    -------
    df_records : pd.DataFrame
        All (α, τ*) rows, columns =
          ['config', 'model_name', 'alpha', 'threshold',
           'sensitivity', 'specificity', 'auc', 'youden_j']
    df_best : pd.DataFrame
        One “best” row per (config, model_name) chosen by the rule above.
    """
    recs: List[Dict] = []
    if not collect_by in ['specificity', 'youden']:
        raise ValueError(f'collect_by must be one of {["specificity", "youden"]}')

    # ------------------------------------------------------------------ #
    # Iterate over each config / model
    # ------------------------------------------------------------------ #
    for (cfg, model), g in df.groupby(["config", "model_name"]):
        # Restrict to requested dataset_type
        g = g[g["dataset_type"] == dataset_type]
        if g.empty:
            continue

        for alpha in alpha_values:
            df_thr = compute_best_thresholds(
                df=g,
                target_fpr=alpha,
                prevalence=prevalence,
                dataset_type=dataset_type,
                apply_veto=veto,
            )

            if df_thr.empty:
                continue

            # `compute_best_thresholds` returns a single-row DF
            sens = float(df_thr["sensitivity_fpr"].iloc[0])
            spec = float(df_thr["specificity_fpr"].iloc[0])
            thr  = float(df_thr["best_threshold_fpr"].iloc[0])
            auc  = float(df_thr["auc"].iloc[0])

            recs.append(
                {
                    "config":       cfg,
                    "model_name":   model,
                    "alpha":        float(alpha),
                    "threshold":    thr,
                    "sensitivity":  sens,
                    "specificity":  spec,
                    "auc":          auc,
                    "youden_j":     sens + spec - 1,
                }
            )

    # ------------------------------------------------------------------ #
    # Assemble all rows
    # ------------------------------------------------------------------ #
    df_records = (
        pd.DataFrame(recs)
          .sort_values(["config", "model_name"])
          .reset_index(drop=True)
    )

    if df_records.empty:
        # No valid (α, τ) found for any model
        return df_records, df_records.copy()

    if collect_by == 'specificity':
        def _pick_closest(
                df: pd.DataFrame,
                sens_min: float,  # minimum acceptable sensitivity
                spec_ref: float,  # target (reference) specificity
        ) -> pd.DataFrame:
            """
            For every (config, model_name) keep **one** row:

            1.  Filter rows with sensitivity ≥ sens_min
                (rows below this cut-off are discarded).

            2.  Among the remaining rows, compute the absolute distance
                |specificity − spec_ref|.

            3.  Pick the row with the **smallest distance** for each
                (config, model_name).  If there is a tie, keep the row with the
                **highest sensitivity**, then the highest specificity.

            Parameters
            ----------
            df : pd.DataFrame
                Must contain columns ['config', 'model_name',
                'specificity', 'sensitivity'].
            sens_min : float
                Minimum sensitivity required for a row to be eligible.
            spec_ref : float
                Desired (reference) specificity.

            Returns
            -------
            pd.DataFrame
                One “best” row per (config, model_name).
            """
            if df.empty:
                return df.copy()

            # -------------------------------------------------------------
            # 1. Keep only rows with sensitivity ≥ sens_min
            # -------------------------------------------------------------
            df_eligible = df[df["sensitivity"] >= sens_min].copy()
            if df_eligible.empty:  # nothing meets the sensitivity cut-off
                return df_eligible  # returns empty DF with same columns

            # -------------------------------------------------------------
            # 2. Distance to reference specificity
            # -------------------------------------------------------------
            df_eligible["dist"] = (df_eligible["specificity"] - spec_ref).abs()

            # -------------------------------------------------------------
            # 3. Select the closest row per (config, model_name)
            #    Tie-break: higher sensitivity, then higher specificity
            # -------------------------------------------------------------
            df_best = (
                df_eligible
                .sort_values(
                    ["config", "model_name", "dist",
                     "sensitivity", "specificity"],
                    ascending=[True, True, True, False, False],
                )
                .groupby(["config", "model_name"], as_index=False)
                .first()  # keep the top-ranked row
                .drop(columns="dist")
            )

            return df_best

        df_best = _pick_closest(
            df_records,
            sens_min=0.80,  # sensitivity must be ≥ 0.80
            spec_ref=0.99,  # pick the row whose specificity is closest to 0.99
        )

    else:
        df_best = (
            df_records
              .sort_values(
                  ["config", "model_name",
                   "youden_j",          # ① highest J
                   "specificity",       # ② highest specificity
                   "sensitivity"],      # ③ highest sensitivity
                  ascending=[True, True, False, False, False],
              )
              .groupby(["config", "model_name"], as_index=False)
              .first()                         # keep top-ranked row
              .drop(columns="youden_j")
        )

    return df_records, df_best


# def collect_best_thresholds_with_veto(
#     df: pd.DataFrame,
#     alpha_values: np.ndarray,
#     prevalence: float,
#     dataset_type: str = "full",
#     sensitivity_threshold: float = 0.95,
#     specificity_threshold: float = 0.99
# ) -> pd.DataFrame:
#     """Same as above but evaluates each τ after applying the HLA veto."""
#     df_ = df[df["dataset_type"] == dataset_type].copy()
#     df_["hla_results"] = df_["hla_results"].astype(int)
#     df_["true_label"] = df_["true_label"].astype(int)
#
#     recs: List[Dict] = []
#     for (cfg, model), g in df_.groupby(["config", "model_name"]):
#         y_true = g["true_label"].to_numpy()
#         y_prob = g["predicted_prob"].to_numpy()
#         hla_neg = g["hla_results"].to_numpy() == 0
#
#         for alpha in alpha_values:
#             fpr, tpr, thr = roc_curve(y_true, y_prob)
#             idx_ok = np.where(fpr <= alpha)[0]
#             if idx_ok.size == 0:
#                 continue
#             idx_best = idx_ok[np.argmax(tpr[idx_ok])]
#             tau = thr[idx_best]
#
#             pred = (y_prob >= tau).astype(int)
#             # veto FP that are HLA negative
#             pred[(pred == 1) & (y_true == 0) & hla_neg] = 0
#
#             m = _compute_metrics(pred, y_true, prevalence)
#             spec, sens, auc = m["specificity"], m["sensitivity"], m["auc"]
#             if spec >= specificity_threshold and sens >= sensitivity_threshold:
#                 recs.append(
#                     {
#                         "config": cfg,
#                         "model_name": model,
#                         "best_alpha_veto": alpha,
#                         "best_threshold_veto": tau,
#                         "sensitivity_veto": sens,
#                         "specificity_veto": spec,
#                         'auc_veto': auc,
#                     }
#                 )
#
#     df_records = pd.DataFrame(recs)
#     df_records = df_records.sort_values(by=['config', 'model_name']).reset_index(drop=True)
#
#     def _pick_closest(df: pd.DataFrame,
#                       sens_ref: float,
#                       spec_ref: float) -> pd.DataFrame:
#         """
#         From df that may contain several τ/α pairs per (config, model_name),
#         keep the row with the smallest L1-distance to the target
#         (specificity_threshold, sensitivity_threshold).
#         """
#         if df.empty:
#             return df
#
#         target = np.array([spec_ref, sens_ref])
#
#         # Compute Manhattan distance to the target point
#         df["dist"] = (df[["specificity_veto", "sensitivity_veto"]]
#                       .sub(target)
#                       .abs()
#                       .sum(axis=1))
#
#         df_best = (df.sort_values(["config", "model_name", "dist"])
#                    .groupby(["config", "model_name"], as_index=False)
#                    .first()  # keep the row with the smallest distance
#                    .drop(columns="dist"))
#
#         return df_best
#
#     df_best = _pick_closest(df_records,
#                             sens_ref=sensitivity_threshold,
#                             spec_ref=specificity_threshold)
#
#     return df_records


def collect_best_thresholds_with_veto(
    df: pd.DataFrame,
    alpha_values: np.ndarray,
    prevalence: float,
    dataset_type: str = "full",
) -> pd.DataFrame:
    """
    Evaluate every τ (threshold) under the HLA veto and return, for each
    (config, model_name), the pair (α, τ) that maximises Youden’s J
    (sensitivity + specificity − 1).  No fixed cut-offs are applied.
    """
    recs: List[Dict] = []

    df_ = df[df["dataset_type"] == dataset_type].copy()
    df_[["hla_results", "true_label"]] = df_[["hla_results", "true_label"]].astype(int)

    for (cfg, model), g in df_.groupby(["config", "model_name"]):
        y_true = g["true_label"].to_numpy()
        y_prob = g["predicted_prob"].to_numpy()
        hla_neg = g["hla_results"].to_numpy() == 0

        # roc_curve returns one set of points per call, so loop on α
        for alpha in alpha_values:
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            idx_ok = np.where(fpr <= alpha)[0]
            if idx_ok.size == 0:
                continue

            idx_best = idx_ok[np.argmax(tpr[idx_ok])]
            tau = thr[idx_best]

            # Hard calls + veto
            pred = (y_prob >= tau).astype(int)
            pred[(pred == 1) & (y_true == 0) & hla_neg] = 0

            m = _compute_metrics(pred, y_true, prevalence)

            recs.append({
                "config":            cfg,
                "model_name":        model,
                "alpha_veto":        alpha,
                "threshold_veto":    tau,
                "sensitivity_veto":  m["sensitivity"],
                "specificity_veto":  m["specificity"],
                "auc_veto":          m["auc"],
                "youden_j":          m["sensitivity"] + m["specificity"] - 1,
            })

    df_records = pd.DataFrame(recs)

    # ---- pick “best” row per (config, model_name) --------------------------
    if df_records.empty:
        return df_records  # nothing matched the ROC-α condition

    df_best = (
        df_records
        .sort_values(
            ["config", "model_name", "youden_j", "sensitivity_veto", "specificity_veto"],
            ascending=[True, True, False, False, False]  # largest Youden J first
        )
        .groupby(["config", "model_name"], as_index=False)
        .first()              # keep the row with the highest Youden J
        .drop(columns="youden_j")
    )

    return df_best


def collect_best_thresholds_with_veto_new(
    df: pd.DataFrame,
    alpha_values: np.ndarray,
    prevalence: float,
    dataset_type: str = "full",
    sensitivity_threshold: float = 0.95,
    specificity_threshold: floa = 0.99
) -> pd.DataFrame:
    """Same as above but evaluates each τ after applying the HLA veto."""
    df_ = df[df["dataset_type"] == dataset_type].copy()
    df_["hla_results"] = df_["hla_results"].astype(int)
    df_["true_label"] = df_["true_label"].astype(int)

    recs: List[Dict] = []
    for (cfg, model), g in df_.groupby(["config", "model_name"]):
        y_true = g["true_label"].to_numpy()
        y_prob = g["predicted_prob"].to_numpy()
        hla_neg = g["hla_results"].to_numpy() == 0

        for alpha in alpha_values:
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            idx_ok = np.where(fpr <= alpha)[0]
            if idx_ok.size == 0:
                continue
            idx_best = idx_ok[np.argmax(tpr[idx_ok])]
            tau = thr[idx_best]

            pred = (y_prob >= tau).astype(int)
            # veto FP that are HLA negative
            pred[(pred == 1) & (y_true == 0) & hla_neg] = 0

            m = _compute_metrics(pred, y_true, prevalence)
            spec, sens, ppv= m["specificity"], m["sensitivity"], m['ppv']
            if spec >= specificity_threshold and sens >= sensitivity_threshold:
                recs.append(
                    {
                        "config": cfg,
                        "model_name": model,
                        "best_alpha_veto": alpha,
                        "best_threshold_veto": tau,
                        "sensitivity_veto": sens,
                        "specificity_veto": spec,
                    }
                )

    df_records = pd.DataFrame(recs)
    df_records = df_records.sort_values(by=['config', 'model_name']).reset_index(drop=True)

    def _pick_closest(df: pd.DataFrame,
                      sens_ref: float,
                      spec_ref: float) -> pd.DataFrame:
        """
        From df that may contain several τ/α pairs per (config, model_name),
        keep the row with the smallest L1-distance to the target
        (specificity_threshold, sensitivity_threshold).
        """
        if df.empty:
            return df

        target = np.array([spec_ref, sens_ref])

        # Compute Manhattan distance to the target point
        df["dist"] = (df[["specificity_veto", "sensitivity_veto"]]
                      .sub(target)
                      .abs()
                      .sum(axis=1))

        df_best = (df.sort_values(["config", "model_name", "dist"])
                   .groupby(["config", "model_name"], as_index=False)
                   .first()  # keep the row with the smallest distance
                   .drop(columns="dist"))

        return df_best

    df_best = _pick_closest(df_records,
                            sens_ref=sensitivity_threshold,
                            spec_ref=specificity_threshold)

    return df_records


# def collect_best_thresholds_with_veto_long(
#     df: pd.DataFrame,
#     alpha_values: np.ndarray,
#     prevalence: float,
#     dataset_type: str = "full",
#     sensitivity_threshold: float = 0.90,
#     specificity_threshold: float = 0.99,
#     apply_veto:bool=False
# ) -> pd.DataFrame:
#     """
#     For each (config, model_name) × α find the τ that maximises sensitivity
#     while keeping post-veto FPR ≤ α.  Record the (α, τ) pair only if the
#     resulting Sens ≥ sensitivity_threshold **and** Spec ≥ specificity_threshold.
#     """
#     df_ = df.loc[
#         (df["dataset_type"] == dataset_type)
#         &~df["config"].str.contains('DQB1\*06:0', regex=True)
#         ].copy()
#
#     # Ensure numeric types
#     df_[["hla_results", "true_label"]] = df_[["hla_results", "true_label"]].astype(int)
#
#     recs: list[dict] = []
#
#     for (cfg, model), g in df_.groupby(["config", "model_name"]):
#         y_true = g["true_label"].to_numpy()
#         y_prob = g["predicted_prob"].to_numpy()
#         hla_neg = g["hla_results"].to_numpy() == 0   # boolean mask
#
#         # 1. Generate a list of unique thresholds (descending)
#         #    NB: add -inf so every α has at least one τ to choose.
#         thresh_grid = np.unique(y_prob)[::-1]
#         thresh_grid = np.r_[-np.inf, thresh_grid]
#
#         # Pre-allocate metric arrays
#         tpr_veto = np.empty_like(thresh_grid, dtype=float)
#         fpr_veto = np.empty_like(thresh_grid, dtype=float)
#
#         # 2. Compute TPR & FPR after veto for every τ
#         for i, tau in enumerate(thresh_grid):
#             y_pred = (y_prob >= tau).astype(int)
#             if apply_veto:
#                 # y_pred[hla_neg] = 0                     # unconditional post-hoc veto
#                 y_pred[(y_true == 1) & hla_neg] = 0                     # conditional post-hoc veto
#             tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#             fpr_veto[i] = fp / (fp + tn) if fp + tn else 0.0
#             tpr_veto[i] = tp / (tp + fn) if tp + fn else 0.0
#
#         # 3. For each α pick the τ with maximal TPR subject to FPR≤α
#         for alpha in alpha_values:
#             ok = np.where(fpr_veto <= alpha)[0]
#             if ok.size == 0:
#                 continue
#             idx = ok[tpr_veto[ok].argmax()]
#             tau_star = thresh_grid[idx]
#
#             # Final predictions & metrics at (α, τ*)
#             y_pred_star = (y_prob >= tau_star).astype(int)
#             if apply_veto:
#                 y_pred_star[hla_neg] = 0
#             m = _compute_metrics(y_pred_star, y_true, prevalence)
#
#             if (m["specificity"] >= specificity_threshold and
#                     m["sensitivity"] >= sensitivity_threshold):
#                 recs.append(
#                     {
#                         "config": cfg,
#                         "model_name": model,
#                         "best_alpha_veto": float(alpha),
#                         "best_threshold_veto": float(tau_star),
#                         "sensitivity_veto": m["sensitivity"],
#                         "specificity_veto": m["specificity"],
#                         "ppv_veto": m["ppv"],
#                         # "ppvprev_veto": m["ppvprev"],
#                     }
#                 )
#     df_records = pd.DataFrame(recs).sort_values(["config", "model_name"]).reset_index(drop=True)
#
#     def _pick_closest(df: pd.DataFrame,
#                       sens_ref: float,
#                       spec_ref: float) -> pd.DataFrame:
#         """
#         From df that may contain several τ/α pairs per (config, model_name),
#         keep the row with the smallest L1-distance to the target
#         (specificity_threshold, sensitivity_threshold).
#         """
#         if df.empty:
#             return df
#
#         target = np.array([spec_ref, sens_ref])
#
#         # Compute Manhattan distance to the target point
#         df["dist"] = (df[["specificity_veto", "sensitivity_veto"]]
#                       .sub(target)
#                       .abs()
#                       .sum(axis=1))
#
#         df_best = (df.sort_values(["config", "model_name", "dist"])
#                    .groupby(["config", "model_name"], as_index=False)
#                    .first()  # keep the row with the smallest distance
#                    .drop(columns="dist"))
#
#         return df_best
#
#     df_best = _pick_closest(df_records,
#                             sens_ref=sensitivity_threshold,
#                             spec_ref=specificity_threshold)
#
#     return df_records, df_best


def collect_best_thresholds_with_veto_long(
        df: pd.DataFrame,
        alpha_values: np.ndarray,
        prevalence: float,
        dataset_type: str = "full",
        apply_veto: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (config, model_name) × α (FPR cap) find the threshold τ that
    maximises sensitivity while keeping post-veto FPR ≤ α.

    ▸ Records **every** (α, τ) pair in `df_records`.
    ▸ For each (config, model_name) keeps the row that:
          1. maximises Youden’s J  =  sensitivity + specificity − 1
          2. breaks ties on *higher specificity*
          3. then on *higher sensitivity* (for deterministic output)
        and returns those winners in `df_best`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns
        ['dataset_type', 'config', 'model_name',
         'predicted_prob', 'true_label', 'hla_results'].
    alpha_values : 1-D array-like
        Array of FPR limits (α) to evaluate.
    prevalence : float
        Prevalence used by `_compute_metrics` for PPV adjustment.
    dataset_type : {"full", "train", "val", "test"}, default "full"
        Subset of data to evaluate.
    apply_veto : bool, default False
        Whether to apply the HLA veto (set FP with HLA-negative to 0).

    Returns
    -------
    df_records : pd.DataFrame
        One row per (config, model_name, α) with the best τ at that α.
    df_best : pd.DataFrame
        One “best” row per (config, model_name) chosen by the rule above.
    """
    # ------------------------------------------------------------------ #
    # 0.  subset & clean
    # ------------------------------------------------------------------ #
    df_ = df.loc[
        (df["dataset_type"] == dataset_type)
        & ~df["config"].str.contains(r"DQB1\*06:0", regex=True)
        ].copy()

    df_[["hla_results", "true_label"]] = df_[["hla_results", "true_label"]].astype(int)

    recs: List[Dict] = []

    # ------------------------------------------------------------------ #
    # 1. iterate over every (config, model)
    # ------------------------------------------------------------------ #
    for (cfg, model), g in df_.groupby(["config", "model_name"]):
        y_true = g["true_label"].to_numpy()
        y_prob = g["predicted_prob"].to_numpy()
        hla_neg = g["hla_results"].to_numpy() == 0

        # unique thresholds, descending, prepend -inf
        thresh_grid = np.r_[-np.inf, np.unique(y_prob)[::-1]]

        tpr_veto = np.empty_like(thresh_grid, dtype=float)
        fpr_veto = np.empty_like(thresh_grid, dtype=float)

        # ------------- compute TPR/FPR for every τ -------------------- #
        for i, tau in enumerate(thresh_grid):
            pred = (y_prob >= tau).astype(int)

            if apply_veto:
                # veto: flip FP that are HLA-negative to 0
                pred[(pred == 1) & (y_true == 0) & hla_neg] = 0

            tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
            fpr_veto[i] = fp / (fp + tn) if (fp + tn) else 0.0
            tpr_veto[i] = tp / (tp + fn) if (tp + fn) else 0.0

        # ------------- for each α pick τ with max TPR ----------------- #
        for alpha in alpha_values:
            idx = np.where(fpr_veto <= alpha)[0]
            if idx.size == 0:
                continue
            # τ* = argmax_TPR
            best_idx = idx[np.argmax(tpr_veto[idx])]
            tau_star = thresh_grid[best_idx]

            pred_star = (y_prob >= tau_star).astype(int)
            if apply_veto:
                pred_star[(pred_star == 1) & (y_true == 0) & hla_neg] = 0

            m = _compute_metrics(pred_star, y_true, prevalence)
            youden_j = m["sensitivity"] + m["specificity"] - 1

            recs.append(
                {
                    "config": cfg,
                    "model_name": model,
                    "alpha_veto": float(alpha),
                    "threshold_veto": float(tau_star),
                    "sensitivity_veto": m["sensitivity"],
                    "specificity_veto": m["specificity"],
                    "youden_j": youden_j,
                    "ppv_veto": m["ppv"],
                }
            )

    # ------------------------------------------------------------------ #
    # 2. assemble all rows
    # ------------------------------------------------------------------ #
    df_records = (
        pd.DataFrame(recs)
        .sort_values(["config", "model_name"])
        .reset_index(drop=True)
    )

    if df_records.empty:
        return df_records, df_records.copy()

    # ------------------------------------------------------------------ #
    # 3. pick the row with max Youden J, tie-break by specificity
    # ------------------------------------------------------------------ #
    df_best = (
        df_records
        .sort_values(
            ["config", "model_name",
             "youden_j",  # primary: max J
             "specificity_veto",  # 1st tie-break: max specificity
             "sensitivity_veto"  # 2nd tie-break: max sensitivity
             ],
            ascending=[True, True, False, False, False]
        )
        .groupby(["config", "model_name"], as_index=False)
        .first()  # keep top-ranked row
        .drop(columns="youden_j")
    )

    return df_records, df_best

# ---------------------------------------------------------------------------
# VISUALS – ROC plots marking the selected α / τ
# ---------------------------------------------------------------------------

def _plot_roc_with_selected_thresholds(
    df: pd.DataFrame,
    thr_tbl: pd.DataFrame,
    title_suffix: str,
    zoom: tuple = (0.06, 0.9),
    out_path: Path | None = None,
):
    """ROC for every (config, model) *with α / τ shown in the legend*.

    The helper expects *thr_tbl* to contain either:
      • columns `best_alpha`, `best_threshold`   (plain Stage I) or
      • columns `best_alpha_veto`, `best_threshold_veto` (Stag I′)
    """
    configs = sorted(df["config"].unique())
    models  = sorted(df["model_name"].unique())

    n_cols = 2
    n_rows = int(np.ceil(len(configs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    palette = sns.color_palette("tab10", len(models))
    model_colors = {m: palette[i] for i, m in enumerate(models)}

    # work out which column names exist for α / τ
    alpha_col = [c for c in thr_tbl.columns if c.startswith("best_alpha")][0]
    tau_col   = [c for c in thr_tbl.columns if c.startswith("best_threshold")][0]

    for i_cfg, cfg in enumerate(configs):
        ax = axes[i_cfg]
        for model in models:
            sub = df[(df["config"] == cfg) & (df["model_name"] == model)]
            if sub.empty:
                continue
            y_true = sub["true_label"].to_numpy()
            y_prob = sub["predicted_prob"].to_numpy()
            fpr, tpr, thr = roc_curve(y_true, y_prob)

            # default label if τ not found
            label_txt = model

            row_thr = thr_tbl[(thr_tbl["config"] == cfg) & (thr_tbl["model_name"] == model)]
            if not row_thr.empty:
                tau = row_thr[tau_col].iat[0]
                alpha_sel = row_thr[alpha_col].iat[0]
                idx_sel = np.argmin(np.abs(thr - tau))
                ax.scatter(fpr[idx_sel], tpr[idx_sel], color=model_colors[model], s=60, edgecolors="k")
                label_txt = (
                    fr'$\mathbf{{{model}}}$'
                    fr"(α={alpha_sel:.3f}, τ={tau:.3f}, "
                    fr"Sens={row_thr.filter(like='sensitivity').values[0][0]:.2f}, "
                    fr"Spec={row_thr.filter(like='specificity').values[0][0]:.2f})"
                )

            ax.plot(fpr, tpr, color=model_colors[model], lw=1.2, label=label_txt)

        ax.set_xlim([0, zoom[0]]); ax.set_ylim([zoom[1], 1.02])
        ax.set_title(cfg)
        ax.grid(True)
        if i_cfg % n_cols == 0:
            ax.set_ylabel("TPR")
        ax.set_xlabel("FPR")
        ax.legend(fontsize=7)

    for j in range(len(configs), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"ROC curves with selected α/τ – {title_suffix}")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def _plot_roc_custom_layout(
        df: pd.DataFrame,
        thr_tbl: pd.DataFrame,
        axes_map: Dict[str, Dict[str, str]],
        zoom: tuple = (0.06, 0.9),
        out_path: Path | None = None,
):
    """Plot ROC curves in a *fixed* 2×4 grid as specified in `axes_map`."""
    thr_tbl['sensitivity'] = thr_tbl['sensitivity'].apply(lambda x: x*100).round(2).astype(str)
    thr_tbl['specificity'] = thr_tbl['specificity'].apply(lambda x: x*100).round(2).astype(str)

    thr_tbl['sensitivity'] = thr_tbl['sensitivity'].astype(str).replace('100.0', '100')
    thr_tbl['specificity'] = thr_tbl['specificity'].astype(str).replace('100.0', '100')


    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 11))
    palette = sns.color_palette("tab10", len(df["model_name"].unique()))
    model_colors = {m: palette[i] for i, m in enumerate(sorted(df["model_name"].unique()))}

    # helper to fetch tau/alpha/sens/spec safely
    def _row_vals(row, col_stub):
        for col in row.filter(like=col_stub).columns:
            if not pd.isna(row[col].iat[0]):
                return row[col].iat[0]
        return np.nan

    for key, meta in axes_map.items():
        r, c = int(key[3]), int(key[4])  # ax_ij → i=row, j=col
        ax = axes[r, c]
        cfg = meta["cfg"]
        typ = meta["type"]
        tau_col = "best_threshold" # + ("_veto" if typ == "veto" else "")
        alpha_col = "best_alpha" # + ("_veto" if typ == "veto" else "")

        for model in sorted(df["model_name"].unique()):
            sub = df[(df["config"] == cfg) & (df["model_name"] == model)]
            if sub.empty:
                continue
            y_true, y_prob = sub["true_label"], sub["predicted_prob"]
            fpr, tpr, thr = roc_curve(y_true, y_prob)
            ax.plot(fpr, tpr, color=model_colors[model], lw=1.3) #, label=model)

            row_thr = thr_tbl[(thr_tbl["config"] == cfg) &
                              (thr_tbl["model_name"] == model) &
                              (thr_tbl["type"] == typ)]
            if not row_thr.empty:
                tau = row_thr[tau_col].iat[0]
                alpha = row_thr[alpha_col].iat[0]
                sens = _row_vals(row_thr, "sensitivity")
                spec = _row_vals(row_thr, "specificity")
                idx = np.argmin(np.abs(thr - tau))
                ax.scatter(fpr[idx], tpr[idx], s=70, edgecolors="k", facecolors=model_colors[model])
                ax.plot([], [], marker='o', color=model_colors[model], linestyle='',
                        label=(f"$\\bf{{{model}}}$\n  (α={alpha:.3f}, τ={tau:.3f}, "
                               f"Sen={sens}, Spe={spec})"))

        ax.set_xlim([0, zoom[0]])
        ax.set_ylim([zoom[1], 1.02])
        if typ == 'veto':
            ax.set_title(f"{cfg} [{typ.capitalize()}]")
        else:
            ax.set_title(f"{cfg}")
        ax.grid(True)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8)

        # turn off unused axes (e.g., bottom‑right two)
        for i in range(n_rows):
            for j in range(n_cols):
                if f"ax_{i}{j}" not in axes_map:
                    axes[i, j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle("ROC – predefined grid order", y=1.03)
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# CORE PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(input_file: Path,
                 outdir: Path,
                 prevalence:float,
                 alpha_values:np.ndarray) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    logging.info("Reading predictions from %s", input_file)
    df_all = pd.read_csv(input_file, low_memory=False)

    # Stage I – plain PFR sweep
    logging.info("Stage I: selecting plain thresholds …")
    thr_plain, thr_plain_best = collect_best_thresholds_for_models(df=df_all,
                                                   alpha_values=alpha_values,
                                                   prevalence=prevalence,
                                                   dataset_type='full',
                                                   # sensitivity_threshold=0.90,
                                                   # specificity_threshold=0.99,
                                                    collect_by='specificity',
                                                   veto=False
                                                   )

    thr_plain.to_csv(outdir / "thresholds_stage1.csv", index=False)

    # Stage II – metrics with veto using plain τ
    logging.info("Stage II: computing metrics with veto (plain τ) …")
    m_plain, val_plain = compute_metrics_across_folds_with_best_thresholds_and_veto(
        df_probs=df_all,
        best_thresholds=thr_plain,
        best_threshold_col="best_threshold",
        prevalence=PREVALENCE
    )
    m_plain.to_csv(outdir / "metrics_stage2.csv", index=False)
    val_plain.to_csv(outdir / "validation_predictions_stage2.csv", index=False)

    # Stage I′ – veto‑aware PFR sweep
    logging.info("Stage I′: selecting veto‑aware thresholds …")
    # thr_veto = collect_best_thresholds_with_veto(df=df_all,
    #                                              alpha_values=alpha_values,
    #                                              prevalence=prevalence,
    #                                              dataset_type='full',
    #                                              sensitivity_threshold=0.90,
    #                                              specificity_threshold=0.99
    #                                              )

    # thr_veto = collect_best_thresholds_for_models(df=df_all,
    #                                                alpha_values=alpha_values,
    #                                                prevalence=prevalence,
    #                                                dataset_type='full',
    #                                                sensitivity_threshold=0.90,
    #                                                specificity_threshold=0.99,
    #                                                veto=True
    #                                                )
    thr_veto, thr_veto_best = collect_best_thresholds_with_veto_long(df=df_all,
                                           alpha_values=alpha_values,
                                           prevalence=prevalence,
                                           dataset_type='full',
                                           # sensitivity_threshold=0.88,
                                           # specificity_threshold=0.99,
                                            apply_veto=True
                                           )

    thr_veto.to_csv(outdir / "thresholds_stage1_veto.csv", index=False)
    thr_veto_best.to_csv(outdir / "thresholds_stage1_veto_best.csv", index=False)

    thr_plain['type'] = 'plain'
    thr_veto_copy = thr_veto_best.copy()
    thr_veto_copy.columns = [col.replace('_veto', '') for col in thr_veto_copy.columns ]
    thr_veto_copy['type'] = 'veto'
    thr = pd.concat([thr_plain, thr_veto_copy], axis=0)

    # thr = pd.merge(left=thr_plain,
    #                right=thr_veto,
    #                on=['config', 'model_name'])
    thr = thr[['config', 'model_name'] + sorted([col for col in thr.columns if col not in ['config', 'model_name']])]
    col_numerical = [col for col in thr.columns if not col in ['config', 'model_name'] ]
    thr[col_numerical] = thr[col_numerical].round(5)
    # remove the feature set that the veto has no effect
    mask_hla = thr["config"].str.contains("DQB1*06:02", regex=False)  # HLA feature present
    mask_veto = thr["type"].eq("veto")  # veto rows only
    thr = thr.loc[~(mask_veto & mask_hla)].copy()


    # Stage II′ – metrics with veto using veto‑aware τ
    logging.info("Stage II′: computing metrics with veto (veto‑aware τ) …")
    m_veto, val_veto = compute_metrics_across_folds_with_best_thresholds_and_veto(
        df_probs=df_all,
        best_thresholds=thr_veto.rename(columns={"best_threshold_veto": "best_threshold"}),
        best_threshold_col="best_threshold",
        prevalence=PREVALENCE,
        dataset_type='validation'
    )
    m_veto.to_csv(outdir / "metrics_stage2_veto_selected.csv", index=False)
    val_veto.to_csv(outdir / "validation_predictions_stage2b.csv", index=False)

    # ---------------- PLOTS --------------------------------------
    logging.info("Plotting ROC curves for Stage I (plain) …")

    # _plot_roc_with_selected_thresholds(
    #     df=df_all,
    #     thr_tbl=thr_plain,
    #     title_suffix="Stage I plain",
    #     # alpha_values=ALPHA_VALUES,
    #     zoom=(0.06, 0.86),
    #     out_path=outdir / "roc_stage1_plain.png",
    # )
    #
    # logging.info("Plotting ROC curves for Stage I′ (veto‑aware) …")
    # _plot_roc_with_selected_thresholds(
    #     df=df_all,
    #     thr_tbl=thr_veto,
    #     title_suffix="Stage I′ veto‑aware",
    #     zoom=(0.06, 0.86),
    #     out_path=outdir / "roc_stage1_veto.png",
    # )
    #
    # logging.info("Plotting ROC curves for Stage I′ (plain and veto‑aware) …")
    # _plot_roc_with_selected_thresholds(
    #     df=df_all,
    #     thr_tbl=thr,
    #     title_suffix="Stage I′ veto‑aware",
    #     zoom=(0.06, 0.86),
    #     out_path=outdir / "roc_stage1_plain_and_veto.png",
    # )

    axes_map = {
        'ax_00': {
            'cfg': 'Full Feature Set + DQB1*06:02  (k=27)',
            'type': 'plain',
        },
        'ax_10': {
            'cfg': 'Full feature set (k=26)',
            'type': 'plain',
        },
        'ax_01': {
            'cfg': 'Reduced Feature Set + DQB1*06:02 (k=11)',
            'type': 'plain',
        },
        'ax_11': {
            'cfg': 'Reduced Feature Set (k=10)',
            'type': 'plain',
        },
        'ax_02': {
            'cfg': 'Full feature set (k=26)',
            'type': 'veto',
        },
        'ax_12': {
            'cfg': 'Reduced Feature Set (k=10)',
            'type': 'veto',
        },
    }


    _plot_roc_custom_layout( df=df_all,
        thr_tbl=thr,
        zoom=(0.06, 0.65),
        axes_map=axes_map,
        # out_path=outdir / "roc_stage1_plain_and_veto.png",)
                             )

    logging.info("✅ Pipeline finished – outputs in %s", outdir.resolve())



if __name__ == "__main__":
    TEST = True
    test_flag = 'test_' if TEST else ''
    base_path = config.get('results_path').get('results')
    PREVALENCE= 30 / 100000
    alpha_values = np.linspace(0.001, 0.05, 40)
    outdir = base_path.joinpath('stage_one_with_veto')
    run_pipeline(input_file=base_path.joinpath(f'{test_flag}pred_prob_all_models.csv'),
                    outdir=outdir,
                 prevalence=PREVALENCE,
                 alpha_values=alpha_values
                 )



