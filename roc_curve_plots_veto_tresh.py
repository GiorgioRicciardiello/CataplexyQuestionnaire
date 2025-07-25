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
            spec_ref=spec_ref,  # pick the row whose specificity is closest to 0.99
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




def collect_best_thresholds_with_veto_long(
        df: pd.DataFrame,
        alpha_values: np.ndarray,
        prevalence: float,
        dataset_type: str = "full",
        apply_veto: bool = False,
        collect_by: str = 'specificity',
        spec_ref: float = 0.99,
        sens_min:float = 0.80,
        include_hla_feature_set:bool = False,
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
    if not collect_by in ['specificity', 'youden']:
        raise ValueError(f'collect_by must be one of {["specificity", "youden"]}')

    # ------------------------------------------------------------------ #
    # 0.  subset & clean
    # ------------------------------------------------------------------ #
    if not include_hla_feature_set:
        # remove the feature sets that uses the HLA
        df_ = df.loc[
            (df["dataset_type"] == dataset_type)
            & ~df["config"].str.contains(r"DQB1\*06:0", regex=True)
            ].copy()
    else:
        df_ = df.copy()

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

            m = _compute_metrics(y_pred=y_prob,
                                 y_true=y_true,
                                 prevalence=prevalence,
                                 threshold=tau_star)
            youden_j = m["sensitivity"] + m["specificity"] - 1

            recs.append(
                {
                    "config": cfg,
                    "model_name": model,
                    "alpha": float(alpha),
                    "threshold": float(tau_star),
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "youden_j": youden_j,
                    "auc": m["auc"],
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

    if collect_by == 'specificity':
        # def _pick_closest(
        #         df: pd.DataFrame,
        #         sens_min: float,  # minimum acceptable sensitivity
        #         spec_ref: float,  # target (reference) specificity
        # ) -> pd.DataFrame:
        #     """
        #     For every (config, model_name) keep **one** row:
        #
        #     1.  Filter rows with sensitivity ≥ sens_min
        #         (rows below this cut-off are discarded).
        #
        #     2.  Among the remaining rows, compute the absolute distance
        #         |specificity − spec_ref|.
        #
        #     3.  Pick the row with the **smallest distance** for each
        #         (config, model_name).  If there is a tie, keep the row with the
        #         **highest sensitivity**, then the highest specificity.
        #
        #     Parameters
        #     ----------
        #     df : pd.DataFrame
        #         Must contain columns ['config', 'model_name',
        #         'specificity', 'sensitivity'].
        #     sens_min : float
        #         Minimum sensitivity required for a row to be eligible.
        #     spec_ref : float
        #         Desired (reference) specificity.
        #
        #     Returns
        #     -------
        #     pd.DataFrame
        #         One “best” row per (config, model_name).
        #     """
        #     if df.empty:
        #         return df.copy()
        #
        #     # -------------------------------------------------------------
        #     # 1. Keep only rows with sensitivity ≥ sens_min
        #     # -------------------------------------------------------------
        #     df_eligible = df[df["sensitivity"] >= sens_min].copy()
        #     if df_eligible.empty:  # nothing meets the sensitivity cut-off
        #         return df_eligible  # returns empty DF with same columns
        #
        #     # -------------------------------------------------------------
        #     # 2. Distance to reference specificity
        #     # -------------------------------------------------------------
        #     df_eligible["dist"] = (df_eligible["specificity"] - spec_ref).abs()
        #
        #     # -------------------------------------------------------------
        #     # 3. Select the closest row per (config, model_name)
        #     #    Tie-break: higher sensitivity, then higher specificity
        #     # -------------------------------------------------------------
        #     df_best = (
        #         df_eligible
        #         .sort_values(
        #             ["config", "model_name", "dist",
        #              "sensitivity", "specificity"],
        #             ascending=[True, True, True, False, False],
        #         )
        #         .groupby(["config", "model_name"], as_index=False)
        #         .first()  # keep the top-ranked row
        #         .drop(columns="dist")
        #     )
        #
        #     return df_best

        # df_best = _pick_closest(
        #     df_records,
        #     sens_min=0.80,
        #     spec_ref=spec_ref
        # )
        #
        # df_best = _pick_closest_with_fallback(
        #     df=df_records,
        #     sens_min=sens_min,
        #     spec_ref_start=spec_ref,
        #     spec_ref_min=0.85,  # the lowest specificity
        #     spec_step=0.01  # Steps down to find something that works
        # )

        def _pick_highest_spec_with_sens_min(
                df: pd.DataFrame,
                sens_min: float = 0.80,
        ) -> pd.DataFrame:
            """
            For each (config, model_name) group, selects the row with the highest specificity
            among those with sensitivity greater than or equal to `sens_min`.

            If no row in the group meets the minimum sensitivity threshold, the fallback is the
            row with the highest sensitivity.

            Parameters:
                df (pd.DataFrame): Input DataFrame containing evaluation metrics.
                sens_min (float): Minimum acceptable sensitivity.

            Returns:
                pd.DataFrame: A DataFrame with one selected row per (config, model_name) group.
            """
            df_best_list = []

            for (cfg, model), group in df.groupby(["config", "model_name"]):
                # Filter to rows with acceptable sensitivity
                group_sens_ok = group[group['sensitivity'] >= sens_min]

                if not group_sens_ok.empty:
                    # From those, pick the row with the highest specificity
                    best_row = group_sens_ok.loc[[group_sens_ok['specificity'].idxmax()]]
                else:
                    # Fallback: pick row with highest sensitivity
                    best_row = group.loc[[group['sensitivity'].idxmax()]]

                df_best_list.append(best_row)

            return pd.concat(df_best_list, ignore_index=True)

        df_best = _pick_highest_spec_with_sens_min(
            df=df_records,
            sens_min=sens_min,
        )
    else:
        df_best = (
            df_records
            .sort_values(
                ["config", "model_name",
                 "youden_j",  # ① highest J
                 "specificity",  # ② highest specificity
                 "sensitivity"],  # ③ highest sensitivity
                ascending=[True, True, False, False, False],
            )
            .groupby(["config", "model_name"], as_index=False)
            .first()  # keep top-ranked row
            .drop(columns="youden_j")
        )

    if apply_veto:
        rename_dict = {'sensitivity': 'sensitivity_veto',
                                   'specificity': 'specificity_veto',
                                   'threshold': 'threshold_veto',
                                   'alpha': 'alpha_veto'}
        df_records.rename(columns=rename_dict)
        df_best.rename(columns=rename_dict)
    return df_records, df_best


def _pick_closest_with_fallback(
        df: pd.DataFrame,
        spec_ref_start: float=0.99,
        sens_min: float = 0.80,
        spec_ref_min: float = 0.95,
        spec_step: float = 0.01,
) -> pd.DataFrame:
    """
    Selects the best-performing threshold for each (config, model_name) group,
    prioritizing high specificity while ensuring sensitivity meets a minimum threshold.

    The function iteratively lowers the specificity target (starting from `spec_ref_start`)
    until it finds a group of candidates meeting both the current specificity threshold
    and the minimum required sensitivity (`sens_min`). From that group, it selects
    the first row (after sorting by specificity and sensitivity) that satisfies the sensitivity constraint.

    If no row meets the sensitivity constraint at any specificity level ≥ `spec_ref_min`,
    it falls back to the row with the highest sensitivity among the most specific candidates,
    or, if the group is empty, to the row with the highest specificity overall.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing model evaluation metrics.
        spec_ref_start (float): Starting (highest) specificity threshold for search.
        sens_min (float): Minimum required sensitivity.
        spec_ref_min (float): Minimum specificity value allowed during search.
        spec_step (float): Step size to decrement specificity threshold during search.

    Returns:
        pd.DataFrame: A DataFrame containing the best threshold row per (config, model_name) group.
    """
    df_best_list = []
    # start from the highest specificity value to the lowest acceptable
    spec_array = np.arange(spec_ref_start, spec_ref_min, -spec_step)
    group_val_spec = pd.DataFrame()

    for (cfg, model), group in df.groupby(["config", "model_name"]):
        for spec_vals in spec_array:
            group_val_spec = group.loc[group['specificity'] >= spec_vals]
            if not group_val_spec.empty:

                if group_val_spec['sensitivity'].max() < sens_min:
                    # we need to get a higher sensitivity, otherwise it will take 1 in Spec
                    continue
                else:
                    break
        if group_val_spec.empty:
            # if it did not find, then tale the one with the highest specificity
            group_val_spec = group.loc[group['specificity'].idxmax()]

        group_val_spec = group_val_spec.sort_values(by=['specificity', 'sensitivity'],
                                   ascending=[True, True])
        try:
            group_highest_spec_sens = group_val_spec.loc[group_val_spec['sensitivity'] >= sens_min].iloc[0].to_frame().T
        except IndexError:
            group_highest_spec_sens = group.loc[[group_val_spec['sensitivity'].idxmax()]]

        # from the group of best specificity, take the best sensitivity
        # group_highest_spec_sens = group.loc[group_val_spec['sensitivity'].idxmax(), :].to_frame().T

        df_best_list.append(group_highest_spec_sens)

    df_best = pd.concat(df_best_list, ignore_index=True)

    return df_best

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


# def _plot_roc_custom_layout(
#         df: pd.DataFrame,
#         thr_tbl: pd.DataFrame,
#         axes_map: Dict[str, Dict[str, str]],
#         dataset_type: str = 'full',
#         zoom: tuple = (0.06, 0.9),
#         out_path: Path | None = None,
# ):
#     """Plot ROC curves in a *fixed* 2×4 grid as specified in `axes_map`."""
#     for col in ['sensitivity', 'specificity']:
#         thr_tbl[col] = (
#             (thr_tbl[col] * 100)
#             .round(2)
#             .astype(str)
#             .str.replace('100.0', '100', regex=False)
#         )
#
#     n_rows, n_cols = 2, 3
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 11))
#     palette = sns.color_palette("tab10", len(df["model_name"].unique()))
#     model_colors = {m: palette[i] for i, m in enumerate(sorted(df["model_name"].unique()))}
#
#     # helper to fetch tau/alpha/sens/spec safely
#     def _row_vals(row, col_stub):
#         for col in row.filter(like=col_stub).columns:
#             if not pd.isna(row[col].iat[0]):
#                 return row[col].iat[0]
#         return np.nan
#
#     for key, meta in axes_map.items():
#         r, c = int(key[3]), int(key[4])  # ax_ij → i=row, j=col
#         ax = axes[r, c]
#         cfg = meta["cfg"]
#         typ = meta["type"]
#         tau_col = "threshold" # + ("_veto" if typ == "veto" else "")
#         alpha_col = "alpha" # + ("_veto" if typ == "veto" else "")
#
#         for model in sorted(df["model_name"].unique()):
#             sub = df[(df["config"] == cfg) &
#                      (df["model_name"] == model) &
#                     (df["dataset_type"] == dataset_type)]
#             if sub.empty:
#                 continue
#             y_true, y_prob = sub["true_label"], sub["predicted_prob"]
#             fpr, tpr, thr = roc_curve(y_true, y_prob)
#             ax.plot(fpr, tpr, color=model_colors[model], lw=1.3) #, label=model)
#
#             row_thr = thr_tbl[(thr_tbl["config"] == cfg) &
#                               (thr_tbl["model_name"] == model) &
#                               (thr_tbl["type"] == typ)]
#             if not row_thr.empty:
#                 tau = row_thr[tau_col].iat[0]
#                 alpha = row_thr[alpha_col].iat[0]
#                 sens = _row_vals(row_thr, "sensitivity")
#                 spec = _row_vals(row_thr, "specificity")
#                 idx = np.argmin(np.abs(thr - tau))
#                 ax.scatter(fpr[idx], tpr[idx], s=70, edgecolors="k", facecolors=model_colors[model])
#                 ax.plot([], [], marker='o', color=model_colors[model], linestyle='',
#                         label=(f"$\\bf{{{model}}}$\n  (α={alpha:.3f}, τ={tau:.3f}, "
#                                f"Sen={sens}, Spe={spec})"))
#
#         ax.set_xlim([0, zoom[0]])
#         ax.set_ylim([zoom[1], 1.02])
#         if typ == 'veto':
#             ax.set_title(f"{cfg} [{typ.capitalize()}]")
#         else:
#             ax.set_title(f"{cfg}")
#         ax.grid(True)
#         ax.set_xlabel("FPR")
#         ax.set_ylabel("TPR")
#         ax.legend(fontsize=8)
#
#         # turn off unused axes (e.g., bottom‑right two)
#         for i in range(n_rows):
#             for j in range(n_cols):
#                 if f"ax_{i}{j}" not in axes_map:
#                     axes[i, j].axis("off")
#
#     plt.tight_layout(rect=[0, 0, 1, 0.97])
#     plt.suptitle("ROC – predefined grid order", y=1.03)
#     if out_path:
#         plt.savefig(out_path, dpi=300, bbox_inches='tight')
#     plt.show()
#


def _plot_roc_custom_layout(
    df: pd.DataFrame,
    thr_tbl: pd.DataFrame,
    axes_map: Dict[str, Dict[str, str]],
    dataset_type: str = 'full',
    zoom: tuple = (0.06, 0.9),
    out_path: Path | None = None,
):
    """
    Plot ROC curves in a fixed 2×3 grid using model-threshold metadata from `thr_tbl`.

    Each subplot corresponds to a (config, type) pair defined in `axes_map`.
    For each model within a subplot, the ROC curve is plotted alongside a
    threshold-specific scatter point (with sensitivity and specificity shown in the legend).
    """
    # Format sensitivity/specificity values in the threshold table
    for col in ['sensitivity', 'specificity', 'auc' ]:
        thr_tbl[col] = (
            (thr_tbl[col] * 100)
            .round(2)
            .astype(str)
            .str.replace('100.0', '100', regex=False)
        )

    # Setup subplot grid
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 11))

    # Ensure consistent model order and color assignment across all plots
    assert len(set(df.model_name.unique()) ^ set(thr_tbl.model_name.unique()))== 0

    model_order = sorted(df["model_name"].unique())

    palette = sns.color_palette("tab10", len(model_order)) # Set2
    model_colors = {model: palette[i] for i, model in enumerate(model_order)}

    # Helper to extract the first available sensitivity/specificity value
    def _row_vals(row: pd.DataFrame, col_stub: str):
        for col in row.filter(like=col_stub).columns:
            if not pd.isna(row[col].iat[0]):
                return row[col].iat[0]
        return np.nan

    # Plot each axis based on the predefined layout
    for key, meta in axes_map.items():
        # key = 'ax_00'
        # meta = axes_map.get(key)
        r, c = int(key[3]), int(key[4])  # ax_ij → i=row, j=col
        ax = axes[r, c]
        cfg = meta["cfg"]
        typ = meta["type"]
        title = meta.get("title", cfg)
        tau_col = "threshold"  # Threshold column name
        alpha_col = "alpha"    # Alpha (FPR limit) column name


        for model in model_order:
            # model = model_order[0]
            # Subset predictions for this config × model × dataset
            sub = df[
                (df["config"] == cfg) &
                (df["model_name"] == model) &
                (df["dataset_type"] == dataset_type)
            ]

            if sub.empty:
                print(f'empty sub:\n'
                      f'cfg{cfg}\n'
                      f'model{model}\n'
                      f'dataset_type{dataset_type}\n')
                continue

            y_true, y_prob = sub["true_label"], sub["predicted_prob"]
            fpr, tpr, thr = roc_curve(y_true, y_prob)

            # Plot ROC curve
            # if model == 'LogReg (ESS Only)':
            #     ax.plot(fpr,
            #             tpr,
            #             alpha=0.7,
            #             color=model_colors[model],
            #             label=model,
            #             lw=5)
            # else:
            ax.plot(fpr,
                    tpr,
                    alpha=0.7,
                    color=model_colors[model],
                    # label=model,
                    lw=1.3)

            # # Get threshold table row for this model/config/type
            row_thr = thr_tbl[
                (thr_tbl["config"] == cfg) &
                (thr_tbl["model_name"] == model) &
                (thr_tbl["type"] == typ)
            ]

            if not row_thr.empty and not sub.empty:
                # Get alpha, threshold, sensitivity, specificity
                tau = row_thr[tau_col].iat[0]
                alpha = row_thr[alpha_col].iat[0]
                sens = _row_vals(row_thr, "sensitivity")
                spec = _row_vals(row_thr, "specificity")
                auc = _row_vals(row_thr, "auc")
                # Find the closest index on the ROC curve to tau
                idx = np.argmin(np.abs(thr - tau))

            if dataset_type == 'validation':
                m = _compute_metrics(y_pred=y_prob,
                                     y_true=y_true,
                                     threshold=tau)
                sens = round(m["sensitivity"] * 100, 3)
                spec = round(m["specificity"] * 100, 3)
                auc = round(m["auc"] * 100, 3)

            # Add scatter point at (FPR, TPR) corresponding to selected τ
            ax.scatter(fpr[idx], tpr[idx], s=70,
                       edgecolors="k", facecolors=model_colors[model])

            # Add custom legend entry
            ax.plot([], [], marker='o', linestyle='', color=model_colors[model],
                    label=(f"$\\bf{{{model}}}$\n"
                           f"(α:{alpha:.3f}, τ:{tau:.3f}, "
                           f"Sen:{sens}, Spe:{spec},"
                           f"AUC:{auc}")
                    )

        # Axis formatting
        ax.set_xlim([0, zoom[0]])
        ax.set_ylim([zoom[1], 1.02])
        ax.grid(True)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")

        if typ == 'plain':
            ax.set_title(f"{title} [{typ.capitalize()}]")
        else:
            ax.set_title(f"{title}")

        ax.legend(fontsize=8)

    # Turn off any unused axes (e.g., bottom-right if only 5 plots)
    for i in range(n_rows):
        for j in range(n_cols):
            if f"ax_{i}{j}" not in axes_map:
                axes[i, j].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle(f"{dataset_type.capitalize()}", y=0.99)
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()

def merge_two_best(df_plain_best:pd.DataFrame,
                   df_veto_best:pd.DataFrame,) -> pd.DataFrame:
    df_plain_best['type'] = 'plain'
    df_veto_best.columns = [col.replace('_veto', '') for col in df_veto_best.columns]
    df_veto_best['type'] = 'veto'
    # thr = pd.concat([df_plain_best, df_veto_best], axis=0)
    return pd.concat([df_plain_best, df_veto_best], axis=0)

def insert_best_threshold_per_config_and_model(
        thr_lbls: pd.DataFrame,
        df_preds: pd.DataFrame
) -> pd.DataFrame:
    df = df_preds.copy()
    df["best_threshold"] = np.nan

    for (cfg, model), g in thr_lbls.groupby(["config", "model_name"]):
        if "threshold" in g.columns:
            threshold = g["threshold"].values[0]
        elif "best_threshold" in g.columns:
            threshold = g["best_threshold"].values[0]
        else:
            continue  # skip if no threshold column

        df.loc[
            (df["config"] == cfg) & (df["model_name"] == model),
            "best_threshold"
        ] = threshold

    return df
# ---------------------------------------------------------------------------
# CORE PIPELINE
# ---------------------------------------------------------------------------

def run_pipeline(input_file: Path,
                 outdir: Path,
                 prevalence:float,
                 alpha_values:np.ndarray) -> None:
    """

    :param input_file:
    :param outdir:
    :param prevalence:
    :param alpha_values:
    :return:
    """
    outdir.mkdir(parents=True, exist_ok=True)
    logging.info("Reading predictions from %s", input_file)

    df_all = pd.read_csv(input_file, low_memory=False)

    # Stage I – plain PFR sweep
    logging.info("Stage I: selecting plain thresholds …")
    # thr_plain, thr_plain_best = collect_best_thresholds_for_models(df=df_all,
    #                                                alpha_values=alpha_values,
    #                                                prevalence=prevalence,
    #                                                dataset_type='full',
    #                                                # sensitivity_threshold=0.90,
    #                                                # specificity_threshold=0.99,
    #                                                 collect_by='specificity',
    #                                                veto=False
    #                                                )

    thr_plain, thr_plain_best = collect_best_thresholds_with_veto_long(df=df_all,
                                                                     alpha_values=alpha_values,
                                                                     prevalence=prevalence,
                                                                     dataset_type='full',
                                                                     apply_veto=False,
                                                                     collect_by='specificity',
                                                                    sens_min=0.80,
                                                                    include_hla_feature_set=True
                                                                     )

    thr_plain.to_csv(outdir / "thresholds_stage1.csv", index=False)
    thr_plain_best.to_csv(outdir / "thresholds_stage1_best.csv", index=False)

    # Stage I′ – veto‑aware PFR sweep
    logging.info("Stage I′: selecting veto‑aware thresholds …")
    thr_veto, thr_veto_best = collect_best_thresholds_with_veto_long(df=df_all,
                                                                     alpha_values=alpha_values,
                                                                     prevalence=prevalence,
                                                                     dataset_type='full',
                                                                     apply_veto=True,
                                                                     collect_by = 'specificity',
                                                                     sens_min=.80,
                                                                     include_hla_feature_set=False
                                                                     )


    thr_veto.to_csv(outdir / "thresholds_stage1_veto.csv", index=False)
    thr_veto_best.to_csv(outdir / "thresholds_stage1_veto_best.csv", index=False)

    # thr_plain_best = thr_plain.groupby(['config', 'model_name'], group_keys=False).nth(1)
    # thr_plain_best['type'] = 'plain'
    #
    # thr_veto_best = thr_veto.groupby(['config', 'model_name'], group_keys=False).nth(1)
    # thr_veto_best['type'] = 'veto'
    # thr = pd.concat([thr_plain_best, thr_veto_best], axis=0)
    #

    thr = merge_two_best(df_plain_best=thr_plain_best,
                         df_veto_best=thr_veto_best )

    for col_round in ['sensitivity', 'specificity', 'auc']:
        if col_round in thr.columns:
            thr[col_round] = pd.to_numeric(thr[col_round], errors='coerce').round(4)


    # thr_lbls = thr.copy()
    # df_preds = df_all.copy()

    # df_preds_thr = insert_best_threshold_per_config_and_model(
    #     thr_lbls=thr_lbls,
    #     df_preds=df_preds
    # )


    # Stage II – metrics with veto using plain τ
    logging.info("Stage II: computing metrics with veto (plain τ) …")
    m_plain, val_plain = compute_metrics_across_folds_with_best_thresholds_and_veto(
        df_probs=df_all,
        best_thresholds=thr_plain_best,
        best_threshold_col="threshold",
        prevalence=PREVALENCE
    )
    m_plain.to_csv(outdir / "metrics_stage2.csv", index=False)
    val_plain.to_csv(outdir / "validation_predictions_stage2.csv", index=False)


    # Stage II′ – metrics with veto using veto‑aware τ
    logging.info("Stage II′: computing metrics with veto (veto‑aware τ) …")
    m_veto, val_veto = compute_metrics_across_folds_with_best_thresholds_and_veto(
        df_probs=df_all,
        best_thresholds=thr_veto_best.rename(columns={"threshold_veto": "threshold"}),
        best_threshold_col="threshold",
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
            'cfg': 'Full feature set (k=26)',
            'type': 'plain',
            'title': r'$\bf(a1)$ Full feature set (k=26)'
        },
        'ax_10': {
            'cfg': 'Full Feature Set + DQB1*06:02  (k=27)',
            'type': 'plain',
            'title': r'$\bf(a2)$ Full Feature Set + DQB1*06:02  (k=27)'
        },
        'ax_01': {
            'cfg': 'Reduced Feature Set (k=10)',
            'type': 'plain',
            'title': r'$\bf(b1)$ Reduced Feature Set (k=10)'
        },
        'ax_11': {
            'cfg': 'Reduced Feature Set + DQB1*06:02 (k=11)',
            'type': 'plain',
            'title': r'$\bf(b2)$ Reduced Feature Set + DQB1*06:02 (k=11)'
        },
        'ax_02': {
            'cfg': 'Full feature set (k=26)',
            'type': 'veto',
            'title': r'$\bf(a3)$ Full feature set (k=26) [Veto]'
        },
        'ax_12': {
            'cfg': 'Reduced Feature Set (k=10)',
            'type': 'veto',
            'title': r'$\bf(b3)$ Reduced Feature Set (k=10) [Veto]'
        },
    }
    #
    # df_all = df_all.loc[df_all['model_name'] != 'LogReg (ESS Only)', :]
    # thr = thr.loc[thr['model_name'] != 'LogReg (ESS Only)', :]

    _plot_roc_custom_layout(
        df=df_all,
        # df=df_all.loc[df_all['model_name'] == 'LogReg (ESS Only)', :],
        thr_tbl=thr.copy(),
        # thr_tbl=thr.loc[thr['model_name'] == 'LogReg (ESS Only)', :].copy(),
        zoom=(0.06, 0.65),
        dataset_type='full',
        # zoom=(0.90, 0.20),
        axes_map=axes_map,
        out_path=outdir / "roc_stage1_plain_and_veto.png",
        )

    _plot_roc_custom_layout(
        df=df_all,
        thr_tbl=thr.copy(),
        zoom=(0.06, 0.65),
        dataset_type='validation',
        # zoom=(0.20, 0.20),
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



