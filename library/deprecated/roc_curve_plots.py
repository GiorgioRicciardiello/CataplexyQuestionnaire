from config.config import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import math
from pathlib import Path
import seaborn as sns
from typing import Optional, Tuple
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from scipy.stats import norm
from typing import Dict, List


def _compute_metrics(
    y_pred: np.ndarray,          # probabilities or decision scores
    y_true:  np.ndarray,
    prevalence: float | None = None,
    threshold: Optional[float] = 0.50,      # threshold for hard calls
) -> Dict[str, float]:
    """
    Compute classification metrics (no CIs) given **probability scores**
    and true labels.  `auc` is the standard ROC-AUC.
    """
    # hard predictions for confusion-matrix metrics
    if  set(np.unique(y_pred)) == {1, 0}:
        y_pred_bin = y_pred  # the predictions are already binary
        auc = float("nan")
    else:
        # we are given probabilities that we will threshold
        y_pred_bin = (y_pred >= threshold).astype(int)
        # ROC-AUC (set to nan if only one class present)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float("nan")

    # default prevalence if none provided
    if prevalence is None:
        prevalence = 30 / 100_000

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()

    sensitivity  = tp / (tp + fn) if (tp + fn) else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy     = (tp + tn) / (tp + tn + fp + fn)
    precision    = tp / (tp + fp) if (tp + fp) else 0.0
    f1_score     = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) else 0.0
    )
    npv          = tn / (tn + fn) if (tn + fn) else 0.0
    fpr          = fp / (fp + tn) if (fp + tn) else 0.0
    fnr          = fn / (fn + tp) if (fn + tp) else 0.0

    # prevalence-adjusted PPV
    denom = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    ppv   = (sensitivity * prevalence) / denom if denom else 0.0



    return {
        "sensitivity":  sensitivity,
        "specificity":  specificity,
        "accuracy":     accuracy,
        "ppv_apparent": precision,     # PPV on this fold/test
        "f1_score":     f1_score,
        "auc":          auc,           # standard ROC-AUC
        "npv":          npv,
        "fpr":          fpr,
        "fnr":          fnr,
        "ppv":          ppv,           # prevalence-adjusted PPV
    }

def _compute_confidence_interval(values: List[float], formal_txt:Optional[bool]=False) -> str:
    """
    Given a list of metric values (one per fold), compute and return a 95% CI string.
    Format: "mean\n(lower, upper)" with four decimal places.
    """
    arr = np.array(values)
    mean_val = arr.mean()
    std_err = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0
    margin = norm.ppf(0.975) * std_err
    lower = max(0.0, mean_val - margin)
    upper = min(100, mean_val + margin)

    # Helper function for formatting
    def format_value(val: float) -> str:
        if val == 0.0 or val == 100.0:
            return f"{val:.0f}"
        return f"{val:.3f}"
    if formal_txt:
        return f"{mean_val:.2f} 95% CI: ({format_value(lower)}, {format_value(upper)})"
    else:
        return f"{mean_val:.2f}\n({format_value(lower)}, {format_value(upper)})"



# %% Stage I functions

def plot_roc_curves(
        df: pd.DataFrame,
        zoom: Optional[Tuple[float, float]] = (1.02, -0.02),
        output_path: Optional[Path] = None,
        dataset_type: str = 'full',
        target_fpr: float = 0.01,
        figsize: Tuple[int, int] = (10, 8),
        fontsize_legend:int=11,
        fontsize_lbl:int=16,
) -> None:
    """
    Plot ROC curves for each unique 'model_name' in `df` for each feature set ('config'),
    circle the point corresponding to the threshold that maximizes Youden’s J (tpr - fpr),
    and mark with an 'X' the point corresponding to the threshold that enforces FPR ≤ target_fpr.
    Include both threshold values in the legend.

    Expects `df` to have columns:
      - 'config'         (feature‐set name)
      - 'model_name'     (string)
      - 'true_label'     (0/1)
      - 'predicted_prob' (float in [0, 1])
      - 'dataset_type'   (e.g., 'full', 'validation', etc.)

    Args:
        df (pd.DataFrame):
            A “long” DataFrame where each row corresponds to one sample's
            prediction for a particular model/config. Required columns:
            ['config', 'model_name', 'dataset_type', 'true_label', 'predicted_prob'].

        zoom (tuple):
            (x_max, y_min) to restrict ROC axes. E.g. zoom=(0.1, 0.9) sets
            xlim=[0, 0.1], ylim=[0.9, 1.02].

        output_path (Path | None):
            If given, save the figure as PNG to this path; otherwise, just show it.

        dataset_type (str):
            Which subset of df to plot (usually 'full').

        target_fpr (float):
            Maximum allowed FPR (e.g., 0.01 for 1%). Marks with 'X' the point where
            FPR ≤ target_fpr and TPR is maximized.

        figsize (tuple):
            Figure size in inches, e.g. (10, 8).
    """
    configs = df['config'].unique()
    n = len(configs)
    if n == 0:
        return

    # Determine grid dimensions: ncols = ceil(sqrt(n)), nrows = ceil(n / ncols)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False
    )

    for idx, feature_set in enumerate(configs):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        # Subset to this feature_set and dataset_type
        subset_fs = df[(df['config'] == feature_set) & (df['dataset_type'] == dataset_type)]
        if subset_fs.empty:
            ax.axis('off')
            continue


        for model in subset_fs['model_name'].unique():
            subset = subset_fs[subset_fs['model_name'] == model]
            y_true = subset['true_label'].to_numpy()
            y_prob = subset['predicted_prob'].to_numpy()
            N = len(y_true)
            n_cases = int(y_true.sum())
            n_controls = N - n_cases

            if np.isnan(y_prob).all():
                continue

            # Compute ROC + AUC
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            # Youden’s J = tpr – fpr → best index
            j_stats = tpr - fpr
            best_idx_j = j_stats.argmax()
            best_fpr_j = fpr[best_idx_j]
            best_tpr_j = tpr[best_idx_j]
            best_thresh_j = thresholds[best_idx_j]

            # FPR‐constrained threshold: max sensitivity with fpr ≤ target_fpr
            valid_idx = np.where(fpr <= target_fpr)[0]
            if valid_idx.size > 0:
                best_idx_fpr = valid_idx[np.argmax(tpr[valid_idx])]
            else:
                best_idx_fpr = np.argmin(np.abs(fpr - target_fpr))
            best_fpr_fpr = fpr[best_idx_fpr]
            best_tpr_fpr = tpr[best_idx_fpr]
            best_thresh_fpr = thresholds[best_idx_fpr]

            # Plot ROC line
            line, = ax.plot(
                fpr, tpr,
                lw=1.4,
                label=(
                    f"{model} (AUC={roc_auc:.3f}, "
                    f"Jthr (o)={best_thresh_j:.3f}, "
                    f"FPRthr (*)={best_thresh_fpr:.3f})"
                ),
                alpha=0.7,
            )
            color = line.get_color()

            # Circle Youden‐J optimum
            ax.scatter(
                [best_fpr_j],
                [best_tpr_j],
                s=80,
                facecolors='none',
                edgecolors=color,
                linewidths=1.3
            )
            # Mark FPR‐constrained point with 'X'
            ax.scatter(
                [best_fpr_fpr],
                [best_tpr_fpr],
                marker='*',
                c=color,
                s=80,
                linewidths=1.3
            )

        # Diagonal random‐guess line
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)

        # Axis limits & labels
        ax.set_xlim([0, zoom[0]])
        ax.set_ylim([zoom[1], 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=fontsize_lbl)
        ax.set_ylabel("True Positive Rate", fontsize=fontsize_lbl)
        ax.set_title(
            f"{feature_set}\nN={N}; NT1={n_cases}; Controls={n_controls}",
            fontsize=12
        )
        ax.legend(loc="lower right", fontsize=fontsize_legend)
        ax.grid(alpha=0.6)

    # Hide any unused subplots
    total_plots = nrows * ncols
    for empty_idx in range(n, total_plots):
        r = empty_idx // ncols
        c = empty_idx % ncols
        axes[r][c].axis('off')

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()

def compute_best_thresholds(
        df: pd.DataFrame,
        dataset_type: str = 'full',
        target_fpr: float = 0.01,  # the lower the higher the specifcity
        prevalence: float = 30 / 100000,
        apply_veto: bool = False,
) -> pd.DataFrame:
    """
    Given a “long” DataFrame with columns:
        - 'config'         (feature‐set name)
        - 'model_name'     (string)
        - 'true_label'     (0/1)
        - 'predicted_prob' (float [0, 1])
        - 'dataset_type'   (e.g., 'full', 'validation', etc.)

    Returns a DataFrame with one row per (config, model_name) computed only on
    rows where dataset_type == 'full'. The returned columns are:
        - 'config'
        - 'model_name'
        - 'auc'                  (area under the ROC curve)

        # Youden’s J optimal threshold
        - 'best_threshold_j'
        - 'sensitivity_j'
        - 'specificity_j'
        - 'ppv_j'                (apparent PPV at best_threshold_j)
        - 'ppvprev_j'            (prevalence‐adjusted PPV at best_threshold_j)

        # FPR‐constrained threshold (FPR ≤ target_fpr)
        - 'best_threshold_fpr'
        - 'sensitivity_fpr'
        - 'specificity_fpr'
        - 'ppv_fpr'              (apparent PPV at best_threshold_fpr)
        - 'ppvprev_fpr'          (prevalence‐adjusted PPV at best_threshold_fpr)

    Any group with all‐NaN predicted_prob (on the 'full' subset) will be skipped.

    After computing, the returned DataFrame is sorted within each 'config' by
    'specificity_j' in descending order.

    Args:
        df (pd.DataFrame): Must contain the required columns.
        target_fpr (float): Maximum allowed false positive rate (e.g. 0.01 for 1% FPR)
                            when selecting threshold by specificity priority.
        prevalence (float): Prevalence to use for prevalence‐adjusted PPV calculation.

    Returns:
        pd.DataFrame: One row per (config, model_name) with all the columns listed above,
                      sorted within each config by descending specificity_j.
    """
    records: List[Dict] = []

    # 1) Filter to the “full” dataset only
    df_full = df[df['dataset_type'] == dataset_type]

    # 2) Group by (config, model_name)
    grouped = df_full.groupby(['config', 'model_name'])
    for (config, model_name), subset in grouped:
        y_true = subset['true_label'].to_numpy()
        y_prob = subset['predicted_prob'].to_numpy()

        if apply_veto:
            # veto FP that are HLA negative
            # pred[(pred == 1) & (y_true == 0) & hla_neg] = 0
            hla_neg = subset["hla"].to_numpy() == 0
            y_prob[(y_true == 0) & (hla_neg)] = 0

        # Skip if all probabilities are NaN
        if pd.isna(y_prob).all():
            continue

        # 3) Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # --- Youden’s J criterion ---
        j_stats = tpr - fpr
        best_idx_j = j_stats.argmax()
        best_thresh_j = thresholds[best_idx_j]
        sens_j = tpr[best_idx_j]
        spec_j = 1.0 - fpr[best_idx_j]

        # Confusion counts at best_thresh_j
        y_pred_j = (y_prob >= best_thresh_j).astype(int)


        tn_j, fp_j, fn_j, tp_j = confusion_matrix(y_true, y_pred_j).ravel()
        ppv_j = tp_j / (tp_j + fp_j) if (tp_j + fp_j) > 0 else 0.0
        # Prevalence‐adjusted PPV at best_thresh_j
        denom_j = (sens_j * prevalence) + ((1 - spec_j) * (1 - prevalence))
        ppvprev_j = (sens_j * prevalence) / denom_j if denom_j > 0 else 0.0

        # --- Constrain FPR to ≤ target_fpr, maximize sensitivity ---
        valid_idx = np.where(fpr <= target_fpr)[0]
        if valid_idx.size > 0:
            best_idx_fpr = valid_idx[np.argmax(tpr[valid_idx])]
        else:
            # If no threshold meets fpr ≤ target_fpr, choose threshold whose FPR is closest
            best_idx_fpr = np.argmin(np.abs(fpr - target_fpr))

        best_thresh_fpr = thresholds[best_idx_fpr]
        sens_fpr = tpr[best_idx_fpr]
        spec_fpr = 1.0 - fpr[best_idx_fpr]

        # Confusion counts at best_thresh_fpr
        y_pred_fpr = (y_prob  >= best_thresh_fpr).astype(int)

        tn_fpr, fp_fpr, fn_fpr, tp_fpr = confusion_matrix(y_true, y_pred_fpr).ravel()
        ppv_fpr = tp_fpr / (tp_fpr + fp_fpr) if (tp_fpr + fp_fpr) > 0 else 0.0
        # Prevalence‐adjusted PPV at best_thresh_fpr
        denom_fpr = (sens_fpr * prevalence) + ((1 - spec_fpr) * (1 - prevalence))
        ppvprev_fpr = (sens_fpr * prevalence) / denom_fpr if denom_fpr > 0 else 0.0

        # 4) Append results for this (config, model_name)
        records.append({
            'config': config,
            'model_name': model_name,
            'auc': float(roc_auc),

            'best_threshold_fpr': float(best_thresh_fpr),  # chosen under a false‐positive‐rate constraint
            'sensitivity_fpr': float(sens_fpr),  # chosen under a false‐positive‐rate constraint
            'specificity_fpr': float(spec_fpr),  # chosen under a false‐positive‐rate constraint
            'ppv_fpr': float(ppv_fpr),  # chosen under a false‐positive‐rate constraint
            'ppvprev_fpr': float(ppvprev_fpr),  # chosen under a false‐positive‐rate constraint

            'best_threshold_j': float(best_thresh_j),  # maximizing Youden’s J
            'sensitivity_j': float(sens_j),  # maximizing Youden’s J
            'specificity_j': float(spec_j),  # maximizing Youden’s J
            'ppv_j': float(ppv_j),  # maximizing Youden’s J
            'ppvprev_j': float(ppvprev_j),  # maximizing Youden’s J

        })

    # 5) Build DataFrame and sort
    df_best = pd.DataFrame.from_records(records)

    # Sort within each config by specificity_j (descending)
    df_best = df_best.sort_values(
        by=['config', 'specificity_j'],
        ascending=[True, False]
    ).reset_index(drop=True)

    return df_best


def plot_model_logits_distribution_by_config(
    df: pd.DataFrame,
    model_name: str,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 12),
    bins: int = 30
) -> None:
    """
    For a given model_name, create a grid of subplots—one cell per feature set (config).
    Each cell contains two small subplots side-by-side:
      (a) A scatter of predicted probabilities (logits) for the 'full' dataset,
          colored by true_label (0=Control, 1=NT1), with a horizontal line at the best threshold.
      (b) A histogram showing the distribution of predicted probabilities for Control vs. NT1,
          with a vertical line at the same best threshold.

    The grid dimensions are chosen to be as square as possible based on the number of configs.

    Expects df to have columns:
      - 'config'         (feature‐set name)
      - 'model_name'     (string)
      - 'dataset_type'   (string; e.g. 'full', 'validation')
      - 'true_label'     (0 or 1)
      - 'predicted_prob' (float between 0 and 1)

    Args:
        df (pd.DataFrame):
            A “long” DataFrame where each row is one sample’s prediction for a given model/config.
            Required columns: ['config', 'model_name', 'dataset_type', 'true_label', 'predicted_prob'].

        model_name (str):
            The name of the model (e.g., 'Elastic Net') for which to plot all configs.

        output_path (Path | None):
            If provided, save the combined figure to this path (PNG). Otherwise, display it.

        figsize (tuple):
            Overall figure size in inches, e.g. (12, 12). Height will scale by the number of rows.

        bins (int):
            Number of bins for each histogram subplot.
    """
    # Filter to 'full' dataset and specified model_name
    df_model = df[
        (df['dataset_type'] == 'full') &
        (df['model_name'] == model_name)
    ].copy()

    configs = df_model['config'].unique()
    if len(configs) == 0:
        raise ValueError(f"No data for model '{model_name}' on dataset_type='full'.")

    n_configs = len(configs)
    # Compute grid size: ncols ~ sqrt(n_configs), nrows appropriately
    ncols = math.ceil(math.sqrt(n_configs))
    nrows = math.ceil(n_configs / ncols)

    fig = plt.figure(figsize=figsize)
    outer_grid = GridSpec(nrows,
                          ncols,
                          figure=fig,
                          # wspace=0.1,
                          # hspace=0.1
                          )

    for idx, config in enumerate(configs):
        # Determine row/col in the outer grid
        row = idx // ncols
        col = idx % ncols
        # Create a 1x2 sub‐grid within this outer cell
        inner_grid = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_grid[row, col], wspace=0.3, hspace=0.0
        )

        # Axes for scatter (left) and histogram (right)
        ax_scatter = fig.add_subplot(inner_grid[0, 0])
        ax_hist    = fig.add_subplot(inner_grid[0, 1])

        # Subset to this config & model on the full dataset
        df_fc = df_model[df_model['config'] == config]
        if df_fc.empty:
            ax_scatter.axis('off')
            ax_hist.axis('off')
            continue

        y_true = df_fc['true_label'].to_numpy()
        y_prob = df_fc['predicted_prob'].to_numpy()

        # Compute ROC + best threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        j_stats = tpr - fpr
        best_idx = j_stats.argmax()
        best_thresh = thresholds[best_idx]

        # --- Scatter subplot ---
        sorted_idx = np.argsort(y_prob)
        y_prob_sorted = y_prob[sorted_idx]
        y_true_sorted = y_true[sorted_idx]
        colors = np.where(y_true_sorted == 1, 'tab:orange', 'tab:blue')

        ax_scatter.scatter(
            np.arange(len(y_prob_sorted)),
            y_prob_sorted,
            c=colors,
            edgecolors='k',
            alpha=0.7
        )
        ax_scatter.axhline(
            best_thresh,
            color='red',
            linestyle='--',
            linewidth=1.5
        )
        ax_scatter.set_ylim([-0.02, 1.02])
        ax_scatter.set_xticks([])
        ax_scatter.set_ylabel("Pred. Prob.", fontsize=9)
        ax_scatter.set_title(f"{config}\n(AUC={roc_auc:.3f})", fontsize=10)

        # --- Histogram subplot ---
        prob_control = y_prob[y_true == 0]
        prob_nt1     = y_prob[y_true == 1]

        ax_hist.hist(
            prob_control,
            bins=bins,
            density=True,
            alpha=0.5,
            color='tab:blue',
            label='Control'
        )
        ax_hist.hist(
            prob_nt1,
            bins=bins,
            density=True,
            alpha=0.5,
            color='tab:orange',
            label='NT1'
        )
        ax_hist.axvline(
            best_thresh,
            color='red',
            linestyle='--',
            linewidth=1.5
        )
        ax_hist.set_xlim([-0.02, 1.02])
        ax_hist.set_xlabel("Pred. Prob.", fontsize=9)
        ax_hist.set_ylabel("Density", fontsize=9)
        ax_hist.legend(fontsize=8)

    plt.suptitle(f"{model_name} — Full Dataset Logits & Distributions by Feature Set", fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()



def plot_roc_curves_with_alpha_points_matched(
        df: pd.DataFrame,
        alpha_values: list,
        prevalence: float = 30 / 100000,
        zoom: Tuple[float, float] = (0.1, 0.9),
        dataset_type: str = 'full',
        sensitivity_threshold: float = 0.90,
        specificity_threshold: float = 0.98,
        output_path:Path=None
):
    """
    Plot the ROC curves for the different alpha levels. We visualize the best alpha for each configuration and
    model. We only plot the first alpha that achieves the desired sensitivity and specificity defined in the input
    :param df:
    :param alpha_values: list of alpha values to make plot the sensitity and specificity
    :param prevalence: pervalence in the population
    :param zoom: zoom on the roc curve
    :param dataset_type: full or validation, this is satge 1 so it should be full
    :param sensitivity_threshold:
    :param specificity_threshold:
    :param output_path:
    :return:
    """
    configs = df['config'].unique()
    model_names = df['model_name'].unique()
    n_configs = len(configs)

    # Set up color palette and model-color mapping
    palette = sns.color_palette("tab10", len(model_names))
    model_colors = {model: palette[i] for i, model in enumerate(model_names)}

    # Create subplots grid
    n_cols = 2
    n_rows = int(np.ceil(n_configs / n_cols))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(7 * n_cols, 5 * n_rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, config in enumerate(configs):
        ax = axes[i]
        df_config = df[df['config'] == config]

        for model in model_names:
            df_model = df_config[df_config['model_name'] == model]
            y_true = df_model['true_label'].to_numpy()
            y_prob = df_model['predicted_prob'].to_numpy()

            if len(y_true) == 0 or pd.isna(y_prob).all():
                continue

            fpr, tpr, thresholds = roc_curve(y_true, y_prob)

            # Try to find the first valid alpha point
            alpha_label = ""
            for alpha in alpha_values:
                df_thresholds = compute_best_thresholds(
                    df=df_model,
                    target_fpr=alpha,
                    prevalence=prevalence,
                    dataset_type=dataset_type
                )

                if df_thresholds.empty:
                    continue

                specificity = df_thresholds['specificity_fpr'].iloc[0]
                sensitivity = df_thresholds['sensitivity_fpr'].iloc[0]
                best_threshold = df_thresholds['best_threshold_fpr'].iloc[0]

                if specificity >= specificity_threshold and sensitivity >= sensitivity_threshold:
                    idx = np.argmin(np.abs(thresholds - best_threshold))
                    fpr_val = fpr[idx]
                    tpr_val = tpr[idx]

                    ax.scatter(fpr_val, tpr_val, color=model_colors[model], s=40, edgecolor='black', zorder=5)
                    alpha_label = f' \nα={alpha:.3f}, τ={best_threshold:.3f}'
                    break

            ax.plot(fpr, tpr, label=fr'$\mathbf{{{model}}}$ {alpha_label}', color=model_colors[model])

        ax.set_xlim([0, zoom[0]])
        ax.set_ylim([zoom[1], 1.02])
        ax.set_title(f'Feature Set: {config}')
        ax.set_xlabel('False Positive Rate (FPR)')
        if i % n_cols == 0:
            ax.set_ylabel('True Positive Rate (TPR)')
        ax.grid(True)
        ax.legend(fontsize=8)

    for j in range(len(configs), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("ROC Curves with α-Threshold Points", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()


def collect_best_thresholds_for_models(
        df: pd.DataFrame,
        alpha_values: list,
        prevalence: float = 30 / 100000,
        dataset_type: str = 'full',
        sensitivity_threshold: float = 0.90,
        specificity_threshold: float = 0.98
) -> pd.DataFrame:
    """
    For each (config, model_name), returns the best alpha and corresponding threshold
    that satisfies the input sensitivity and specificity requirements.
    If none meets both, the pair is skipped.

    Using the PFR criteria and not the Youden index
    Returns:
        DataFrame with columns:
        - config
        - model_name
        - best_alpha
        - best_threshold
        - sensitivity
        - specificity
    """
    records = []
    configs = df['config'].unique()
    model_names = df['model_name'].unique()

    for config in configs:
        df_config = df[df['config'] == config]

        for model in model_names:
            df_model = df_config[df_config['model_name'] == model]
            y_true = df_model['true_label'].to_numpy()
            y_prob = df_model['predicted_prob'].to_numpy()

            if len(y_true) == 0 or pd.isna(y_prob).all():
                continue

            for alpha in alpha_values:
                df_thresholds = compute_best_thresholds(
                    df=df_model,
                    target_fpr=alpha,
                    prevalence=prevalence,
                    dataset_type=dataset_type
                )

                if df_thresholds.empty:
                    continue

                specificity = df_thresholds['specificity_fpr'].iloc[0]
                sensitivity = df_thresholds['sensitivity_fpr'].iloc[0]
                best_threshold = df_thresholds['best_threshold_fpr'].iloc[0]

                if specificity >= specificity_threshold and sensitivity >= sensitivity_threshold:
                    records.append({
                        'config': config,
                        'model_name': model,
                        'best_alpha': alpha,
                        'best_threshold': best_threshold,
                        'sensitivity': sensitivity,
                        'specificity': specificity
                    })
                    break  # Stop after the first valid alpha

    return pd.DataFrame.from_records(records)


def plot_roc_curves_cv(
        df: pd.DataFrame,
        output_dir: Optional[Path] = None,
        zoom: Tuple[float, float] = (1.02, -0.02),
        figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    For each cross-validation fold in `df`, create one figure containing a grid of subplots—
    one subplot per feature set (df['config']). Within each subplot, plot ROC curves for all
    models in that feature set/fold (dataset_type='validation'), circle the “best” threshold
    (max Youden’s J), and include that threshold in the legend.

    Expects `df` with columns:
      - 'config'         (feature-set name, e.g. "questionnaire", "ukbb", etc.)
      - 'model_name'     (string)
      - 'true_label'     (0/1)
      - 'predicted_prob' (float in [0, 1])
      - 'dataset_type'   (e.g. 'validation' or 'full')
      - 'fold'           (int)

    Args:
        df (pd.DataFrame):
            “Long” DataFrame where each row is one sample’s prediction for a given
            (config, model, fold). Required columns:
                ['config', 'model_name', 'true_label', 'predicted_prob', 'dataset_type', 'fold'].

        output_dir (Path | None):
            If provided, save each fold’s figure as a PNG into this directory (named
            "roc_fold_{fold}.png"). If None, just show the figures interactively.

        zoom (tuple):
            (x_max, y_min) to restrict axis ranges. E.g. zoom=(0.1, 0.9) → xlim=[0,0.1], ylim=[0.9,1.02].

        figsize (tuple):
            Base figure size (width, height). The function will multiply height
            by the number of grid rows to keep subplots readable.
    """
    # Identify unique validation folds
    folds = sorted(df.loc[df['dataset_type'] == 'validation', 'fold_number'].dropna().unique())
    if not folds:
        return

    # Identify all feature sets
    configs = df['config'].unique()

    for fold in folds:
        # Subset to this fold and validation set
        df_fold = df[(df['dataset_type'] == 'validation') & (df['fold_number'] == fold)]
        if df_fold.empty:
            continue

        n = len(configs)
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        # Scale figure height by nrows
        width, height = figsize
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(width, height * nrows),
            squeeze=False
        )

        for idx, feature_set in enumerate(configs):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            subset_fs = df_fold[df_fold['config'] == feature_set]
            if subset_fs.empty:
                # If no data for this feature_set in this fold, hide axis
                ax.axis('off')
                continue

            for model in subset_fs['model_name'].unique():
                sub = subset_fs[subset_fs['model_name'] == model]
                y_true = sub['true_label'].to_numpy()
                y_prob = sub['predicted_prob'].to_numpy()

                if np.isnan(y_prob).all():
                    continue

                # Compute ROC + AUC
                fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)

                # Youden’s J = tpr – fpr → find best index
                j_stats = tpr - fpr
                best_idx = j_stats.argmax()
                best_fpr = fpr[best_idx]
                best_tpr = tpr[best_idx]
                best_thresh = thresholds[best_idx]

                # Interpolate for a smoother line
                fpr_smooth = np.linspace(0, 1, 200)
                tpr_smooth = np.interp(fpr_smooth, fpr, tpr)

                # Plot the smoothed ROC line
                line, = ax.plot(
                    fpr_smooth, tpr_smooth,
                    lw=1.4,
                    label=f"{model} (AUC={roc_auc:.3f}, thr={best_thresh:.3f})"
                )
                color = line.get_color()

                # Circle best-threshold point (using original fpr/tpr)
                ax.scatter(
                    [best_fpr],
                    [best_tpr],
                    s=80,
                    facecolors='none',
                    edgecolors=color,
                    linewidths=1.3
                )

            # Plot random‐guess diagonal
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)

            # Axis limits & labels
            ax.set_xlim([0, zoom[0]])
            ax.set_ylim([zoom[1], 1.02])
            ax.set_xlabel("False Positive Rate", fontsize=10)
            ax.set_ylabel("True Positive Rate", fontsize=10)
            ax.set_title(f"Fold {fold}\n{feature_set}'", fontsize=12)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.6)

        # Hide any empty subplots
        total_plots = nrows * ncols
        for empty_idx in range(len(configs), total_plots):
            r = empty_idx // ncols
            c = empty_idx % ncols
            axes[r][c].axis('off')

        plt.tight_layout()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"roc_fold_{fold}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

# %% Stage II functions
def compute_metrics_across_folds_with_best_thresholds_and_veto(
        df_probs: pd.DataFrame,
        best_thresholds: pd.DataFrame,
        prevalence=float,
        best_threshold_col:str='best_threshold_fpr',
        dataset_type:Optional[str]='validation'
) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Re-compute metrics (with 95% CIs) across validation folds for each (config, model_name),
    using the best_threshold found on the 'full' dataset. Returns a DataFrame with one row
    per (config, model_name) and columns:
        - config
        - model_name
        - sensitivity_ci
        - specificity_ci
        - accuracy_ci
        - ppv_apparent_ci
        - f1_score_ci
        - npv_ci
        - fpr_ci
        - fnr_ci
        - ppv_ci

    Args:
        df_probs: DataFrame with columns [
            'config', 'model_name', 'dataset_type', 'fold',
            'true_label', 'predicted_prob', ...
        ] including cross-validation folds.

        best_thresholds: DataFrame from compute_best_thresholds(), with columns [
            'config', 'model_name', 'best_threshold', 'auc'
        ] computed on dataset_type == 'full'.

    Returns:
        pd.DataFrame with metrics+CI per (config, model_name).
    """
    records: List[Dict] = []
    # Filter to validation rows only
    df_val = df_probs[df_probs['dataset_type'] == dataset_type].copy()
    # Merge best_threshold onto df_val so each row has its config/model's threshold
    df_merged = pd.merge(
        df_val,
        best_thresholds[['config', 'model_name', best_threshold_col]],
        on=['config', 'model_name'],
        how='left'
    )
    # Drop any rows where best_threshold is missing (e.g. no full-data threshold)
    df_merged = df_merged.dropna(subset=[best_threshold_col])

    # Convert probabilities to binary predictions using threshold
    df_merged['hla'] = df_merged['hla'].astype(int)
    df_merged['true_label'] = df_merged['true_label'].astype(int)

    # Convert probabilities to binary predictions using threshold
    df_merged['prediction_best_threshold'] = (
            df_merged['predicted_prob'] >= df_merged[best_threshold_col]
    ).astype(int)

    # apply veto rule on metrics
    df_merged['prediction_hla_veto'] = df_merged['prediction_best_threshold']
    mask_fp_hla_neg = (
            (df_merged['prediction_best_threshold'] == 1) &
            (df_merged['true_label'] == 0) &
            (df_merged['hla'] == 0)
    )
    df_merged.loc[mask_fp_hla_neg, 'prediction_hla_veto'] = 0  # correct to the true label (zero class)

    # Now group by config & model_name & fold to compute metrics per fold
    grouped_folds = df_merged.groupby(['config', 'model_name', 'fold_number'])

    # We will collect metrics per-fold in a nested dict: metrics_per_group[(config,model)][metric] = [list of values]
    metrics_per_group: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    for (config, model_name, fold), sub in grouped_folds:
        y_true = sub['true_label'].to_numpy()
        y_pred = sub['prediction_best_threshold'].to_numpy()
        y_pred_veto = sub['prediction_hla_veto'].to_numpy()

        # Compute metrics on this fold as percentages
        # print(config, model_name, fold, y_pred_veto.size)
        m = _compute_metrics(y_pred=y_pred, y_true=y_true, prevalence=prevalence)
        m = {k: float(v * 100) for k, v in m.items()}

        # veto metrics
        m_veto = _compute_metrics(y_pred=y_pred_veto, y_true=y_true, prevalence=prevalence)
        m_veto = {f"{k}_veto": float(v * 100) for k, v in m_veto.items()}

        key = (config, model_name)
        if key not in metrics_per_group:
            # initialize lists for each metric
            metrics_per_group[key] = {
                'sensitivity': [],
                'specificity': [],
                'accuracy': [],
                'ppv_apparent': [],
                'f1_score': [],
                'npv': [],
                'fpr': [],
                'fnr': [],
                'ppv': [],
                'auc': [],

                'sensitivity_veto': [],
                'specificity_veto': [],
                'accuracy_veto': [],
                'ppv_apparent_veto': [],
                'f1_score_veto': [],
                'npv_veto': [],
                'fpr_veto': [],
                'fnr_veto': [],
                'ppv_veto': [],
                'auc_veto': [],

            }

        # append this fold's metrics
        # Append regular metrics
        for metric_name, metric_val in m.items():
            metrics_per_group[key][metric_name].append(metric_val)

        # Append veto rule metrics
        for metric_name_veto, metric_val_veto in m_veto.items():
            metrics_per_group[key][metric_name_veto].append(metric_val_veto)

    # Now, for each (config, model_name), compute mean+CI for each metric
    for (config, model_name), metric_lists in metrics_per_group.items():
        record: Dict[str, str] = {
            'config': config,
            'model_name': model_name
        }
        # Compute CI for each metric
        for metric_name, values in metric_lists.items():
            ci_str = _compute_confidence_interval(values)
            record[f'{metric_name}_ci'] = ci_str

        records.append(record)

    return pd.DataFrame.from_records(records), df_merged


    # bar plot


def _parse_ci_string(ci_str: str) -> Tuple[float, float, float]:
    """
    Given a CI string of the form "mean\n(lower, upper)", parse and return
    (mean, lower, upper) as floats.
    """
    # Split on newline
    ci_str = ci_str.replace('%', '')
    mean_part, range_part = ci_str.strip().split('\n')
    mean_val = float(mean_part)
    # Remove parentheses and split on comma
    range_part = range_part.strip().strip('()')
    lower_str, upper_str = [s.strip() for s in range_part.split(',')]
    lower_val = float(lower_str)
    upper_val = float(upper_str)
    return mean_val, lower_val, upper_val

def plot_model_metrics_with_ci(
        df: "pd.DataFrame",
        metrics: List[str],
        palette: Optional[str] = 'muted',
        figsize: Optional[Tuple[int, int]] = (16, 8),
        output_path: Optional[str] = None
) -> None:
    """
    Plot specified metrics (with 95% CIs) for each model and configuration.
    Bars represent mean*100, error bars represent CI range*100.
    All annotations in a given config sit at the same height (group_max + 0.5),
    while the y-axis bottom is set via np.min(mean_min).

    df must contain:
      - 'config'       (feature-set name)
      - 'model_name'   (model name)
      - for each metric in `metrics`, a "<metric>_ci" column formatted as "mean\n(lower, upper)"

    Args:
      df (pd.DataFrame)
      metrics (List[str]): e.g. ['sensitivity', 'specificity', 'ppv', 'f1_score']
      palette (str):       seaborn palette name
      figsize (tuple):     (width, height) inches per subplot
      output_path (str|None): if provided, save to this path; otherwise show()
    """

    # 1) Check required columns
    if 'model_name' not in df.columns or 'config' not in df.columns:
        raise ValueError("DataFrame must contain 'model_name' and 'config' columns.")

    # 2) Sort for consistent ordering
    df = df.sort_values(by=['model_name', 'config']).reset_index(drop=True)

    # 3) Extract unique configs & models
    configs = sorted(df['config'].unique())
    models = sorted(df['model_name'].unique())

    # 4) Parse CI columns into a nested dict:
    #    data_dict[metric][(config, model)] = (mean*100, lower*100, upper*100)
    data_dict: Dict[str, Dict[Tuple[str, str], Tuple[float, float, float]]] = {}
    for metric in metrics:
        ci_col = f"{metric}_ci"
        if ci_col not in df.columns:
            raise ValueError(f"Column '{ci_col}' not found in DataFrame.")
        data_dict[metric] = {}
        for _, row in df.iterrows():
            cfg = row['config']
            mdl = row['model_name']
            ci_str = row[ci_col]
            if isinstance(ci_str, str) and ci_str.strip():
                mean_val, low_val, high_val = _parse_ci_string(ci_str)
                data_dict[metric][(cfg, mdl)] = (
                    mean_val, # * 100.0,
                    low_val, # * 100.0,
                    high_val, # * 100.0
                )
            else:
                data_dict[metric][(cfg, mdl)] = (0.0, 0.0, 0.0)

    # 5) Collect all lower/upper bounds to set y-axis bottom (if needed elsewhere)
    all_lowers, all_uppers = [], []
    for metric in metrics:
        for (cfg, mdl), (m_val, l_val, u_val) in data_dict[metric].items():
            all_lowers.append(l_val)
            all_uppers.append(u_val)
    global_min_lower = min(all_lowers) if all_lowers else 0.0
    # (We will compute mean_min per subplot instead for setting ylim.)

    # 6) Prepare figure with one subplot per metric
    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        nrows=n_metrics,
        ncols=1,
        figsize=(figsize[0], figsize[1] * n_metrics),
        squeeze=False
    )
    sns.set_theme(style="whitegrid", context="talk", palette=palette)

    # 7) Assign colors
    palette_colors = sns.color_palette(palette, n_colors=len(models))
    model_to_color = {models[i]: palette_colors[i] for i in range(len(models))}

    # 8) Loop over each metric
    for i, metric in enumerate(metrics):
        ax = axes[i][0]
        bar_width = 0.8 / len(models)

        # Build mean_min list so we can set the bottom of y-axis later
        mean_min = []

        # Precompute “group_max” for each config so that all annotations align
        group_max: Dict[str, float] = {}
        group_min: Dict[str, float] = {}
        for cfg in configs:
            max_for_cfg = max(
                data_dict[metric].get((cfg, mdl), (0.0, 0.0, 0.0))[0]
                for mdl in models
            )
            min_for_cfg = min(
                data_dict[metric].get((cfg, mdl), (0.0, 0.0, 0.0))[0]
                for mdl in models
            )
            group_max[cfg] = max_for_cfg
            group_min[cfg] = min_for_cfg - 0.5

        # X‐axis positions
        x_indices = np.arange(len(configs))

        # Now draw each model’s bars and label them at group_max
        for j, model in enumerate(models):
            means, err_l, err_u = [], [], []
            for cfg in configs:
                m_val, l_val, u_val = data_dict[metric].get((cfg, model), (0.0, 0.0, 0.0))
                means.append(m_val)
                err_l.append(m_val - l_val)
                err_u.append(u_val - m_val)

            # Keep track of the minimum mean across all bars (for y-axis bottom)
            mean_min.append(np.min(means))

            # Compute bar positions
            offsets = (j - (len(models) - 1) / 2) * bar_width
            bar_positions = x_indices + offsets

            # Draw bars
            ax.bar(
                bar_positions,
                means,
                width=bar_width,
                color=model_to_color[model],
                edgecolor='black',
                label=model
            )
            # Draw error bars
            ax.errorbar(
                bar_positions,
                means,
                yerr=np.vstack([err_l, err_u]),
                fmt='none',
                ecolor='black',
                capsize=5,
                linewidth=1.2
            )

            # Annotate each bar at the group_max height (so all labels in this config align)
            for idx_cfg, xpos in enumerate(bar_positions):
                cfg_name = configs[idx_cfg]
                y_annot = group_max[cfg_name] + 0.5
                val = means[idx_cfg]
                label = f"{val:.0f}" if val in [0.0, 100.0] else f"{val:.2f}"
                ax.text(
                    xpos,
                    y_annot,
                    label,
                    ha='center',
                    va='bottom',
                    fontsize=12,
                    fontweight='bold',
                    color='black',
                    bbox={
                        'facecolor': 'white',
                        'edgecolor': 'none',
                        'pad': 2.6
                    }
                )

        # 9) Now set y-axis limits exactly as requested
        ax.set_xticks(x_indices)
        ax.set_xticklabels(configs, rotation=0, ha='center', fontsize=20)
        ax.tick_params(axis='y', labelsize=16)  # You can change 16 to your preferred font size
        ax.set_ylabel(f"{metric.replace('_', ' ').upper()} (%)", fontsize=16)
        # Bottom of y-axis = min(mean_min); Top = 100.5
        ax.set_ylim(np.min(mean_min)-4, 100)
        # ax.set_ylim(np.min(group_min[cfg_name]), 100)
        # ax.set_title(f"{metric.replace('_', ' ').upper()}", fontsize=24)
        ax.grid(axis='y', alpha=0.7)

    # 10) Shared legend at top
    handles = [Line2D([0], [0], color=model_to_color[m], lw=4) for m in models]
    fig.legend(
        handles,
        models,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(models),
        frameon=False,
        fontsize=20
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

# %% confusion matrix
def compute_confusion_counts(df: pd.DataFrame, pred_col: str, suffix: str) -> pd.DataFrame:
    """
    Compute TP, FP, TN, and FN counts grouped by config, model_name, dataset_type, and fold_number.

    Parameters:
        df: DataFrame with 'true_label' and prediction column
        pred_col: column name with binary predictions
        suffix: string to append to the metric column names
    Returns:
        DataFrame with TP, FP, TN, FN counts
    """
    group_cols = ['config', 'model_name', 'dataset_type', 'fold_number']

    # Define masks
    mask_tp = (df['true_label'] == 1) & (df[pred_col] == 1)
    mask_fp = (df['true_label'] == 0) & (df[pred_col] == 1)
    mask_tn = (df['true_label'] == 0) & (df[pred_col] == 0)
    mask_fn = (df['true_label'] == 1) & (df[pred_col] == 0)

    # Group and count each class
    df_tp = df[mask_tp].groupby(group_cols).size().reset_index(name=f'tp_{suffix}')
    df_fp = df[mask_fp].groupby(group_cols).size().reset_index(name=f'fp_{suffix}')
    df_tn = df[mask_tn].groupby(group_cols).size().reset_index(name=f'tn_{suffix}')
    df_fn = df[mask_fn].groupby(group_cols).size().reset_index(name=f'fn_{suffix}')

    # Merge all counts together
    df_counts = pd.merge(df_tp, df_fp, on=group_cols, how='outer')
    df_counts = pd.merge(df_counts, df_tn, on=group_cols, how='outer')
    df_counts = pd.merge(df_counts, df_fn, on=group_cols, how='outer')

    # Replace NaNs with 0 and convert to int
    count_cols = [f'tp_{suffix}', f'fp_{suffix}', f'tn_{suffix}', f'fn_{suffix}']
    df_counts[count_cols] = df_counts[count_cols].fillna(0).astype(int)

    return df_counts


def plot_confusion_matrix_grid(df: pd.DataFrame,
                               true_col: str = 'true_label',
                               pred_orig_col: str = 'prediction_best_threshold',
                               pred_veto_col: str = 'prediction_hla_veto',
                               config_col: str = 'config',
                               dataset_type: str = 'validation',
                               model_name: str = 'Elastic Net',
                               figsize: tuple = (12, 10)) -> None:
    """
    Plot a grid of confusion matrices for each config:
    Left column: original predictions
    Right column: veto-corrected predictions

    Each title includes fold-averaged PPV with 95% CI.
    """
    configs = sorted(df[config_col].unique())
    n_configs = len(configs)

    fig, axes = plt.subplots(n_configs, 2, figsize=figsize)
    fig.subplots_adjust(hspace=0.7, wspace=0.2)

    if n_configs == 1:
        axes = np.array([axes])  # ensure 2D for consistent indexing

    for i, cfg in enumerate(configs):
        df_cfg = df[(df[config_col] == cfg) &
                    (df['dataset_type'] == dataset_type) &
                    (df['model_name'] == model_name)]

        y_true = df_cfg[true_col].astype(int)
        y_pred_orig = df_cfg[pred_orig_col].astype(int)
        y_pred_veto = df_cfg[pred_veto_col].astype(int)

        for j, (y_pred, label, ax) in enumerate(zip(
                [y_pred_orig, y_pred_veto],
                ['Original', 'Veto-Corrected'],
                axes[i]
        )):
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_sum = cm.sum()
            cm_pct = cm / cm_sum * 100
            annot = np.array([[f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)])

            sns.heatmap(cm,
                        annot=annot,
                        fmt='',
                        cmap='Blues',
                        cbar=False,
                        xticklabels=['Control', 'NT1'],
                        yticklabels=['Control', 'NT1'],
                        annot_kws={'fontsize': 13},
                        ax=ax)

            # Compute fold-averaged PPV
            fold_ppvs = []
            for fold, df_fold in df_cfg.groupby('fold_number'):
                y_true_f = df_fold[true_col].astype(int)
                y_pred_f = df_fold[y_pred.name].astype(int)
                m = _compute_metrics(y_pred=y_pred_f, y_true=y_true_f, prevalence=PREVALENCE)
                fold_ppvs.append(float(m['ppv'] * 100))

            ppv_text = _compute_confidence_interval(fold_ppvs, formal_txt=True)
            cfgshort = cfg.replace(' + DQB1*06:02', '+HLA')
            ax.set_title(f"{label}\n{cfgshort}\nPPV: {ppv_text}", fontsize=12)

            # Set axis tick visibility
            if j == 0:
                ax.set_xticklabels([] if i < n_configs - 1 else ['Control', 'NT1'])
                ax.set_yticklabels(['Control', 'NT1'])  # always show y on the left
            else:
                ax.set_xticklabels(['Control', 'NT1'] if i == n_configs - 1 else [])
                ax.set_yticklabels([])  # hide y on the right

            # Set axis labels **only for top row**
            if i == 0:
                if j == 0:
                    ax.set_ylabel('True Label', fontsize=11)
                if j == 1:
                    ax.set_xlabel('Predicted Label', fontsize=11)

            # Always turn off tick marks
            ax.tick_params(axis='x', bottom=False, top=False)
            ax.tick_params(axis='y', left=False, right=False)

    fig.text(0.5, 0.04, 'Predicted Label (Control vs NT1)', ha='center', fontsize=12)
    fig.text(0.04, 0.5, 'True Label (Control vs NT1)', va='center', rotation='vertical', fontsize=12)
    fig.text(0.5, 1 - 0.04, model_name, va='center', fontsize=18)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    TEST = True
    # %% input paths 
    test_flag = 'test_' if TEST else ''
    base_path = config.get('results_path').get('results')
    # pred and prob of full dataset concat with k-folds
    path_models_pred_prob            = base_path.joinpath(f'{test_flag}pred_prob_all_models.csv')
    path_output = base_path.joinpath(f'poster_ppt')
    path_output.mkdir(parents=True, exist_ok=True)
    PREVALENCE = 30 / 100000
    # %%
    df_all_model_probs = pd.read_csv(path_models_pred_prob)
    # model_names = df_all_model_probs['model_name'].unique()
    # dataset_type = 'full'
    # mask = (
    #         (df_all_model_probs['model_name'] == model_names[0])
    #         &
    #         (df_all_model_probs['dataset_type'] == dataset_type)
    # )
    # filtered = df_all_model_probs[mask]
    # %% Ful dataset best thresholds
    plot_roc_curves(df_all_model_probs,
                    # output_path=path_output.joinpath(f'roc_curve.png'),
                    target_fpr=0.04,
                    zoom=(0.06, 0.9),
                    figsize=(16, 12),
                    fontsize_legend=11,
                    fontsize_lbl=16)
    # %% Visualize the logits
    PLOT_LOGIST = False
    if PLOT_LOGIST:
        plot_model_logits_distribution_by_config(
            df=df_all_model_probs,
            model_name='Elastic Net',
            output_path=None,
            figsize=(18, 10),
            bins=50
        )
    # %% =====================================================================
    # compute the best treshold Stage I
    # =======================================================================
    best_thresholds_max_spec = compute_best_thresholds(df=df_all_model_probs,
                                                       target_fpr=0.004,  # fewer false positives (higher specificity).
                                                       prevalence= 30 / 100000,
                                                       dataset_type='full')

    # print(tabulate(best_thresholds_max_spec, headers=best_thresholds_max_spec.columns))
    cols_report = ['config', 'model_name', 'auc', 'best_threshold_j',
       'sensitivity_j', 'specificity_j', 'ppv_j', 'ppvprev_j']
    print(
        tabulate(
            best_thresholds_max_spec[cols_report],
            headers=best_thresholds_max_spec[cols_report].columns.tolist(),
            tablefmt="psql",
            showindex=False
        )
    )

    # %% test different alpha values and collect for the each model and configuration
    alpha_values = np.linspace(0.001, 0.01, 20)
    df_selected_thresholds = collect_best_thresholds_for_models(
        df=df_all_model_probs,
        alpha_values=alpha_values,
        prevalence=30 / 100000,
        dataset_type='full',
        sensitivity_threshold=0.90,
        specificity_threshold=0.99
    )
    # df_selected_thresholds['sensitivity'] = df_selected_thresholds['sensitivity'].round(4) * 100
    # df_selected_thresholds['specificity'] = df_selected_thresholds['specificity'].round(4) * 100
    # df_selected_thresholds['best_threshold'] = df_selected_thresholds['best_threshold'].round(4)
    PLOT_ROC_ALPHA = False
    if PLOT_ROC_ALPHA:
        plot_roc_curves_with_alpha_points_matched(
            df=df_all_model_probs.loc[df_all_model_probs['model_name'] != 'LogReg (ESS Only)'],
            alpha_values=alpha_values,
            zoom=(0.1, 0.8),
            prevalence=30 / 100000,
            dataset_type='full',
            sensitivity_threshold=0.90,
            specificity_threshold=0.99,
            output_path=path_output.joinpath('roc_curves_alpha_criteria.png'),
        )


    # %% Cross validations
    PLT_CV = False
    if PLT_CV:
        plot_roc_curves_cv(df_all_model_probs,
                        output_dir=None,
                        zoom=(0.1, 0.9),
                        figsize=(12, 6))

    # %% =====================================================================
    # Stage II: compute final metrics with best tresholds computes from the full dataset
    # =======================================================================
    df_metrics_with_ci, df_all_model_probs_veto = compute_metrics_across_folds_with_best_thresholds_and_veto(
        df_probs=df_all_model_probs,
        best_thresholds=df_selected_thresholds, #best_thresholds_max_spec
        best_threshold_col='best_threshold',
        prevalence=PREVALENCE
    )

    cols_quick_view = ['config', 'model_name', 'sensitivity_ci', 'specificity_ci',
       'ppv_ci', 'sensitivity_veto_ci', 'specificity_veto_ci',
       'ppv_veto_ci']
    print(
        tabulate(
            df_metrics_with_ci[cols_quick_view],
            headers=df_metrics_with_ci[cols_quick_view].columns.tolist(),
            tablefmt="psql",
            showindex=False
        )
    )

    df_metrics_with_ci.to_csv(path_output.joinpath('metrics_with_ci.csv'), index=False)

    df_metrics_with_ci_plot = df_metrics_with_ci.copy()
    #%% confusion matrix plot for only the elastic net model
    model_name = 'Elastic Net'
    plot_confusion_matrix_grid(df=df_all_model_probs_veto,
                               figsize=(7, 10),
                               model_name=model_name,)

    # for model_name in df_all_model_probs_veto.model_name.unique():
    #     plot_confusion_matrix_grid(df=df_all_model_probs_veto,
    #                                figsize=(8, 10),
    #                                model_name=model_name,)

    # Compute before and after veto counts
    # Compute counts before and after veto
    df_counts_orig = compute_confusion_counts(df_all_model_probs_veto,
                                              pred_col='prediction',
                                              suffix='orig')
    df_counts_veto = compute_confusion_counts(df_all_model_probs_veto,
                                              pred_col='prediction_hla_veto',
                                              suffix='veto')

    # Merge both sets of counts
    df_confusion_comparison = pd.merge(
        df_counts_orig, df_counts_veto,
        on=['config', 'model_name', 'dataset_type', 'fold_number'],
        how='outer'
    ).fillna(0)

    # Optional: convert all count columns to int
    for col in df_confusion_comparison.columns:
        if col.startswith(('tp_', 'fp_', 'tn_', 'fn_')):
            df_confusion_comparison[col] = df_confusion_comparison[col].astype(int)

    # Optional: Add delta columns
    df_confusion_comparison['fp_diff'] = df_confusion_comparison['fp_orig'] - df_confusion_comparison['fp_veto']
    df_confusion_comparison['tp_diff'] = df_confusion_comparison['tp_orig'] - df_confusion_comparison['tp_veto']

    cols = ['config', 'model_name', 'dataset_type', 'fold_number', 'tp_orig',
            'tp_veto', 'tp_diff', 'fp_orig', 'fp_veto', 'fp_diff']

    print(
        tabulate(
            df_confusion_comparison[cols],
            headers=df_confusion_comparison[cols].columns.tolist(),
            tablefmt="psql",
            showindex=False
        )
    )
    df_confusion_comparison = df_confusion_comparison[cols].copy()
    df_confusion_comparison.to_csv(path_output.joinpath('confusion_matrix_models_config.csv'), index=False)




    # %% Elastic net model
    from library.plot_functions import (plot_model_metrics, plot_model_metrics_specific_columns,
                                        create_venn_diagram, plot_elastic_net_model_coefficients,
                                        plot_dcurves_per_fold, ppv_curve, multi_ppv_plot_combined, plot_calibration,
                                        multi_ppv_plot, multi_calibration_plot)
    # %% Prevalance plot
    best_model = 'Elastic Net'
    df_all_model_probs_elastic_net = df_all_model_probs.loc[df_all_model_probs['model_name'] == best_model, :].copy()

    # Merge best_threshold onto df_val so each row has its config/model's threshold
    df_all_model_probs_elastic_net = pd.merge(
        df_all_model_probs_elastic_net,
        df_selected_thresholds[['config', 'model_name', 'best_threshold']],
        on=['config', 'model_name'],
        how='left'
    )
    df_all_model_probs_elastic_net.rename(columns={'config': 'configuration'}, inplace=True)

    # Convert probabilities to binary predictions using threshold
    df_all_model_probs_elastic_net['prediction'] = (
            df_all_model_probs_elastic_net['predicted_prob'] >= df_all_model_probs_elastic_net['best_threshold']
    ).astype(int)



    multi_ppv_plot_combined(df_predictions_model=df_all_model_probs_elastic_net,
                            figsize=(10, 6),
                            population_prevalence=30 / 100000,  # 0.0003
                            output_path=None)

    # %% calibration plot

    # df_elastic_pred_config['configuration'].replace(feature_set_mapper, inplace=True)

    # all in one figure
    df_brier_scores = multi_calibration_plot(df_predictions=df_all_model_probs_elastic_net,
                                             model_name=best_model,
                                             rows=2,
                                             output_path=None)

    df_grouped = df_brier_scores.groupby('configuration').agg(
        brier_score=('brier_score', lambda x: f"{x.mean():.4f} ({x.std():.4f})"),
        log_loss=('log_loss', lambda x: f"{x.mean():.4f} ({x.std():.4f})"),
        auc=('auc', lambda x: f"{x.mean():.4f} ({x.std():.4f})")
    ).reset_index()

    print(
        tabulate(
            df_grouped,
            headers=df_grouped.columns.tolist(),
            tablefmt="psql",
            showindex=False
        )
    )

    # Assume you have a DataFrame "results" with columns 'true_label' and 'pred_prob'.
    prevalence = 30 / 100000  # i.e., 0.0003
    df_all_model_probs_elastic_net.rename(columns={
                                                    # 'predicted_prob': 'pred_prob',
                                                   'config': 'configuration_formal'},
                                          inplace=True)
    df_all_model_probs_elastic_net['configuration'] = df_all_model_probs_elastic_net['configuration_formal']
    for feature_set_config in df_all_model_probs_elastic_net['configuration_formal'].unique():
        plot_dcurves_per_fold(df_results= df_all_model_probs_elastic_net,
                              prevalence=prevalence,
                              configuration=feature_set_config,
                              output_path=None)









