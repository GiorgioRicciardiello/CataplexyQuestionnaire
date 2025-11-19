#!/usr/bin/env python3
"""
Utility for plotting group ROC curves with confidence intervals
and an operating point marker (mean sensitivity/specificity).
"""
import textwrap
import pandas as pd
import seaborn as sns
from pathlib import Path
import pathlib
from matplotlib import colors, patches
from typing import Optional, Callable, Dict, Tuple
from sklearn.metrics import roc_curve, confusion_matrix
from library.ml_questionnaire.scoring import select_best_model_type_per_config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from sklearn.metrics import roc_curve
from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy import stats



def plot_models_summary(
        df_metrics: pd.DataFrame,        # long: one row per (model_type, optimization, threshold) with *_ci strings
        df_predictions: pd.DataFrame,    # long: one row per subject × (outer_fold, model_type, optimization, threshold_type)
        class_names: Dict[int, str],
        optimization: str = "auc",
        out_path: Optional[Path] = None,
        font_size_title: int = 14,
        font_size_big_title: int = 18,
        font_size_label: int = 12,
        font_size_legend: int = 10,
        font_size_cm: int = 12,
):
    """
    For each model (row) and threshold_type (columns), plot:
      [ ROC (mean across folds, std band, highlighted τ) | Confusion Matrix ]
    Uses *_ci strings from df_metrics for legend.
    """

    def _scatter_threshold(ax, y_true, y_score, thr_val, color, label=None):
        """Place a marker on the (pooled) ROC at the threshold closest to thr_val."""
        fpr, tpr, thr = roc_curve(y_true, y_score)
        # sklearn's thr excludes 1 last point; guard for empty arrays
        if thr.size == 0:
            return
        idx = np.argmin(np.abs(thr - thr_val))
        ax.scatter(fpr[idx], tpr[idx], color=color, s=70, edgecolors="k", zorder=3, label=label)

    def _make_title(model: str, thr_typ: str, tau: float) -> str:
        model_disp = model.replace("_", " ").title()
        thr_disp = thr_typ.replace("_", " ").replace("0p5", "0.5")
        return fr"$\mathbf{{{model_disp}}}$" + f" — {thr_disp}\n" + fr"$({tau:.2f})$"

    def _draw_confusion_matrix(ax, cm, color, title):
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        cmap = sns.light_palette(color, as_cmap=True)
        im = ax.imshow(cm, cmap=cmap)

        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val = cm[r, c]
                pct = cm_pct[r, c]
                bg = im.cmap(im.norm(cm[r, c]))
                bright = colors.rgb_to_hsv(bg[:3])[2]
                txt_color = "black" if bright > 0.5 else "white"
                ax.text(c, r, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        fontsize=font_size_cm, color=txt_color)

        ax.set_xticks([0, 1]); ax.set_xticklabels([class_names[0], class_names[1]])
        ax.set_yticks([0, 1]); ax.set_yticklabels([class_names[0], class_names[1]])
        ax.set_xlabel("Predicted", fontsize=font_size_label)
        ax.set_ylabel("True", fontsize=font_size_label)
        ax.set_title(title, fontsize=font_size_title)

    def _draw_mean_roc_with_highlight(
            ax,
            *,
            y_true,
            y_score,
            folds,
            color,
            auc_text: str,
            sens_text: str,
            spec_text: str,
            thr_val: float,
            title: str,
            font_size_title: int = 12,
            font_size_label: int = 10,
            font_size_legend: int = 9,
            scatter_func: Optional[Callable] = None
    ):
        # Mean ROC across folds
        mean_fpr = np.linspace(0, 1, 200)
        tprs = []
        for f in np.unique(folds):
            mask = (folds == f)
            if mask.sum() < 2:  # need at least one pos/neg; skip pathological fold
                continue
            fpr, tpr, _ = roc_curve(y_true[mask], y_score[mask])
            # interpolate onto uniform grid
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        if len(tprs) == 0:
            # fallback to pooled ROC if fold-level failed
            fpr, tpr, _ = roc_curve(y_true, y_score)
            mean_fpr = fpr
            mean_tpr = tpr
            std_tpr = np.zeros_like(tpr)
        else:
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)

        ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                label=f"{auc_text}\n{sens_text}\n{spec_text}")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color=color, alpha=0.2)

        if scatter_func is not None:
            scatter_func(ax=ax, y_true=y_true, y_score=y_score, thr_val=thr_val, color=color)

        ax.set_title(title, fontsize=font_size_title)
        ax.set_xlabel("False Positive Rate", fontsize=font_size_label)
        ax.set_ylabel("True Positive Rate", fontsize=font_size_label)
        ax.legend(fontsize=font_size_legend, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)
        return ax

    # --- sanity on optimization ---
    if optimization not in df_predictions["optimization"].unique():
        raise ValueError(f'optimization "{optimization}" not present. '
                         f'Available: {sorted(df_predictions["optimization"].unique())}')

    # establish order for threshold types
    thr_rank = {"0p5": 0, "youden": 1}
    # put "*_max" after those; others go last
    all_thr = sorted(df_predictions["threshold_type"].unique(),
                     key=lambda s: thr_rank.get(s, 2 if str(s).endswith("_max") else 9))

    models = sorted(df_predictions["model_type"].unique())
    n_rows, n_cols = len(models), len(all_thr) * 2  # ROC + CM per threshold
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)  # force 2D

    palette = sns.color_palette("tab10", n_rows)
    model_colors = {m: palette[i] for i, m in enumerate(models)}

    # Precompute class distribution (pooled over selection)
    any_sel = df_predictions[df_predictions["optimization"] == optimization]
    counts = any_sel["y_true"].value_counts().reindex([0, 1]).fillna(0).astype(int)
    big_title = "Class distribution: " + ", ".join([f"{class_names[i]}={counts.get(i, 0)}" for i in [0, 1]])
    big_title += f" | {optimization}"

    for i, model in enumerate(models):
        color = model_colors[model]
        for j, thr_type in enumerate(all_thr):
            ax_roc = axes[i, j * 2]
            ax_cm  = axes[i, j * 2 + 1]

            # ---- slice predictions for this model/opt/threshold_type ----
            sel = df_predictions[
                (df_predictions["model_type"] == model) &
                (df_predictions["optimization"] == optimization) &
                (df_predictions["threshold_type"] == thr_type)
            ]
            if sel.empty:
                ax_roc.axis("off"); ax_cm.axis("off")
                continue

            y_true = sel["y_true"].to_numpy()
            y_score = sel["y_score"].to_numpy()
            folds = sel["outer_fold"].to_numpy()
            y_pred_bin = sel["y_pred"].to_numpy()
            thr_val = float(sel["threshold_value"].mean())  # mean τ across folds for labeling

            # ---- CI strings from df_metrics (one row per model/opt/threshold) ----
            met = df_metrics[
                (df_metrics["model_type"] == model) &
                (df_metrics["optimization"] == optimization) &
                (df_metrics["threshold"] == thr_type)
            ]
            if met.empty:
                # fallback: plain numbers pooled (no CI)
                # compute pooled AUC/Se/Sp if needed; here we just show placeholders
                auc_text  = "AUC=—"
                sens_text = "Se=—"
                spec_text = "Sp=—"
            else:
                auc_text  = f"AUC={met['auc_score_ci'].iloc[0]}"
                sens_text = f"Se={met['sensitivity_ci'].iloc[0]}"
                spec_text = f"Sp={met['specificity_ci'].iloc[0]}"

            # ---- ROC ----
            _draw_mean_roc_with_highlight(
                ax=ax_roc,
                y_true=y_true, y_score=y_score, folds=folds,
                color=color,
                auc_text=auc_text, sens_text=sens_text, spec_text=spec_text,
                thr_val=thr_val,
                title=_make_title(model, thr_type, thr_val),
                font_size_title=font_size_title,
                font_size_label=font_size_label,
                font_size_legend=font_size_legend,
                scatter_func=_scatter_threshold
            )

            # ---- CM ----
            cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
            _draw_confusion_matrix(ax_cm, cm, color, _make_title(model, thr_type, thr_val))

            # subtle background blocks to separate ROC/CM
            def _add_bg(ax, color="lightgray", alpha=0.12):
                rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                         facecolor=color, alpha=alpha, zorder=-1)
                ax.add_patch(rect)
            _add_bg(ax_roc, "lightgray")
            _add_bg(ax_cm, "white")

    fig.suptitle(big_title, fontsize=font_size_big_title, y=0.99)
    plt.tight_layout()
    if out_path:
        if len(models) > 1:
            plt.savefig(out_path / f"plt_all_models_{optimization}.png", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(out_path / f"plt_{models[0]}_{optimization}.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()



def plot_single_model_and_optimization(
    *,
    best_model: str = "xgboost",
    optimization: str = "youden",
    out_path: Optional[pathlib.Path] = None,
    class_names: Dict[int, str] = None,
    df_predictions: pd.DataFrame,
    df_metrics: pd.DataFrame,
) -> None:
    """
    Plot a SINGLE model under a SINGLE optimization using the LONG-format data.

    Expects:
      df_predictions columns ⟶
        ['outer_fold','model_type','optimization','subject_id','y_true','y_score',
         'threshold_type','threshold_value','y_pred']
      df_metrics columns ⟶
        ['model_type','optimization','threshold','auc_score_ci','prc_score_ci',
         'sensitivity_ci','specificity_ci', ... (optionally other columns)]

    This is a thin filter-and-delegate wrapper around plot_models_summary.
    """
    # ---- basic checks ----
    if class_names is None:
        class_names = {0: "Control", 1: "Case"}

    req_pred_cols = {
        "outer_fold","model_type","optimization","subject_id","y_true","y_score",
        "threshold_type","threshold_value","y_pred"
    }
    missing_pred = req_pred_cols - set(df_predictions.columns)
    if missing_pred:
        raise ValueError(f"df_predictions missing columns: {sorted(missing_pred)}")

    req_met_cols = {"model_type","optimization","threshold","auc_score_ci","sensitivity_ci","specificity_ci"}
    missing_met = req_met_cols - set(df_metrics.columns)
    if missing_met:
        raise ValueError(f"df_metrics missing columns: {sorted(missing_met)}")

    # ---- filter to single (model, optimization) ----
    preds_sel = df_predictions[
        (df_predictions["model_type"] == best_model) &
        (df_predictions["optimization"] == optimization)
    ].copy()

    mets_sel = df_metrics[
        (df_metrics["model_type"] == best_model) &
        (df_metrics["optimization"] == optimization)
    ].copy()

    if preds_sel.empty:
        raise ValueError(f"No predictions for model='{best_model}' and optimization='{optimization}'.")
    if mets_sel.empty:
        # Not fatal—plot will show dashes for CI text—but warn early if desired:
        # print(f"Warning: no metrics rows for model='{best_model}', optimization='{optimization}'.")
        pass

    # ---- delegate to the summary plotter (already adapted for long format) ----
    plot_models_summary(
        df_metrics=mets_sel,
        df_predictions=preds_sel,
        class_names=class_names,
        out_path=out_path,
        optimization=optimization,
        font_size_title=14,
        font_size_big_title=18,
        font_size_label=12,
        font_size_legend=12,
        font_size_cm=12,
    )





# %%



def plot_hla_vs_nonhla_roc_and_cm(
        df_predictions_all,
        df_metrics_all,
        *,
        # columns
        config_col="config",
        model_col="model_type",
        opt_col="optimization",
        fold_col="outer_fold",
        subject_col="subject_id",
        ytrue_col="y_true",
        yscore_col="y_score",
        ypred_col="y_pred",
        thr_type_col="threshold_type",
        dqb_flag_col="dqb",
        feature_set_mapper=None,  # pretty names
        class_name={0: "Control", 1: "NT1"},  # axis names for CM
        # selection
        min_spec: float = 98.0, min_sens: float = 70.0,
        prefer_thresholds=("spec_max", "youden", "0p5"),
        # pairing map: non-HLA -> HLA
        pair_map=None,  # {"questionnaire":"questionnaire_hla", ...}
        # ROC view
        xlim=None, ylim=None,
        # fonts
        fonts=None,
        # title
        title="ROC + Confusion Matrices (non-HLA vs HLA)",
        fig_bg="#f7f7f8",          # <— NEW: figure background
        axes_bg="#eeeeee",         # <— NEW: panel background (ROC + CM axes)
):
    """
    2 rows × (2 + N) layout:
      - Cols 0–1: ROC for all selected configs (scatter points at (1-Sp, Se))
      - Cols 2..: per pair of configs (top non-HLA CM, bottom HLA CM)
    Best model per config is chosen from df_metrics_all (no veto).
    Uses metrics already present in df_metrics_all; no recomputation.


    pair_map = {"questionnaire": "questionnaire_hla",
                "ukbb": "ukbb_hla",
                "ess": "ess_hla"}

    fig = plot_hla_vs_nonhla_roc_and_cm(
        df_predictions_all, df_metrics_all,
        feature_set_mapper=feature_set_mapper,
        class_name={0: "CNT", 1: "NT1"},
        pair_map=pair_map,
        min_spec=98.0, min_sens=70.0,
        xlim=(0, 0.2), ylim=(0.8, 1.0),
        title="ROC + CMs — non-HLA vs HLA",
        fonts={"scale": 1.5},
        # fig_size=(12, 10)
    )
    plt.show()


    """

    def _choose_palette(keys, palette="colorblind"):
        """
        Return {key: color} for the given iterable of keys.
        `palette` can be:
          - Seaborn palette name (e.g., "colorblind", "Set2", "pastel", "deep", "tab10")
          - Matplotlib colormap: prefix with "mpl:", e.g. "mpl:tab20", "mpl:viridis"
          - A list/tuple of hex/RGB colors
          - A callable n -> list of colors
        """
        keys = list(sorted(keys))
        n = len(keys)

        # resolve palette to a list of RGB tuples
        if isinstance(palette, str):
            if palette.lower().startswith("mpl:"):
                cmap = plt.get_cmap(palette.split(":", 1)[1])
                cols = [cmap(i / max(n - 1, 1)) for i in range(n)]
            else:
                cols = sns.color_palette(palette, n)
        elif isinstance(palette, (list, tuple)):
            cols = list(palette)
        elif callable(palette):
            cols = list(palette(n))
        else:
            cols = sns.color_palette("tab10", n)

        # cycle if fewer colors than keys
        if len(cols) < n:
            cols = [cols[i % len(cols)] for i in range(n)]

        return {k: cols[i] for i, k in enumerate(keys)}

    def _plot_cm_sns(
            ax,
            cm: np.ndarray,
            title: str,
            class_name,
            base_color="C0",
            fonts: dict = None
    ):
        """
        Seaborn confusion matrix:
          - pastel colormap derived from base_color
          - annotations show "count\npct%"
          - white grid lines for separation
          - font sizes controlled via `fonts` dict
            keys: {"cm", "title", "tick"}
        """
        fonts = fonts or {}
        total = cm.sum() if cm.sum() > 0 else 1

        # labels with count + percentage
        labels = np.array([
            [f"{cm[0, 0]:,}\n{cm[0, 0] / total * 100:.1f}%", f"{cm[0, 1]:,}\n{cm[0, 1] / total * 100:.1f}%"],
            [f"{cm[1, 0]:,}\n{cm[1, 0] / total * 100:.1f}%", f"{cm[1, 1]:,}\n{cm[1, 1] / total * 100:.1f}%"]
        ])

        # Light/pastel cmap tied to the ROC color
        cmap = sns.light_palette(base_color, as_cmap=True)

        sns.heatmap(
            cm, ax=ax, cmap=cmap, cbar=False, square=True,
            vmin=0, vmax=max(1, cm.max()),
            linewidths=2, linecolor="white",
            annot=labels, fmt="",
            annot_kws={
                "fontsize": fonts.get("cm", 14),
                "ha": "center",
                "va": "center",
                "color": "black"
            }
        )

        # Titles and ticks with adjustable fonts
        ax.set_title(title, fontsize=fonts.get("title", 14))
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])  # center ticks
        ax.set_xticklabels([f"Pred {class_name[0]}", f"Pred {class_name[1]}"],
                           fontsize=fonts.get("tick", 12))
        ax.set_yticklabels([f"True {class_name[0]}", f"True {class_name[1]}"],
                           fontsize=fonts.get("tick", 12))
        ax.set_xlabel("")
        ax.set_ylabel("")

    def _confusion_counts(y_true, y_pred):
        y_true = y_true.astype(int);
        y_pred = y_pred.astype(int)
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        return np.array([[int(tn), int(fp)], [int(fn), int(tp)]])


    feature_set_mapper = feature_set_mapper or {}

    # --- select best model per configuration (no veto) ---
    sel = select_best_model_type_per_config(
        df_metrics_all, min_spec=min_spec, min_sens=min_sens, prefer_thresholds=prefer_thresholds
    ).copy()

    # build default pair map if not supplied (only from what is present)
    if pair_map is None:
        cfgs = set(sel[config_col])
        maybe = {}
        for base in ["questionnaire", "ukbb", "ess"]:
            non, hla = base, f"{base}_hla"
            if non in cfgs and hla in cfgs:
                maybe[non] = hla
        pair_map = maybe

    pairs = [(non, hla) for non, hla in pair_map.items()
             if (non in set(sel[config_col])) and (hla in set(sel[config_col]))]


    if not pairs:
        raise ValueError("No (non-HLA, HLA) pairs found in selection. Provide a 'pair_map' if names differ.")

    # colors per config for ROC + CMs
    # cfgs_sorted = sorted(sel[config_col].unique())
    # palette = plt.rcParams['axes.prop_cycle'].by_key().get('color',
    #                                                        ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
    #                                                         "C9"])
    # color_map = {cfg: palette[i % len(palette)] for i, cfg in enumerate(cfgs_sorted)}

    cfgs_sorted = sorted(sel[config_col].unique())

    # pick one:
    # color_map = choose_palette(cfgs_sorted, palette="colorblind")   # safe default
    # color_map = choose_palette(cfgs_sorted, palette="Set2")         # soft pastel
    # color_map = choose_palette(cfgs_sorted, palette="mpl:tab20")    # up to 20 distinct
    # color_map = choose_palette(cfgs_sorted, palette=["#1f77b4","#ff7f0e","#2ca02c"])

    color_map = _choose_palette(cfgs_sorted, palette="colorblind")

    def _fetch_preds_for_sel(row):
        """helper to fetch predictions for a selected row & its threshold type"""
        cfg, mdl, opt = row[config_col], row[model_col], row[opt_col]
        thr_type = row.get("threshold", row.get("threshold_type", None))
        q = (
                (df_predictions_all[config_col] == cfg) &
                (df_predictions_all[model_col] == mdl) &
                (df_predictions_all[opt_col] == opt)
        )
        if thr_type is not None and thr_type_col in df_predictions_all.columns:
            q = q & (df_predictions_all[thr_type_col] == thr_type)
        preds = df_predictions_all[q].copy()
        keys = [c for c in (fold_col, subject_col) if c in preds.columns]
        if keys:
            preds = preds.sort_values(keys).drop_duplicates(subset=keys, keep="first")
        return preds

    # fonts
    font_scale = fonts.get("scale", 1.5)  # e.g., 1.5x larger than defaults
    rc = {
        "font.family": fonts.get("family", "DejaVu Sans"),
        "font.size": 10 * font_scale,
        "axes.titlesize": 12 * font_scale,
        "axes.labelsize": 10 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
        "legend.fontsize": 9 * font_scale,
    }

    n_pairs = len(pairs)
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(6.5 + 4.3 * n_pairs, 8),
                         facecolor=fig_bg,
                         )
        gs = GridSpec(
            nrows=2, ncols=2 + n_pairs, figure=fig,
            width_ratios=[1.4, 1.4] + [1] * n_pairs,
            wspace=0.35, hspace=0.3
        )

        # ---------- ROC (two columns, both rows) ----------
        ax_roc = fig.add_subplot(gs[:, :2], facecolor=axes_bg)
        ax_roc.set_title("ROC — Best model per configuration")
        for _, row in sel.iterrows():
            cfg = row[config_col]
            label_formal = feature_set_mapper.get(cfg, row.get("feature_set_label", cfg))

            preds = _fetch_preds_for_sel(row)
            if preds.empty:
                continue
            y_true = preds[ytrue_col].astype(int).to_numpy()
            y_score = preds[yscore_col].astype(float).to_numpy()
            fpr, tpr, _ = roc_curve(y_true, y_score)

            # Use metrics already computed (strings or floats OK)
            auc_txt = row["auc_score_ci"] if "auc_score_ci" in row else f"{row['auc_score']:.3f}"
            se = float(row["sensitivity"]);
            sp = float(row["specificity"])

            # lw = 3.5 if bool(row[dqb_flag_col]) else 2.0

            # lw = 3.5 if bool(row[dqb_flag_col]) else 2.0
            ls = '-' if bool(row[dqb_flag_col]) else '--'  # Dashed if flagged, solid otherwise

            color = color_map[cfg]
            ax_roc.plot(fpr, tpr,
                        color=color,
                        # linewidth=lw,
                        linestyle=ls,
                        label=f"$\\bf{{{label_formal}}}$\nAUC:{auc_txt}, Se:{se:2f}, Sp:{sp:2f}",
                        # zorder=3 if lw > 2 else 2
                        )
            # scatter point at operating point
            ax_roc.scatter(1.0 - sp, se, s=60, color=color, edgecolor="black", linewidths=0.5, zorder=4)

        ax_roc.plot([0, 1], [0, 1], "--", color="#777", linewidth=1)
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
        ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
        if xlim: ax_roc.set_xlim(*xlim)
        if ylim: ax_roc.set_ylim(*ylim)
        ax_roc.grid(True, alpha=0.3)
        ax_roc.legend(loc="lower right", frameon=True)

        # ---------- CMs per pair (non-HLA top, HLA bottom) ----------
        for j, (non_cfg, hla_cfg) in enumerate(pairs, start=2):
            row_non = sel[sel[config_col] == non_cfg].iloc[0]
            row_hla = sel[sel[config_col] == hla_cfg].iloc[0]

            # top: non-HLA
            ax0 = fig.add_subplot(gs[0, j], facecolor=axes_bg)
            preds_non = _fetch_preds_for_sel(row_non)
            cm_non = _confusion_counts(preds_non[ytrue_col].to_numpy(), preds_non[ypred_col].to_numpy())
            title_cm_non_hla = feature_set_mapper.get(non_cfg, row_non.get('feature_set_label', non_cfg))
            _plot_cm_sns(ax=ax0,
                         cm=cm_non,
                     title=f"{title_cm_non_hla}",
                         fonts=fonts,
                            class_name=class_name,
                         base_color=color_map[non_cfg])

            # bottom: HLA
            ax1 = fig.add_subplot(gs[1, j], facecolor=axes_bg)
            preds_hla = _fetch_preds_for_sel(row_hla)
            cm_hla = _confusion_counts(preds_hla[ytrue_col].to_numpy(), preds_hla[ypred_col].to_numpy())
            title_cm_hla = feature_set_mapper.get(hla_cfg, row_hla.get('feature_set_label', hla_cfg)).replace('+', '\n+')
            _plot_cm_sns(ax=ax1,
                         cm=cm_hla,
                     title=f"{title_cm_hla}",
                     class_name=class_name,
                         base_color=color_map[hla_cfg])

        fig.suptitle(title, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        return fig


def _compute_metrics_from_cm(cm: np.ndarray, prevalence: float = 30 / 100_000) -> dict:
    """
    Compute sensitivity, specificity, PPV, and NPV given a confusion matrix and population prevalence.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix as [[TN, FP], [FN, TP]].
    prevalence : float, default=30/100_000
        Disease prevalence in the population (e.g. 30 per 100,000).

    Returns
    -------
    metrics : dict
        Dictionary with sensitivity, specificity, PPV, NPV.
    """
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # Apply Bayes’ theorem for predictive values at given prevalence
    ppv = (sens * prevalence) / (sens * prevalence + (1 - spec) * (1 - prevalence))
    npv = (spec * (1 - prevalence)) / ((1 - sens) * prevalence + spec * (1 - prevalence))

    return {
        "sensitivity": sens,
        "specificity": spec,
        "PPV": ppv,
        "NPV": npv
    }

def compute_fold_metrics(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Compute sensitivity, specificity, and AUC for each outer_fold,
    then return a summary with 95% CI across folds.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['outer_fold', 'y_true', 'y_score'].
    threshold : float, optional
        Threshold for binary classification. Default 0.5.

    Returns
    -------
    pd.DataFrame
        Summary table with mean ± 95% CI for AUC, sensitivity, and specificity.
    """
    results = []
    for fold, d in df.groupby("outer_fold"):
        y_true, y_score = d["y_true"].values, d["y_score"].values
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        auc = roc_auc_score(y_true, y_score)

        results.append({"outer_fold": fold, "AUC": auc, "Sensitivity": sens, "Specificity": spec})

    df_metrics = pd.DataFrame(results)

    def ci(series):
        vals = series.dropna().values
        n = len(vals)
        if n <= 1:
            return np.nan, np.nan, np.nan
        mean = np.mean(vals)
        sem = stats.sem(vals)
        h = sem * stats.t.ppf(0.975, n - 1)
        return mean, mean - h, mean + h

    summary = {}
    for metric in ["AUC", "Sensitivity", "Specificity"]:
        mean, low, high = ci(df_metrics[metric])
        summary[metric] = f"{mean:.3f} ({low:.3f}, {high:.3f})"

    return pd.DataFrame([summary])


def plot_hla_vs_nonhla_roc_and_cm_with_veto(
        df_predictions_all,
        df_metrics_all,
        *,
        # columns
        config_col="config",
        model_col="model_type",
        opt_col="optimization",
        fold_col="outer_fold",
        subject_col="subject_id",
        ytrue_col="y_true",
        yscore_col="y_score",
        ypred_col="y_pred",
        thr_type_col="threshold_type",
        dqb_flag_col="dqb",
        feature_set_mapper=None,  # pretty names
        class_name={0: "Control", 1: "NT1"},
        # selection
        min_spec: float = 98.0, min_sens: float = 70.0,
        prefer_thresholds:Tuple[str]=None,
        # pairing map
        pair_map=None,
        # ROC view
        xlim=None,
        ylim=None,
        # fonts
        fonts=None,
        prevalence:float=30 / 100_000,
        # title
        title="ROC + Confusion Matrices (non-HLA vs HLA vs Veto)",
        palette:str='Set2',
        fig_bg="#f7f7f8",
        axes_bg="#eeeeee",
        output_path:Path=None,
)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    3 rows × (2 + N) layout:
      - Cols 0–1: ROC for all selected configs (spanning all 3 rows)
      - Cols 2..:
           Row 0 = non-HLA CM
           Row 1 = HLA CM
           Row 2 = post-veto CM (non-HLA only)
    Best model per config is chosen from df_metrics_all (no veto).

        fig = plot_hla_vs_nonhla_roc_and_cm_with_veto(
        df_predictions_with_veto, df_metrics_with_veto,
        feature_set_mapper=feature_set_mapper,
        class_name={0: "CNT", 1: "NT1"},
        pair_map=pair_map,
        min_spec=98.0, min_sens=70.0,
        xlim=(0, 0.2), ylim=(0.8, 1.0),
        title="ROC + CMs — non-HLA vs HLA",
        fonts={"scale": 1.5},
        # fig_size=(12, 10)
        )
        plt.show()

    :returns
        def_selected_best: pd.DataFrame, best metric for each feature set, selected among the best models for each
        configuration set
        df_per_model_best: pd.DataFrame, for each of the models, it assings the best model to each configuration set,
        this frame have for each feature set a different model_type.
        df_comparison_veto_metrics: pd.DataFrame, metrics comparing veto vs non veto

    """

    # -------- helpers --------
    def wrap_title(text, width=20):
        return "\n".join(textwrap.wrap(text, width))

    def _choose_palette(keys, palette="colorblind"):
        keys = list(sorted(keys))
        n = len(keys)
        cols = sns.color_palette(palette, n)
        if len(cols) < n:
            cols = [cols[i % len(cols)] for i in range(n)]
        return {k: cols[i] for i, k in enumerate(keys)}

    def _confusion_counts(y_true, y_pred):
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        return np.array([[int(tn), int(fp)], [int(fn), int(tp)]])

    def _plot_cm_sns(ax,
                     cm,
                     title: str,
                     class_name: Dict[int, str],
                     base_color="C0",
                     fonts: dict = None):
        fonts = fonts or {}

        # row sums for normalization
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        row_perc = cm / row_sums * 100

        labels = np.array([
            [f"{cm[0, 0]:,}\n{row_perc[0, 0]:.1f}%", f"{cm[0, 1]:,}\n{row_perc[0, 1]:.1f}%"],
            [f"{cm[1, 0]:,}\n{row_perc[1, 0]:.1f}%", f"{cm[1, 1]:,}\n{row_perc[1, 1]:.1f}%"]
        ])

        cmap = sns.light_palette(base_color, as_cmap=True)
        sns.heatmap(cm, ax=ax, cmap=cmap, cbar=False, square=True,
                    vmin=0, vmax=max(1, cm.max()),
                    linewidths=2, linecolor="white",
                    annot=labels, fmt="",
                    annot_kws={"fontsize": fonts.get("cm", 14),
                               "ha": "center", "va": "center", "color": "black"})
        ax.set_title(title, fontsize=fonts.get("title", 14))
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels([f"Pred {class_name[0]}", f"Pred {class_name[1]}"],
                           fontsize=fonts.get("tick", 12))
        ax.set_yticklabels([f"True {class_name[0]}", f"True {class_name[1]}"],
                           fontsize=fonts.get("tick", 12))
        ax.set_xlabel("")
        ax.set_ylabel("")

    def _fetch_preds_for_sel(row, veto_flag=None):
        """
        Retrieve the prediction rows corresponding to a given configuration/model/optimization
        (and optionally threshold type and veto flag).

        Parameters
        ----------
        row : pd.Series
            A row from `df_metrics_all` (or a similar metrics DataFrame). Must contain:
              - config_col
              - model_col
              - opt_col
            Optionally may contain 'threshold' or 'threshold_type'.
        veto_flag : bool or None, default=None
            - None → ignore veto flag, return all matching predictions
            - False → return only predictions BEFORE veto corrections
            - True  → return only predictions AFTER veto corrections (i.e. with veto applied)

        Returns
        -------
        preds : pd.DataFrame
            Subset of `df_predictions_all` matching the given configuration/model/optimization
            (and threshold, if present). If `veto_flag=True`, returns the corrected
            predictions (i.e., where false positives were flipped to 0).
            If multiple rows per (subject, fold) exist, only the first is kept.

        Notes
        -----
        - When `veto_flag=True`, this does not return only the rows *that were modified* by the veto.
          Instead, it returns the **entire prediction set after applying veto** for the given config/model/opt.
          So:
              preds_non_veto → all predictions without veto
              preds_non_hla  → all predictions trained with HLA included
              preds_non_veto (with veto=True) → all predictions after veto, where some rows may have been flipped.
        - If you want to isolate *only* the subset of rows actually changed by the veto rule,
          you need an extra boolean mask column (e.g., `veto_modified`) and filter on it.
        """
        cfg, mdl, opt = row[config_col], row[model_col], row[opt_col]
        thr_type = row.get("threshold", row.get("threshold_type", None))
        q = ((df_predictions_all[config_col] == cfg) &
             (df_predictions_all[model_col] == mdl) &
             (df_predictions_all[opt_col] == opt))
        if thr_type is not None and thr_type_col in df_predictions_all.columns:
            q = q & (df_predictions_all[thr_type_col] == thr_type)
        if veto_flag is not None and "veto" in df_predictions_all.columns:
            q = q & (df_predictions_all["veto"] == veto_flag)
        preds = df_predictions_all[q].copy()
        keys = [c for c in (fold_col, subject_col) if c in preds.columns]
        if keys:
            preds = preds.sort_values(keys).drop_duplicates(subset=keys, keep="first")
        return preds

    def _compute_veto_comparison(df_predictions_all, df_per_model_best):
        """Compute per-fold metrics before and after veto correction."""
        records = []

        for _, row in df_per_model_best.iterrows():
            cfg = row["config"]
            if "hla" in str(cfg).lower():
                continue

            # get predictions pre- and post-veto using helper
            preds_pre = _fetch_preds_for_sel(row, veto_flag=False)
            preds_post = _fetch_preds_for_sel(row, veto_flag=True)

            if preds_pre.empty or preds_post.empty:
                continue

            df_pre = compute_fold_metrics(df=preds_pre)
            df_post = compute_fold_metrics(df=preds_post)

            for df_m, vflag in [(df_pre, False), (df_post, True)]:
                df_m["config"] = cfg
                df_m["model_type"] = row["model_type"]
                df_m["optimization"] = row["optimization"]
                df_m["threshold_type"] = row.get("threshold_type", None)
                df_m["veto"] = vflag
                records.append(df_m)

        if not records:
            print("⚠️ No veto comparisons available.")
            return pd.DataFrame()

        df_veto_metrics = (
            pd.concat(records, ignore_index=True)
            .sort_values(["config", "model_type", "veto"])
            .reset_index(drop=True)
        )
        return df_veto_metrics

    # -------- function --------
    formal_threshold_name = {
        '0p5': r'$\tau$',
        'youden': r'$\tau^{*}$',
        'spec_max': r'$\tau_{\text{sp}}$',
        'sens_max': r'$\tau_{\text{se}}$',
    }

    formal_model_names = {
        'random_forest': 'RF',
        'LDA': 'LDA',
        'SVM': 'SVM',
        'xgboost': 'XGB',
        'elastic_net': 'ElasticNet',

    }
    if prefer_thresholds is None:
        prefer_thresholds = ("spec_max", "youden", "0p5")

    feature_set_mapper = feature_set_mapper or {}

    # apply the veto to the score
    df_predictions_all.loc[df_predictions_all['veto_modified'] == 1, 'y_score'] = 0

    # --- select best model per config (no veto) ---
    def_selected_best, df_per_model_best = select_best_model_type_per_config(
        df_metrics=df_metrics_all,
        min_spec=min_spec,
        min_sens=min_sens,
        prefer_thresholds=prefer_thresholds
    )
    cols_report = ['feature_set_label', 'model_type', 'auc_score_ci', 'sensitivity_ci', 'specificity_ci']
    print(
        tabulate(
            df_per_model_best[cols_report],
            headers=cols_report,
            tablefmt='psql'
        )
    )

    print(
        tabulate(
            def_selected_best[cols_report],
            headers=cols_report,
            tablefmt='psql'
        )
    )
    # --- Compute metrics pre- and post-veto for each selected best model (non-HLA only) ---
    df_comparison_veto_metrics = _compute_veto_comparison(df_predictions_all, df_per_model_best)

    # quick preview comparison table
    if not df_comparison_veto_metrics.empty:
        df_comparison_veto_metrics_pivot = df_comparison_veto_metrics.pivot(index=["config", "model_type"],
                                         columns="veto",
                                         values=["AUC", "Sensitivity", "Specificity", ])

        print("\n=== VETO comparison summary ===")
        print(
            tabulate(
                df_comparison_veto_metrics_pivot,
                headers="keys", tablefmt="psql"
            )
        )


    # build default pair map if not supplied
    if pair_map is None:
        cfgs = set(def_selected_best[config_col])
        maybe = {}
        for base in ["questionnaire", "ukbb", "ess"]:
            non, hla = base, f"{base}_hla"
            if non in cfgs and hla in cfgs:
                maybe[non] = hla
        pair_map = maybe

    pairs = [(non, hla) for non, hla in pair_map.items()
             if (non in set(def_selected_best[config_col])) and (hla in set(def_selected_best[config_col]))]
    if not pairs:
        raise ValueError("No (non-HLA, HLA) pairs found.")


    # colors per config
    cfgs_sorted = sorted(def_selected_best[config_col].unique())
    # pick one:
    # color_map = choose_palette(cfgs_sorted, palette="colorblind")   # safe default
    # color_map = choose_palette(cfgs_sorted, palette="Set2")         # soft pastel
    # color_map = choose_palette(cfgs_sorted, palette="mpl:tab20")    # up to 20 distinct
    # color_map = choose_palette(cfgs_sorted, palette=["#1f77b4","#ff7f0e","#2ca02c"])

    color_map = _choose_palette(cfgs_sorted, palette=palette)

    # fonts
    font_scale = fonts.get("scale", 1.2) if fonts else 1.2
    rc = {
        "font.family": fonts.get("family", "DejaVu Sans") if fonts else "DejaVu Sans",
        "font.size": 10 * font_scale,
        "axes.titlesize": 12 * font_scale,
        "axes.labelsize": 10 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
        "legend.fontsize": 9 * font_scale,
    }
    n_pairs = len(pairs)
    all_metrics = []
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(6.5 + 4.3 * n_pairs, 12),
                         facecolor=fig_bg)
        gs = GridSpec(
            nrows=3, ncols=3 + n_pairs, figure=fig,
            width_ratios=[1.3, 1.3, 1.3] + [1] * n_pairs,
            # width_ratios=[1.5, 1.5, 1.5] + [1] * n_pairs,
            wspace=0.35, hspace=0.4
        )

        # ---------- ROC spanning all 3 rows ----------
        ax_roc = fig.add_subplot(gs[:, :3], facecolor=axes_bg)
        ax_roc.set_title("ROC — Best model per configuration")
        for _, row in def_selected_best.iterrows():
            cfg = row[config_col]
            label_formal = feature_set_mapper.get(cfg, row.get("feature_set_label", cfg))
            preds = _fetch_preds_for_sel(row, veto_flag=False)
            if preds.empty:
                continue
            y_true = preds[ytrue_col].astype(int).to_numpy()
            y_score = preds[yscore_col].astype(float).to_numpy()
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc_txt = row.get("auc_score_ci", f"{row['auc_score']:.3f}")
            se = float(row["sensitivity"])
            sp = float(row["specificity"])
            ls = '-' if bool(row[dqb_flag_col]) else '--'
            tau = row['threshold_value']
            tau_name = formal_threshold_name.get(row['threshold'])
            model_type_name = formal_model_names.get(row["model_type"])
            fpr, tpr, thresholds = roc_curve(y_true, y_score)

            color = color_map[cfg]
            # ax_roc.plot(fpr, tpr,
            #             color=color, linestyle=ls,
            #             label=f"$\\bf{{{label_formal}}}$\nAUC:{auc_txt}|Se:{se:.2f}|Sp:{sp:.2f}|τ={tau:.3f}")

            ax_roc.plot(
                fpr, tpr,
                color=color,
                linestyle=ls,
                label=(
                    f"$\\bf{{{label_formal}}}$\n"
                    f"AUC: {auc_txt} | "
                    f"{model_type_name} | "
                    f"{tau_name}: {tau:.3f}"
                )
            )

            # --- scatter the operating point from metrics (for consistency) ---
            # ax_roc.scatter(1.0 - sp, se, s=80, color=color,
            #                edgecolor="black", linewidths=0.6, zorder=4, marker="o")

            # --- scatter the τ threshold on the ROC ---
            # Find closest threshold from sklearn’s thresholds
            idx_tau = np.argmin(np.abs(thresholds - tau))
            ax_roc.scatter(fpr[idx_tau], tpr[idx_tau], s=100, color=color,
                           edgecolor="black", linewidths=1.2, marker="D", zorder=5,
                           # label=f"τ={tau:.3f} ({label_formal})"
                           )

        ax_roc.plot([0, 1], [0, 1], "--", color="#777", linewidth=1)
        ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
        ax_roc.set_ylabel("True Positive Rate (Sensitivity)")
        if xlim: ax_roc.set_xlim(*xlim)
        if ylim: ax_roc.set_ylim(*ylim)
        ax_roc.grid(True, alpha=0.3)
        # ax_roc.legend(loc="lower right", frameon=True)
        leg = ax_roc.legend(loc="lower right", frameon=True)
        for legline in leg.get_lines():
            legline.set_linewidth(4.0)  # <— thicker legend lines only

        # ---------- CMs per pair ----------
        for j, (non_cfg, hla_cfg) in enumerate(pairs, start=3):
            row_non = def_selected_best[def_selected_best[config_col] == non_cfg].iloc[0]
            row_hla = def_selected_best[def_selected_best[config_col] == hla_cfg].iloc[0]

            # row 0: non-HLA
            ax0 = fig.add_subplot(gs[0, j], facecolor=axes_bg)
            preds_non = _fetch_preds_for_sel(row_non, veto_flag=False)
            cm_non = _confusion_counts(preds_non[ytrue_col], preds_non[ypred_col])
            title_no_hla = wrap_title(feature_set_mapper.get(non_cfg, non_cfg))
            _plot_cm_sns(ax=ax0,
                         cm=cm_non,
                         title=title_no_hla,
                         class_name=class_name,
                         base_color=color_map[non_cfg])
            metrics_hla = _compute_metrics_from_cm(cm_non, prevalence=prevalence)
            metrics_hla.update({"config": non_cfg,
                                "type": "non-HLA",
                                'feature_set': title_no_hla})
            all_metrics.append(metrics_hla)

            # row 1: HLA
            ax1 = fig.add_subplot(gs[1, j], facecolor=axes_bg)
            preds_hla = _fetch_preds_for_sel(row_hla, veto_flag=False)
            cm_hla = _confusion_counts(preds_hla[ytrue_col], preds_hla[ypred_col])
            title_hla = feature_set_mapper.get(hla_cfg, hla_cfg).replace('+', '\n+')
            _plot_cm_sns(ax=ax1,
                         cm=cm_hla,
                         title=title_hla,
                         class_name=class_name,
                         base_color=color_map[hla_cfg])
            metrics_hla = _compute_metrics_from_cm(cm_hla, prevalence=prevalence)
            metrics_hla.update({"config": hla_cfg, "type": "HLA",
                                'feature_set': title_hla})
            all_metrics.append(metrics_hla)

            # row 2: veto (non-HLA only)
            ax2 = fig.add_subplot(gs[2, j], facecolor=axes_bg)
            preds_non_veto = _fetch_preds_for_sel(row_non, veto_flag=True)
            print(f'Observations affected veto in {non_cfg}: {preds_non_veto.veto_modified.sum()}')
            if not preds_non_veto.empty:
                cm_veto = _confusion_counts(preds_non_veto[ytrue_col], preds_non_veto[ypred_col])
                title_veto = wrap_title(feature_set_mapper.get(non_cfg, non_cfg) + "\n(veto-corrected)")
                _plot_cm_sns(ax=ax2,
                             cm=cm_veto,
                             title=title_veto,
                             class_name=class_name,
                             base_color=color_map[non_cfg])
                metrics_veto = _compute_metrics_from_cm(cm_veto, prevalence=prevalence)
                metrics_veto.update({"config": non_cfg,
                                     "type": "Veto",
                                     'feature_set':title_veto})
                all_metrics.append(metrics_veto)
            else:
                ax2.axis("off")

        fig.suptitle(title, y=0.995)
        if output_path:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.show()

        df_metrics_prevalence = pd.DataFrame(all_metrics)
        print(tabulate(df_metrics_prevalence, headers="keys", tablefmt="psql"))

        # fig.tight_layout(rect=[0, 0, 1, 0.98])
        # return fig
        return def_selected_best, df_per_model_best, df_metrics_prevalence, df_comparison_veto_metrics
