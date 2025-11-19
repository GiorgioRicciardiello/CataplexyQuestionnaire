import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Tuple


def apply_veto_and_recompute(
        df_predictions_all: pd.DataFrame,
        df_metrics_all: pd.DataFrame,
        *,
        subject_col: str = "subject_id",
        hla_col: str = "DQB10602",
        dqb_flag_col: str = "dqb",
        config_col: str = "config",
        model_col: str = "model_type",
        opt_col: str = "optimization",
        fold_col: str = "outer_fold",
        thr_type_col: str = "threshold_type",
        thr_value_col: str = "threshold_value",
        ytrue_col: str = "y_true",
        yscore_col: str = "y_score",
        ypred_col: str = "y_pred",
)->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply veto rule to non-HLA configs only.
    - HLA configs: duplicated with veto=True but metrics/predictions unchanged.
    - Non-HLA configs: false positives with HLA=0 are flipped to 0, then metrics recomputed.
    params
    dqb_flag_col: indicates if HLA was used during training or not. The veto is implemented only where the hla
    was not in training
    Returns
    -------
    df_predictions_with_veto : predictions with veto=False (original) and veto=True (after veto)
    df_metrics_with_veto     : metrics with veto=False (original) and veto=True (after veto)
    """
    # --- Metrics recompute ---
    def _safe_auc(y_true, y_score):
        if len(np.unique(y_true)) < 2: return np.nan
        return roc_auc_score(y_true, y_score)

    def _safe_ap(y_true, y_score):
        if len(np.unique(y_true)) < 2: return np.nan
        return average_precision_score(y_true, y_score)

    def _reduce(group: pd.DataFrame) -> pd.Series:
        y_true = group[ytrue_col].astype(int).to_numpy()
        y_pred = group[ypred_col].astype(int).to_numpy()
        y_score = group[yscore_col].astype(float).to_numpy()

        auc = _safe_auc(y_true, y_score)
        prc = _safe_ap(y_true, y_score)

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        thr_val = group[thr_value_col].dropna().unique()
        thr_val = float(thr_val[0]) if len(thr_val) else np.nan

        return pd.Series({
            "auc_score": auc,
            "prc_score": prc,
            "sensitivity": sens,
            "specificity": spec,
            "threshold": group[thr_type_col].iloc[0],
            "threshold_value": thr_val,
        })


    df_preds = df_predictions_all.copy()
    if ypred_col not in df_preds.columns:
        # apply the treshold in the predictions in the case where the binary predictions are not present
        df_preds[ypred_col] = (
                df_preds[yscore_col].astype(float) >= df_preds[thr_value_col].astype(float)
        ).astype(int)

    # Veto is applied to the rows that do not have the dqb during training
    non_hla_mask_cfg = ~df_preds[dqb_flag_col].astype(bool)
    assert any(df_preds.loc[non_hla_mask_cfg, dqb_flag_col]) == False

    # Duplicate predictions
    df_preds["veto"] = False  # slice without the veto corrections in the predictions
    df_preds_veto = df_preds.copy()  # slice where we will implment the veto correction
    df_preds_veto["veto"] = True

    # Apply veto only for non-HLA configs
    veto_mask = (
            non_hla_mask_cfg &
            (df_preds_veto[ytrue_col] == 0) &
            (df_preds_veto[ypred_col] == 1) &
            (df_preds_veto[hla_col] == 0)
    )
    df_preds_veto["veto_modified"] = veto_mask.astype(int)
    df_preds_veto.loc[veto_mask, ypred_col] = 0

    # only keep the veto
    # df_preds_veto = df_preds_veto[veto_mask]

    # Combine
    df_predictions_with_veto = pd.concat([df_preds, df_preds_veto], ignore_index=True)


    # --- Compute veto counts ---
    df_veto_counts = (
        df_preds_veto[df_preds_veto["veto_modified"] == 1]
        .groupby([config_col, model_col, opt_col, fold_col, thr_type_col], dropna=False)
        .size()
        .reset_index(name="n_veto_modified")
    )


    grp_keys = [config_col, model_col, opt_col, fold_col, thr_type_col, "veto"]


    # Recompute metrics for all, but then overwrite HLA veto with original
    df_metrics_recomputed = (
        df_predictions_with_veto
        .groupby(grp_keys, dropna=False, as_index=False)
        .apply(_reduce)
        .reset_index(drop=True)
    )

    # Copy original metrics for veto=True where config is HLA
    if dqb_flag_col in df_preds.columns:
        hla_cfgs = df_preds[df_preds[dqb_flag_col].astype(bool)][config_col].unique()
    else:
        hla_cfgs = [c for c in df_preds[config_col].unique() if "hla" in str(c).lower()]

    for cfg in hla_cfgs:
        orig_rows = df_metrics_recomputed[
            (df_metrics_recomputed[config_col] == cfg) &
            (df_metrics_recomputed["veto"] == False)
            ]
        veto_rows_idx = df_metrics_recomputed[
            (df_metrics_recomputed[config_col] == cfg) &
            (df_metrics_recomputed["veto"] == True)
            ].index
        df_metrics_recomputed.loc[
            veto_rows_idx, ["auc_score", "prc_score", "sensitivity", "specificity", "threshold",
                            "threshold_value"]] = \
            orig_rows[
                ["auc_score", "prc_score", "sensitivity", "specificity", "threshold", "threshold_value"]].values

    # Combine with original metrics table if provided
    df_no_veto = df_metrics_all.copy()
    if "veto" not in df_no_veto.columns:
        df_no_veto["veto"] = False
    df_metrics_with_veto = pd.concat([df_no_veto, df_metrics_recomputed[df_metrics_recomputed["veto"] == True]],
                                     ignore_index=True)

    return df_predictions_with_veto, df_metrics_with_veto, df_veto_counts

