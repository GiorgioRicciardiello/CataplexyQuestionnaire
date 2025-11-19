import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.optimize import minimize_scalar
from scipy.stats import norm

def find_best_threshold_for_predictions(y_true_train: np.ndarray,
                                        y_pred_train: np.ndarray,
                                        metric: str = 'specificity') -> float:
    """
    Find the best threshold for binary classification predictions based on a specific metric.
    Uses optimization for fine-tuned threshold selection.

    :param y_true_train: Ground truth binary labels.
    :param y_pred_train: Predicted probabilities (or scores).
    :param metric: Metric to optimize. Options: 'f1', 'accuracy', 'precision', 'recall', 'auc'.
    :return: Best threshold based on the metric.
    """

    def metric_for_threshold(threshold):
        y_pred_thresh = (y_pred_train >= threshold).astype(int)
        if metric == 'f1':
            return -f1_score(y_true_train, y_pred_thresh)
        elif metric == 'accuracy':
            return -accuracy_score(y_true_train, y_pred_thresh)
        elif metric == 'sensitivity':  # Sensitivity (Recall)
            return -recall_score(y_true_train, y_pred_thresh)
        elif metric == 'specificity':  # Specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(y_true_train, y_pred_thresh).ravel()
            specificity = tn / (tn + fp)
            return -specificity
        elif metric == 'auc':
            return -roc_auc_score(y_true_train, y_pred_thresh)
        else:
            raise ValueError("Unsupported metric. Choose from 'f1', 'accuracy', 'sensitivity', 'specificity', 'auc'.")

    # Use scalar minimization for the threshold search
    result = minimize_scalar(metric_for_threshold, bounds=(0.0, 1.0), method='bounded')
    best_threshold = result.x
    best_metric_value = -result.fun

    print(f"Best threshold based on {metric}: {best_threshold:.4f} with score: {best_metric_value:.4f}")
    return best_threshold

def bootstrap_auc_ci(y_true, y_prob, n_bootstraps=1000, ci=0.95):
    aucs = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue  # Skip if resampled set doesn't have both classes
        auc = roc_auc_score(y_true[indices], y_prob[indices])
        aucs.append(auc)
    aucs = np.array(aucs)
    lower = np.percentile(aucs, ((1 - ci) / 2) * 100)
    upper = np.percentile(aucs, (1 - (1 - ci) / 2) * 100)
    return np.mean(aucs), (lower, upper)

def compute_metrics(y_pred: np.ndarray,
                    y_true: np.ndarray,
                    prevalence:float=None) -> Dict[str, float]:
    """
    Compute classification metrics including sensitivity and specificity with confidence intervals.

    :param y_pred: array of predictions (binary)
    :param y_true: array of true labels (binary)
    :return: Dictionary containing metrics and their confidence intervals.
    """
    # Set constant prevalence (30 per 100,000 adults) for Narcolepsy
    if prevalence is None:
        prevalence = 30 / 100000  # which is 0.0003

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Compute metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    # Compute PPV using constant prevalence via Bayes' theorem.
    # This adjusts PPV for a target population prevalence rather than the sample prevalence.
    denominator = (sensitivity * prevalence) + (1 - specificity) * (1 - prevalence)
    ppv = (sensitivity * prevalence) / denominator if denominator > 0 else 0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'ppv_apparent': precision,  # true positive rate in your test fold
        'f1_score': f1_score,
        'npv': npv,
        'fpr': fpr,
        'fnr': fnr,
        'ppv': ppv, # prevalence-adjusted using Bayes
    }


def compute_confidence_interval(values: list) -> str:
    """Compute 95% confidence interval across folds."""
    mean_val = np.mean(values)
    std_error = np.std(values, ddof=1) / np.sqrt(len(values))  # Standard Error
    margin = norm.ppf(0.975) * std_error  # 1.96 * std_error for 95% CI
    ci = max(0, mean_val - margin), min(1, mean_val + margin)
    mean_val = str(round(mean_val, 4))[0:5]  # mean value < upper CI, so we cannot truncate at output
    return f'{mean_val},\n({float(ci[0]):.4}, {float(ci[1]):.4})'


def apply_veto_rule_re_classifications(df_classifications:pd.DataFrame,
                                       df_data:pd.DataFrame):
    """
    Apply the veto rule:
    1. Include an 'HLA' column in the classifications dataframe
    2. Identify wrongly classified observations where DQB10602 == 0 in non-HLA folds.
    3. Correct these misclassifications.
    4. Recalculate specificity.

    Parameters:
    - df_classifications (pd.DataFrame): Classification results.
    - df_data (pd.DataFrame): Original dataset including DQB10602 column.

    Returns:
    - pd.DataFrame: Updated classifications with veto rule applied and specificity recalculated.
    """
    # df_classifications = df_classifications_configs.copy()

    # Step 1: Ensure HLA column is included
    df_dqb = df_data[['DQB10602']].copy()
    df_dqb.reset_index(inplace=True, drop=False)

    df_classifications = df_classifications.merge(df_dqb, on='index', how='left')

    # Step 2: Apply the veto rule to correct misclassified observations
    df_classifications['predicted_hla_veto'] = df_classifications['predicted_label']

    # If classification is FP (true=0,pred=1) & HLA is False & DQB10602 == 0, apply veto rule
    # NT1 cases are always HLA positive
    mask_veto = (df_classifications['classification'].isin({'FP'})) & \
           (df_classifications['DQB10602'] == 0)
    # Optional, Step 1: Count how many observations per config and model we are applying the veto rule
    df_filtered = df_classifications[
        (df_classifications['classification'] == 'FP') &
        (df_classifications['DQB10602'] == 0)
        ]

    # number of observations per fold and how many are NT1
    df_counts = df_classifications.groupby(['config', 'model_name', 'fold', 'true_label', 'DQB10602']).size().reset_index(name='count')

    # Optional, Step 2: Count occurrences of FP per (config, model_name, fold)
    df_counts = df_filtered.groupby(['config', 'model_name', 'fold']).size().reset_index(name='count')

    # Optional, Step 3: Pivot the table to match the required format
    df_pivot = df_counts.pivot(index='model_name', columns=['config', 'fold'], values='count').fillna(0)
    print(df_pivot)

    # we are setting them to zero as prediction
    df_classifications.loc[mask_veto, 'predicted_hla_veto'] = df_classifications.loc[mask_veto, 'predicted_hla_veto'] - 1

    # Step 3: re-compute the metrics and obtain the averages across the folds
    metrics_records = []
    for config_model in df_classifications.config.unique():
        for model_name in df_classifications.model_name.unique():
            # model_name = df_classifications.model_name.unique()[0]
            for val_fold_num in df_classifications.fold.unique():
                # val_fold_num = df_classifications.fold.unique()[0]
                mask = (df_classifications['model_name'] == model_name) & \
                       (df_classifications['fold'] == val_fold_num) & \
                       (df_classifications['config'] == config_model)
                y_true = df_classifications.loc[mask, 'true_label']
                y_pred = df_classifications.loc[mask, 'predicted_hla_veto']  # predicted_hla_veto
                metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
                metrics.update({'fold': val_fold_num + 1,
                                'model': model_name,
                                'config': config_model})
                metrics_records.append(metrics)

    df_agg_metrics = pd.DataFrame(metrics_records)
    df_agg_metrics = df_agg_metrics[
        ['model', 'config', 'fold'] + [col for col in df_agg_metrics.columns if col not in ['model', 'fold', 'config']]]

    # Step 4: Compute confidence intervals for sensitivity and specificity
    df_ci = []
    metric_ci = ['sensitivity', 'specificity', 'ppv', 'npv']
    for (config_model, model_name), group in df_agg_metrics.groupby(['config', 'model']):
        ci_dict = {'config': config_model, 'model': model_name}
        for metric in metric_ci:
            values = group[metric].values
            ci_dict[f'{metric}_ci'] = compute_confidence_interval(values)
        df_ci.append(ci_dict)

    df_ci = pd.DataFrame(df_ci)

    # Step 5: Compute the average across the measures for each config + model
    df_avg_metrics = df_agg_metrics.groupby(['config', 'model'], as_index=False).mean(numeric_only=True)
    df_avg_metrics.drop(columns='fold', inplace=True)

    df_avg_metrics = df_avg_metrics.sort_values(by='specificity', ascending=False)

    # Step 6: Merge confidence intervals into the final metrics dataframe
    df_avg_metrics = pd.merge(left=df_avg_metrics, right=df_ci, on=['config', 'model'])


    return df_classifications, df_avg_metrics


def decision_curve_analysis(models,
                            thresholds: Optional[np.ndarray] = None,
                            figsize: Optional[Tuple] = (8, 6)):
    """
    Performs Decision Curve Analysis for multiple models.

    Parameters:
    models (dict): Dictionary where keys are model names and values are tuples (y_true, y_pred_proba).
                   - y_true: Array of true binary labels (0 or 1).
                   - y_pred_proba: Array of predicted probabilities for the positive class.
    thresholds (array, optional): Array of probability thresholds to evaluate. Defaults to np.linspace(0.01, 0.99, 100).
    figsize (tuple, optional): Figure size. Defaults to (8,6).

    Returns:
    Plots the decision curve analysis for the models.
    """

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    plt.figure(figsize=figsize)

    # Calculate Treat-All and Treat-None baselines

    for model_name, (y_true, y_pred_proba) in models.items():
        net_benefit = np.zeros_like(thresholds)

        prevalence = np.mean(y_true)
        treat_none_net_benefit = np.zeros_like(thresholds)  # Treat-none has 0 net benefit
        treat_all_net_benefit = prevalence - ((thresholds / (1 - thresholds)) * (1 - prevalence))  # Treat-all formula


        for i, threshold in enumerate(thresholds):
            # Generate binary predictions based on threshold
            predictions = (y_pred_proba >= threshold).astype(int)

            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

            # Compute net benefit
            n = len(y_true)
            w = threshold / (1 - threshold)  # Relative harm ratio
            benefit = tp / n
            harm = (fp / n) * w
            net_benefit[i] = benefit - harm

        # Plot net benefit curve for the model
        plt.plot(thresholds, net_benefit, label=f'Net Benefit ({model_name})', linewidth=2)

    # Plot baselines
    plt.plot(thresholds, treat_all_net_benefit, label='Treat All', linestyle='--', color='red')
    plt.plot(thresholds, treat_none_net_benefit, label='Treat None', linestyle='--', color='green')

    # Formatting and labels
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#
# def decision_curve_analysis(models,
#                             thresholds:Optional[np.ndarray]=None,
#                             figsize:Optional[Tuple]=(8,6)):
#     """
#     Performs Decision Curve Analysis for multiple models.
#
#     Parameters:
#     models (dict): Dictionary where keys are model names and values are tuples (y_true, y_pred_proba).
#                    - y_true: Array of true binary labels (0 or 1).
#                    - y_pred_proba: Array of predicted probabilities for the positive class.
#     thresholds (array): Array of probability thresholds to evaluate.
#
#     Returns:
#     Plots the decision curve analysis for the models.
#     """
#     if thresholds is None:
#         thresholds = np.linspace(0.01, 0.99, 100)
#
#     plt.figure(figsize=figsize)
#     for model_name, (y_true, y_pred_proba) in models.items():
#         net_benefit = []
#         net_beneit_treat_all = []
#         treat_none_net_benefit = np.zeros_like(thresholds)
#
#         for threshold in thresholds:
#             # Generate binary predictions based on threshold
#             predictions = (y_pred_proba >= threshold).astype(int)
#
#             # Compute confusion matrix
#             tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
#
#             # Compute net benefit
#             n = len(y_true)
#             w = threshold / (1 - threshold)  # Relative harm ratio
#             benefit = tp / n
#             harm = (fp / n) * w
#             net_benefit.append(benefit - harm)
#
#             # net benefit treat all
#             prevalence = np.mean(y_true)
#             print(f'prev: {prevalence}\n treh: {threshold}')
#             value = prevalence - ((threshold / (1 - threshold)) * (1 - prevalence))
#             net_beneit_treat_all.append(value)
#             # net benefit no treat
#
#         # Plot net benefit curve for the model
#         plt.plot(thresholds, net_benefit, label=f'Net Benefit ({model_name})', linewidth=2)
#         plt.plot(thresholds,
#                  net_beneit_treat_all,
#                  label='Treat All',
#                  linestyle='--',
#                  color='red')
#         plt.plot(thresholds,
#                  treat_none_net_benefit,
#                  label='Treat None',
#                  linestyle='--',
#                  color='green')
#         plt.show()
#
#     # # Compute Treat-All and Treat-None baselines
#     # y_true_example = next(iter(models.values()))[0]  # Extract y_true from the first model
#     # treat_all_net_benefit = np.mean(y_true_example) - thresholds * (1 - np.mean(y_true_example))
#
#     plt.plot(thresholds,
#              treat_none_net_benefit,
#              label='Treat None',
#              linestyle='--',
#              color='green')
#     plt.xlabel('Threshold Probability')
#     plt.ylabel('Net Benefit')
#     plt.title('Decision Curve Analysis')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


def calculate_ppv(SE, SP, prevalence):
    """
    Calculate the positive predictive values for a given sensitivity and specificity score.
    :param SE:
    :param SP:
    :param prevalence:
    :return:
    """
    PPV = (SE * prevalence) / ((SE * prevalence) + ((1 - SP) * (1 - prevalence)))
    return np.round(PPV, 3)


def recompute_classification(df):
    """
    Recomputes the 'classification' column based on 'true_label' and 'predicted_hla_veto'.

    Parameters:
        df (pd.DataFrame): The DataFrame containing 'true_label' and 'predicted_hla_veto'.

    Returns:
        pd.DataFrame: Updated DataFrame with the corrected 'classification' column.
    """
    conditions = [
        (df['true_label'] == 1) & (df['predicted_hla_veto'] == 1),
        (df['true_label'] == 0) & (df['predicted_hla_veto'] == 1),
        (df['true_label'] == 0) & (df['predicted_hla_veto'] == 0),
        (df['true_label'] == 1) & (df['predicted_hla_veto'] == 0)
    ]
    classifications = ['TP', 'FP', 'TN', 'FN']

    df['classification'] = np.select(conditions, classifications, default='Unknown')

    return df


def extract_metrics(df):
    """Extracts classification counts and ensures all categories exist."""
    metrics = df.groupby(['model_name', 'fold'])['classification'].value_counts().unstack(fill_value=0)
    metrics.reset_index(inplace=True, drop=True)
    for col in ['FN', 'FP', 'TN', 'TP']:
        if col not in metrics.columns:
            metrics[col] = 0
    return metrics
