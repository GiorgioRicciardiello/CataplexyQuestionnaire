"""
==============================================================================
Script: Optimize fp_weight Hyperparameter for XGBoost (Training Fold Evaluation Only)
------------------------------------------------------------------------------

Description:
    This script implements k-fold cross validation to optimize the 'fp_weight'
    hyperparameter used in a custom XGBoost loss function. In this evaluation,
    model performance (specificity and sensitivity) is computed using only the
    training folds. The validation fold is not used for evaluating performance.

    The custom loss function applies a penalty on false positives, controlled
    by the 'fp_weight' parameter. By varying this parameter, the script identifies
    the value that maximizes specificity on the training data.

Usage:
    - Adjust the 'weight_candidates' list as needed.
    - Run the script to visualize how 'fp_weight' impacts both specificity and sensitivity,
      and to determine the optimal value based solely on the training fold.

Author: Giorgio Ricciardiello
Date: 2024
==============================================================================
"""


import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from scipy.optimize import minimize_scalar
from config.config import config
from typing import List
import matplotlib.pyplot as plt

def train_xgboost(train_data: pd.DataFrame,
                  train_labels: np.ndarray,
                  val_data: pd.DataFrame,
                  val_labels: np.ndarray,
                  fp_weight: float = 3.2):
    """
    Train an XGBoost model using a custom loss function that incorporates a false positive penalty.

    Parameters:
    - train_data, train_labels: Training set.
    - val_data, val_labels: Validation set.
    - fp_weight: Weight to penalize false positives.

    Returns:
    - y_pred_val: Binary predictions for the validation set.
    - y_pred_prob_val: Predicted probabilities for the validation set.
    - y_pred_train: Binary predictions for the training set.
    - y_pred_prob_train: Predicted probabilities for the training set.
    """

    def specificity_loss(preds, dtrain):
        labels = dtrain.get_label()
        preds = 1 / (1 + np.exp(-preds))  # Convert logits to probabilities
        # Use fp_weight to penalize false positives
        grad = -labels * (1 - preds) + (1 - labels) * preds * fp_weight
        hess = preds * (1 - preds) * (1 + (1 - labels))
        return grad, hess

    def specificity_eval_metric(preds, dtrain):
        labels = dtrain.get_label()
        preds = (preds > 0.5).astype(int)  # Threshold at 0.5
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return 'specificity', specificity

    def find_best_threshold_for_predictions(y_true_train: np.ndarray,
                                            y_pred_train: np.ndarray,
                                            metric: str = 'specificity') -> float:
        # Function to find the threshold that maximizes the desired metric
        def metric_for_threshold(threshold):
            y_pred_thresh = (y_pred_train >= threshold).astype(int)
            if metric == 'f1':
                return -f1_score(y_true_train, y_pred_thresh)
            elif metric == 'accuracy':
                return -accuracy_score(y_true_train, y_pred_thresh)
            elif metric == 'sensitivity':
                return -recall_score(y_true_train, y_pred_thresh)
            elif metric == 'specificity':
                tn, fp, fn, tp = confusion_matrix(y_true_train, y_pred_thresh).ravel()
                specificity = tn / (tn + fp)
                return -specificity
            elif metric == 'auc':
                return -roc_auc_score(y_true_train, y_pred_thresh)
            else:
                raise ValueError("Unsupported metric.")

        result = minimize_scalar(metric_for_threshold, bounds=(0.0, 1.0), method='bounded')
        return result.x

    # XGBoost parameters remain mostly fixed
    params = {
        'scale_pos_weight': 16,
        'max_depth': 12,
        'reg_lambda': 0.001,
        'gamma': 0.2,
        'reg_alpha': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }

    dtrain = xgb.DMatrix(train_data, label=train_labels)
    dval = xgb.DMatrix(val_data, label=val_labels)

    model = xgb.train(params,
                      dtrain,
                      num_boost_round=2,
                      custom_metric=specificity_eval_metric,
                      obj=specificity_loss,
                      evals=[(dtrain, 'train'), (dval, 'valid')],
                      verbose_eval=False)

    best_threshold = find_best_threshold_for_predictions(train_labels, model.predict(dtrain), metric='specificity')

    y_pred_val = (model.predict(dval) > best_threshold).astype(int)
    y_pred_prob_val = model.predict(dval)
    y_pred_train = (model.predict(dtrain) > best_threshold).astype(int)
    y_pred_prob_train = model.predict(dtrain)

    return y_pred_val, y_pred_prob_val, y_pred_train, y_pred_prob_train


def optimize_fp_weight(train_data: pd.DataFrame,
                       train_labels: np.ndarray,
                       k: int = 5,
                       weight_candidates:List[float]=None):
    """
    Optimize the fp_weight hyperparameter via k‑fold cross validation. We only use the training folds to find the best
    weight.

    Parameters:
    - train_data, train_labels: The full training dataset.
    - k: Number of folds for cross validation.
    - weight_candidates: List of fp_weight values to try.

    Returns:
    - best_weight: The fp_weight value that achieved the highest mean specificity.
    - best_cv_score: The highest mean specificity observed.
    """
    if weight_candidates is None:
        weight_candidates = [1.0, 2.0, 3.2, 4.0, 5.0,6,7,8,9,10]
    best_weight = None
    best_cv_score = -np.inf
    specificity_scores = []
    sensitivity_scores = []
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for weight in weight_candidates:
        cv_specificity = []
        cv_sensitivity = []
        for train_idx, val_idx in kf.split(train_data):
            X_train, X_val = train_data.iloc[train_idx], train_data.iloc[val_idx]
            y_train, y_val = train_labels[train_idx], train_labels[val_idx]

            # Train the model using the current fp_weight candidate
            y_pred_val, _, y_pred_train, _ = train_xgboost(X_train, y_train, X_val, y_val, fp_weight=weight)

            # Compute specificity on the validation fold
            tn = np.sum((y_train == 0) & (y_pred_train == 0))
            fp = np.sum((y_train == 0) & (y_pred_train == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            cv_specificity.append(specificity)

            # Compute sensitivity on the validation fold
            tp = np.sum((y_train == 1) & (y_pred_train == 1))
            fn = np.sum((y_train == 1) & (y_pred_train == 0))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            cv_sensitivity.append(sensitivity)

        mean_specificity = np.mean(cv_specificity)
        mean_sensitivity = np.mean(cv_sensitivity)

        specificity_scores.append(mean_specificity)
        sensitivity_scores.append(mean_sensitivity)

        print(f"fp_weight: {weight}, Mean Specificity: {mean_specificity:.4f}")

        if mean_specificity > best_cv_score:
            best_cv_score = mean_specificity
            best_weight = weight

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(weight_candidates,
             specificity_scores,
             marker='o',
             label='Specificity')
    plt.plot(weight_candidates,
             sensitivity_scores,
             marker='o',
             label='Sensitivity')
    plt.xlabel('FP Weight')
    plt.ylabel('Metric Value')
    plt.title('Specificity and Sensitivity vs. FP Weight')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    return best_weight, specificity_scores, sensitivity_scores


# Example usage:
if __name__ == "__main__":
    # %% Read data
    df_data = pd.read_csv(config.get('data_pre_proc_files').get('anic_okun'))

    # %% Select columns and drop columns with nans
    target = 'NT1 ICSD3 - TR'
    target_nt2 = target.replace('1', '2')

    categorical_var = ['sex', 'LAUGHING', 'ANGER', 'EXCITED',
                       'SURPRISED', 'HAPPY', 'EMOTIONAL', 'QUICKVERBAL', 'EMBARRAS',
                       'DISCIPLINE', 'SEX', 'DURATHLETIC', 'AFTATHLETIC', 'ELATED',
                       'STRESSED', 'STARTLED', 'TENSE', 'PLAYGAME', 'ROMANTIC',
                       'JOKING', 'MOVEDEMOT', 'KNEES', 'JAW', 'HEAD', 'HAND', 'SPEECH',
                       'DQB10602']
    continuous_var = ['Age', 'BMI', 'ESS', 'SLEEPIONSET']
    columns = list(set(categorical_var + continuous_var + [target]))
    # remove Age and Sex so the questionnaire is inclusive
    columns = [col for col in columns if not col in ['sex', 'Age']]
    df_data = df_data.loc[:, columns]
    # df_data = df_data.dropna(axis=1)
    cols_with_many_nans = df_data.columns[df_data.isna().sum() > 15]
    df_data.drop(cols_with_many_nans, axis=1, inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    df_data = df_data.reindex(sorted(df_data.columns), axis=1)
    print(f'Dataset dimension: {df_data.shape}')
    # %% data splits
    train_data = df_data[[col for col in df_data.columns if col != target]]
    train_labels = df_data[target]

    #%% Optimize the fp_weight hyperparameter
    weight_candidates =  [1.0, 2.0, 3.2, 4.0, 5.0,6,7,8,9,10]
    best_weight, best_score = optimize_fp_weight(train_data,
                                                 train_labels,
                                                 k=5,
                                                 weight_candidates=weight_candidates)
    print(f"\nOptimal fp_weight: {best_weight} with mean specificity: {best_score:.4f}")
