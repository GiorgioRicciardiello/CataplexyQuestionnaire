"""
Find the best cut off for the dataset to distinguish between cases and controls
"""
from config.config import config
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, classification_report, RocCurveDisplay, precision_recall_curve, auc, PrecisionRecallDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from library.effect_measures_plot import EffectMeasurePlot
from typing import Union, Optional
import pathlib
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score
from library.ml_models import okun_decision


def calculate_metrics(y_true, y_pred, output_dict: Optional[str] = False):
    """Calculate sensitivity, specificity, PPV, and NPV."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    if output_dict:
        return {'sensitivity': sensitivity, 'specificity': specificity, 'ppv': ppv, 'npv': npv}
    return sensitivity, specificity, ppv, npv


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


def plot_stratified_distribution(
        splits_target: dict[str, Union[pd.DataFrame, pd.Series]],
        output_path: Union[pathlib.Path, None] = None,
        show_plot: Optional[bool] = True,
        save_plot: Optional[bool] = False,
):
    """
    Plot the stratified target (categorical/ordinal) as a bar plot. The  x axis contains the train, validation, and
    test split. Each x-ticks has the bar of the count of each class in the split
    :return:
    """
    # splits_target = {key: item for key, item in self.splits.items() if 'y' in key}
    splits_count = {}
    for lbl_, split_ in splits_target.items():
        splits_count[lbl_] = split_.value_counts().to_dict()
    # Sorting each inner dictionary by its keys
    splits_count = {outer_k: dict(sorted(outer_v.items())) for outer_k, outer_v in splits_count.items()}

    df_splits_count = pd.DataFrame(splits_count).reset_index(names=['index'])
    df_melted = df_splits_count.melt(id_vars='index',
                                     var_name='split',
                                     value_name='count')

    df_melted.rename(columns={'index': 'class'},
                     inplace=True)

    # Now we can create a seaborn barplot with splits on the x-axis and count on the y-axis
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted,
                x='split',
                y='count',
                hue='class'
                )
    plt.title('Counts of Classes across Different Splits')
    plt.xlabel('Split')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(0.7)
    if save_plot and output_path is not None:
        plt.savefig(output_path, dpi=300)
    if show_plot:
        plt.show()


def create_splits_under_sampler(df: pd.DataFrame,
                                features: list[str],
                                target: list[str],
                                output_path: pathlib.Path,
                                stratify: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Create the train and test validation splits.
    Use random under samples (remove samples from mayority class) to keep a class balance.
    :param df:
    :param features:
    :param target:
    :param output_path:
    :param stratify:
    :return:
        Train and Test features dataframe and target series
    """
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df[stratify],
    )

    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    sampler = RandomUnderSampler(random_state=42,
                                 sampling_strategy='not minority',
                                 )
    plot_stratified_distribution(splits_target={'train': y_train, 'test': y_test},
                                 output_path=output_path.joinpath('TargetDistributionStandard.png'),
                                 save_plot=True,
                                 show_plot=True)
    X_train, y_train = sampler.fit_resample(X_train, y_train)

    plot_stratified_distribution(splits_target={'train': y_train, 'test': y_test},
                                 output_path=output_path.joinpath('TargetDistributionUnderSampled.png'),
                                 save_plot=True,
                                 show_plot=True)

    return X_train, y_train, X_test, y_test


def evaluate_predictions_generate_report(
                                         y_true:pd.Series,
                                        y_prob:pd.Series,
                                         threshold:float=0.5,
                                         output_path:Optional[pathlib.Path] = None,
                                         title:Optional[str] = '',
                                         model_name:Optional[str] = '') -> pd.DataFrame:
    """
    Create the predictions (probabilities) of the classes and generate a classification report for the given
    split.
    Generates the ARUOC with varying the threshold parameter.

    Classification report are roc curve are saved in the output make with the model name as leading name.

    :param y_true: pd.Series, Series of the true class Y_train, Y_test, ...
    :param threshold: float, threshold value to mark predictions as the positive class or negative class
    :return:
    """
    model_name = model_name.replace('_', '').capitalize()
    # make predictions based on the given threshold
    # y_prob = result.predict(X)
    y_pred = (y_prob >= threshold).astype(int)  # get binary predictions from the probabilies

    # generate the classification report and include the metrics and format we desire
    report = classification_report(y_true=y_true,
                                   y_pred=y_pred,
                                   output_dict=True)
    report = pd.DataFrame(report).transpose()
    sensitivity, specificity, ppv, npv = calculate_metrics(y_true=y_true,
                                                           y_pred=y_pred,
                                                           output_dict=False)

    report['global'] = np.nan
    report.loc['accuracy', 'global'] = report.loc['accuracy', :].unique()[0]

    report.loc['accuracy', ['precision', 'recall', 'f1-score', 'support']] = np.nan

    report.loc['macro avg', 'global'] = report.loc['macro avg', :].unique()[0]
    report.loc['macro avg', ['precision', 'recall', 'f1-score', 'support']] = np.nan

    report.loc['weighted avg', 'global'] = report.loc['weighted avg', :].unique()[0]
    report.loc['weighted avg', ['precision', 'recall', 'f1-score', 'support']] = np.nan

    report.loc['ppv', 'global'] = ppv
    report.loc['npv', 'global'] = npv
    report = report.round(3)
    print(report)
    report.to_csv(output_path.joinpath(f'{model_name}_ClassificationReport_{title}.csv'), index=True)
    # ROC curve with varying threshold
    if any([True for val in y_prob if isinstance(val, float)]):
        # it can only be done if we are given probabilities not the classes values
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        # Plotting the ROC curve
        display = RocCurveDisplay(fpr=fpr,
                                  tpr=tpr,
                                  roc_auc=roc_auc,
                                  estimator_name='example estimator')
        display.plot()
        # Find indices of thresholds near desired values
        desired_thresholds = [0, 0.15, 0.25, 0.5, 0.75, 0.85, 1]
        indices = [np.abs(thresholds - value).argmin() for value in desired_thresholds]
        # Plotting selected thresholds and adding text
        threshold_values = []
        for index, (x, y) in zip(indices, zip(fpr[indices], tpr[indices])):
            threshold_value = thresholds[index]
            plt.scatter(x, y, color='red')
            plt.text(x, y,
                     s=f'  {threshold_value:.2f}',
                     fontsize=11,
                     ha='left',
                     va='bottom')
            threshold_values.append(threshold_value)
        # Adding legend for the dots
        plt.scatter(x=[],
                    y=[],
                    color='red',
                    label='Selected Thresholds')
        plt.legend()
        # Setting grid and layout
        plt.grid(alpha=0.7)
        plt.title('Varying Threshold on the Probabilities')
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path.joinpath(f'{model_name}_RocVaryingThresh_{title}'), dpi=300)
        plt.show()

        # Precision-recall curve with varying threshold
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        auprc = auc(recall, precision)

        # Plotting the Precision-Recall curve
        display = PrecisionRecallDisplay(precision=precision,
                                         recall=recall,
                                         average_precision=auprc,
                                         estimator_name='example estimator')
        display.plot()

        # Find indices of thresholds near desired values
        desired_thresholds = [0, 0.15, 0.25, 0.5, 0.75, 0.85, 1]
        indices = [np.abs(thresholds - value).argmin() for value in desired_thresholds]

        # Plotting selected thresholds and adding text
        threshold_values = []
        for index, (x, y) in zip(indices, zip(recall[indices], precision[indices])):
            threshold_value = thresholds[index]
            plt.scatter(x, y, color='red')
            plt.text(x, y,
                     s=f'  {threshold_value:.2f}',
                     fontsize=11,
                     ha='left',
                     va='bottom')
            threshold_values.append(threshold_value)

        # Adding legend for the dots
        plt.scatter(x=[],
                    y=[],
                    color='red',
                    label='Selected Thresholds')
        plt.legend()

        # Setting grid and layout
        plt.grid(alpha=0.7)
        plt.title('Varying Threshold on the Probabilities')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path.joinpath(f'{model_name}_PRCVaryingThresh_{title}'), dpi=300)
        plt.show()
    return report

def compute_odds_ratios(stats_summary_report,
                        output_path:pathlib.Path,
                        title: Optional[str] = '',
                        model_name: Optional[str] = ''
                        ):
    """
    Compute the odds ratios from the summary report obtained from the stats models. Save the odds ratio plot
    :param stats_summary_report:
    :param output_path:
    :param title:
    :param model_name:
    :return:
    """
    model_name = model_name.replace('_', '').capitalize()
    # Get the summary table
    summary = stats_summary_report.summary2().tables[1]
    # summary.reset_index(inplace=True, drop=False, names='labels')
    # Calculate Odds Ratios (OR)
    summary['OR'] = summary['Coef.'].apply(np.exp)
    # Calculate lower and upper confidence bounds for OR
    summary['ci_low_bound'] = np.exp(summary['Coef.'] - 1.96 * summary['Std.Err.'])
    summary['ci_high_bound'] = np.exp(summary['Coef.'] + 1.96 * summary['Std.Err.'])
    # Extract relevant columns
    df_table = summary[['OR', 'ci_low_bound', 'ci_high_bound', 'P>|z|']]
    df_table = df_table.reset_index().rename(columns={'index': 'variable', 'P>|z|': 'p-value'})
    df_table = df_table[df_table['variable'] != 'const']
    # Create the final DataFrame
    results_df = pd.DataFrame({
        'label': df_table['variable'].tolist(),
        'effect_measure': df_table['OR'].tolist(),
        'lcl': df_table['ci_low_bound'].tolist(),
        'ucl': df_table['ci_high_bound'].tolist(),
        'p_value': df_table['p-value'].tolist()
    })
    forest_plot = EffectMeasurePlot(label=df_table.variable.tolist(),
                                    effect_measure=df_table.OR.tolist(),
                                    lcl=df_table.ci_low_bound.tolist(),
                                    ucl=df_table.ci_high_bound.tolist(),
                                    p_value=df_table['p-value'].to_list())
    forest_plot.plot(figsize=(10, 6),
                     max_value=np.round(results_df['ucl'].max() + 0.3, 3),
                     min_value=np.round(results_df['ucl'].min() - 0.3, 3),
                     path_save=output_path.joinpath(f'{model_name}_OddsRatio_{title}.png'),
                     t_adjuster=0.01
                     )
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # df_data = pd.read_csv(config.get('data_pre_proc_files').get('ess_control_all'))
    df_data = pd.read_csv(config.get('data_pre_proc_files').get('anic_okun'))


    result_path = config.get('results_path').get('ess_model')
    result_path.mkdir(parents=True, exist_ok=True)

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

    unique_targets = df_data[target].unique()
    palette = sns.color_palette("viridis", len(unique_targets))
    color_map = dict(zip(unique_targets, palette))
    df_data['palette'] = df_data[target].map(color_map)

    # %% visualize the distribution
    df_data.sort_values(by='ESS', inplace=True)
    group_means = df_data.groupby(by=target)['ESS'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df_data,
                 y='ESS',
                 hue=target,
                 palette=color_map,
                 multiple='stack',
                 ax=ax)

    # Draw horizontal lines at the mean of each group
    for target_value, mean in group_means.items():
        ax.axhline(mean, color=color_map[target_value], linestyle='--', linewidth=2,
                   label=f'Mean {mean:.4} (target={target_value})')

    # Final plot adjustments
    ax.set_xlabel('Count')
    ax.set_ylabel('ESS Score')
    ax.set_title('Distribution of ESS Score by Diagnosis Status\n'
                 f'NT1: {df_data[df_data[target] == "NT1"].shape[0]}\n'
                 f'Controls: {df_data[df_data[target] == "Control"].shape[0]}')
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.7)
    plt.savefig(result_path.joinpath('Distribution_EssControlsCases.png'), dpi=300)
    plt.show()

    # we can only control for age and gener
    df_model = df_data.dropna(subset=['ESS'])[['ESS', target]].copy()
    df_model['ESS'] = df_model['ESS'].astype(int)
    df_model.reset_index(inplace=True)
    print(f'Dimensions of dataset to model: {df_model.shape}')

    # %% simple threshold on the ess score
    # Apply a moving threshold to the ESS scores and compute the Sen/Spec in the dataset
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    # --- Step 1: Compute predictions and metrics by ESS threshold ---
    thresholds = np.sort(np.arange(df_model['ESS'].max(), df_model['ESS'].min() - 1, -1))
    predictions = {}

    for thresh_ in thresholds:
        df_model['predicted'] = (df_model['ESS'] >= thresh_).astype(int)
        sensitivity, specificity, ppv, npv = calculate_metrics(df_model[target], df_model['predicted'])
        youden = sensitivity + specificity - 1
        predictions[thresh_] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'youden': youden
        }

    predictions_df = pd.DataFrame.from_dict(predictions, orient='index')

    # --- Step 2: Identify key thresholds ---
    predictions_df['difference'] = abs(predictions_df['sensitivity'] - predictions_df['specificity'])
    intersection_threshold = predictions_df['difference'].idxmin()
    youden_threshold = predictions_df['youden'].idxmax()

    # Get ROC coordinates for those thresholds
    y_true = df_model[target]
    y_scores = df_model['ESS']
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Find closest threshold on ROC to Youden and Intersection ESS
    youden_idx = np.argmin(np.abs(roc_thresholds - youden_threshold))
    intersection_idx = np.argmin(np.abs(roc_thresholds - intersection_threshold))

    # --- Step 3: Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # -- Subplot 1: Sensitivity/Specificity curves --
    axes[0].plot(predictions_df.index, predictions_df['sensitivity'], label='Sensitivity')
    axes[0].plot(predictions_df.index, predictions_df['specificity'], label='Specificity')

    axes[0].scatter(intersection_threshold, predictions_df.loc[intersection_threshold, 'sensitivity'], color='red',
                    label='Intersection Point')
    axes[0].axvline(intersection_threshold, color='red', linestyle='--', linewidth=1)

    axes[0].scatter(youden_threshold, predictions_df.loc[youden_threshold, 'sensitivity'], color='green',
                    label='Youden Index')
    axes[0].axvline(youden_threshold, color='green', linestyle='--', linewidth=1)

    axes[0].set_title('Sensitivity & Specificity by ESS Threshold')
    axes[0].set_xlabel('ESS Score Threshold')
    axes[0].set_ylabel('Metric Value')
    axes[0].set_xticks(thresholds)
    axes[0].set_yticks(np.arange(0, 1.1, 0.1))
    axes[0].grid(alpha=0.5)
    axes[0].legend()

    # -- Subplot 2: ROC curve --
    axes[1].plot(fpr, tpr, label=f'ESS ROC (AUC = {roc_auc*100:.2f}%)', color='darkviolet')
    # axes[1].plot([0, 1], [0, 1], 'k--', label='Chance')

    # Plot Youden index and intersection points on ROC
    axes[1].scatter(fpr[youden_idx], tpr[youden_idx], color='green', label='Youden Point', zorder=5)
    axes[1].scatter(fpr[intersection_idx], tpr[intersection_idx], color='red', label='Intersection Point', zorder=5)

    axes[1].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[1].set_ylabel('True Positive Rate (Sensitivity)')
    axes[1].set_title('ROC Curve — ESS Alone')
    axes[1].grid(alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(result_path.joinpath("ESS_Thresholds_and_Annotated_ROC.png"), dpi=300)
    plt.show()

    # Sensitivity and Specificity at Intersection Point
    intersection_sens = predictions_df.loc[intersection_threshold, 'sensitivity']
    intersection_spec = predictions_df.loc[intersection_threshold, 'specificity']

    print(f"Intersection Threshold = {intersection_threshold}")
    print(f"  Sensitivity: {intersection_sens:.3f}")
    print(f"  Specificity: {intersection_spec:.3f}")

    # Sensitivity and Specificity at Youden Index
    youden_sens = predictions_df.loc[youden_threshold, 'sensitivity']
    youden_spec = predictions_df.loc[youden_threshold, 'specificity']

    print(f"\nYouden Index Threshold = {youden_threshold}")
    print(f"  Sensitivity: {youden_sens:.3f}")
    print(f"  Specificity: {youden_spec:.3f}")

    # %% PPV
    PLOT_PPV = False
    if PLOT_PPV:
        predictions_df['difference'] = np.abs(predictions_df['ppv'] - predictions_df['npv'])
        intersection_threshold = predictions_df['difference'].idxmin()
        intersection_ppv = predictions_df.loc[intersection_threshold, 'ppv']
        intersection_npv = predictions_df.loc[intersection_threshold, 'npv']

        plt.figure(figsize=(10, 6))
        plt.plot(predictions_df.index, predictions_df['ppv'], label='ppv')
        plt.plot(predictions_df.index, predictions_df['npv'], label='npv')
        plt.scatter(intersection_threshold, intersection_ppv, color='red', zorder=5)
        plt.xlabel('ESS Score Threshold')
        plt.ylabel('Metric Value')
        plt.title(f'Positive Predictive Value (PPV) and Negative Predictive Value (NPV) by ESS Score Threshold\n'
                  f'Best Threshold = {intersection_threshold}\n'
                  f'NT1 = {df_model.loc[df_model[target] == 1].shape[0]} - '
                  f'Controls = {df_model.loc[df_model[target] == 0,].shape[0]}')
        plt.grid(alpha=0.7)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks(thresholds)
        plt.xlim([np.min(thresholds), np.max(thresholds)])
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_path.joinpath('Intersection_PpvNpv.png'), dpi=300)
        plt.show()


    # %% Okun decision tree
    from sklearn.metrics import roc_auc_score

    # --- 1. Run Okun Decision Tree ---
    # Example: columns_tree = ['JOKING', 'LAUGHING', 'ANGER']
    okun_preds = okun_decision(df_data, columns_tree=['JOKING', 'LAUGHING', 'ANGER'])

    # --- 2. Compute Metrics ---
    okun_sens, okun_spec, okun_ppv, okun_npv = calculate_metrics(df_data[target], okun_preds)

    # AUC requires probabilities or continuous predictions; here we just use binary predictions
    # This is valid but results in a conservative (lower bound) AUC
    okun_auc = roc_auc_score(df_data[target], okun_preds)

    # --- 3. Print Results ---
    print("Okun Rule-Based Model Performance:")
    print(f"  Sensitivity: {okun_sens*100:.2f}")
    print(f"  Specificity: {okun_spec*100:.2f}")
    print(f"  AUC:         {okun_auc*100:.2f}")


    # %% Both images in same figure
    # Sorting data for the histogram
    df_data.sort_values(by='ESS', inplace=True)
    group_means = df_data.groupby(by=target)['ESS'].mean()

    # Creating a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns

    # First subplot: Histogram of ESS scores
    sns.histplot(data=df_data,
                 x='ESS',
                 hue=target,
                 palette=color_map,
                 multiple='stack',
                 ax=axes[0])

    # Draw horizontal lines at the mean of each group
    for target_value, mean in group_means.items():
        axes[0].axvline(mean, color=color_map[target_value], linestyle='--', linewidth=2,
                        label=f'Mean {mean:.4} (target={target_value})')

    # Final plot adjustments for first plot
    axes[0].set_xlabel('Count')
    axes[0].set_ylabel('ESS Score')
    axes[0].set_title('Distribution of ESS Score by Diagnosis Status\n'
                      f'NT1: {df_data[df_data[target] == "NT1"].shape[0]}\n'
                      f'Controls: {df_data[df_data[target] == "Control"].shape[0]}')
    axes[0].legend()
    axes[0].grid(alpha=0.7)
    axes[0].set_xlim([df_data.ESS.min(), df_data.ESS.max()])

    # Second subplot:
    predictions_df['difference'] = np.abs(predictions_df['sensitivity'] - predictions_df['specificity'])
    intersection_threshold = predictions_df['difference'].idxmin()
    intersection_sensitivity = predictions_df.loc[intersection_threshold, 'sensitivity']
    axes[1].plot(predictions_df.index, predictions_df['sensitivity'], label='Sensitivity')
    axes[1].plot(predictions_df.index, predictions_df['specificity'], label='Specificity')
    axes[1].scatter(intersection_threshold, intersection_sensitivity, color='red', zorder=5)
    axes[1].axvline(intersection_threshold, color='red', linestyle='--', linewidth=1.5)
    axes[1].axhline(intersection_sensitivity, color='red', linestyle='--', linewidth=1.5)

    # Final plot adjustments for second plot
    axes[1].set_xlabel('ESS Score Threshold')
    axes[1].set_ylabel('Metric Value')
    axes[1].set_title(f'Sensitivity and Specificity by ESS Score Threshold\n'
              f'Best Threshold = {intersection_threshold}\n'
              f'NT1 = {df_model.loc[df_model[target] == 1].shape[0]} - '
              f'Controls = {df_model.loc[df_model[target] == 0].shape[0]}')
    axes[1].grid(alpha=0.7)
    axes[1].set_yticks(np.arange(0, 1.1, 0.1))
    axes[1].set_xticks(thresholds)
    axes[1].set_xlim([np.min(thresholds), np.max(thresholds)])
    axes[1].legend()

    # Tight layout and saving the combined figure
    plt.tight_layout()
    # plt.savefig(result_path.joinpath('Combined_Distribution_Ess_PpvNpv.png'), dpi=300)
    plt.show()

    # %% Predictive Modelling
    col_features = ['sex', 'Age', 'ESS']
    col_target = [target]
    k = 5
    # %% Logistic Regression Base model
    from sklearn.model_selection import StratifiedKFold

    df_agg_metrics = pd.DataFrame()
    data = df_model[col_features]
    labels = df_model[col_target]
    skf = StratifiedKFold(n_splits=k,
                          shuffle=True,
                          random_state=42)
    fold_indices = list(skf.split(data, labels))
    for fold, (train_indices, val_indices) in enumerate(fold_indices):
        print(f"Starting Fold {fold + 1}/{k}")

        # Define train/validation splits
        train_data = data.loc[train_indices]
        train_labels = labels.loc[train_indices]
        val_data = data.loc[val_indices]
        val_labels = labels.loc[val_indices]
        # Fit logistic regression
        train_data_const = sm.add_constant(train_data)
        model = sm.Logit(train_labels, train_data_const).fit(disp=False, cov_type='HC1')

        beta = model.params
        std_err = model.bse
        odds_ratios = np.exp(beta)
        conf_int = model.conf_int(alpha=0.05).apply(np.exp)  # CI for odds ratios

        # Validation
        val_data_const = sm.add_constant(data.iloc[val_indices])
        preds_val = (model.predict(val_data_const) >= 0.5).astype(int)
        preds_train = (model.predict(train_data_const) >= 0.5).astype(int)

        # metrics for validation and train set
        def compute_metrics(y_pred:np.ndarray, y_true:np.ndarray):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)
            # Additional metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            return {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'accuracy': accuracy,
                'precision': precision,
                'f1_score': f1_score,
                'npv': npv,
                'fpr': fpr,
                'fnr': fnr
            }


        compute_metrics(y_pred= preds_val, y_true=val_labels.values.ravel())

    # now let's see if we can fit a model
    # outcome: diagnosis
    # independent variables: gender, age, ESS
    col_features = ['sex', 'Age', 'ESS']
    col_target = [target]

    X_train, y_train, X_test, y_test = create_splits_under_sampler(
        df=df_model,
        features=col_features,
        target=col_target,
        stratify=target,
        output_path=result_path
    )

    # Add a constant term to the train and test data
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Fit the logistic regression model
    # logit_model = sm.Logit(y_train, X_train)
    # logistic_model_fit = logit_model.fit()
    # y_prob = logistic_model_fit.predict(X_train)

    mnlogit_model = sm.MNLogit(y_train, X_train)
    mnlogit_model_fit = mnlogit_model.fit()
    y_prob = mnlogit_model_fit.predict(X_train)

    # Evaluate the model
    # Make predictions on the train data
    report_train = evaluate_predictions_generate_report(y_true=y_train,
                                                        y_prob=y_prob,
                                                        threshold=0.5,
                                                        output_path=result_path,
                                                        title='Train Set',
                                                        model_name='logistic_regression')

    report_test = evaluate_predictions_generate_report(y_true=y_test,
                                                       y_prob=y_prob,
                                                       threshold=0.5,
                                                        output_path=result_path,
                                                        title='Test Set',
                                                       model_name='logistic_regression')

    compute_odds_ratios(stats_summary_report=mnlogit_model_fit,
                        output_path=config.get('results_path'),
                        model_name='logistic_regression',
                        title='train')

    # %% Linear Discriminant Analysis (LDA) - All features
    X_train, y_train, X_test, y_test = create_splits_under_sampler(
        df=df_model,
        features=col_features,
        target=col_target,
        stratify='diagnosis',
        output_path=config.get('results_path')
    )
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(X_train, y_train)
    # min number of componets
    min(X_train.shape[1], y_test.nunique()[0] - 1)

    report_train_lda = evaluate_predictions_generate_report(
                                                        y_true=y_train,
                                                        y_prob=lda.predict(X_train),
                                                        threshold=0.5,
                                                        output_path=result_path,
                                                        title='Train Set',
                                                        model_name='LDA')

    report_test_lda = evaluate_predictions_generate_report(y_true=y_test,
                                                       y_prob=lda.predict(X_test),
                                                       threshold=0.5,
                                                        output_path=result_path,
                                                        title='Test Set',
                                                       model_name='LDA')

    # Transform the training data
    # https://machinelearningmastery.com/linear-discriminant-analysis-for-dimensionality-reduction-in-python/
    # transformed = lda.transform(X_train)
    # from sklearn.pipeline import Pipeline
    # from sklearn.naive_bayes import GaussianNB
    # steps = [('lda', LinearDiscriminantAnalysis()), ('m', GaussianNB())]
    # model = Pipeline(steps=steps)
    # model.fit(X_train, y_train)
    #
    # # Predict on the test data
    # y_pred = model.predict(X_test)
    #
    # # Evaluate the model
    # from sklearn.metrics import classification_report
    #
    # print(classification_report(y_test, y_pred))
    # %% Linear Discriminant Analysis (LDA) - Two dimenions
    # Since we have 3 features, we need to reduce it to 2 for visualization
    #  PCA + LDA
    pca = PCA(n_components=2)

    X_train, y_train, X_test, y_test = create_splits_under_sampler(
        df=df_model,
        features=col_features,
        target=col_target,
        stratify='diagnosis',
        output_path=config.get('results_path')
    )

    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    # Train the LDA model
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Predict and generate classification report
    y_test_pred = lda.predict(X_test)
    report = classification_report(y_test, y_test_pred)
    print(report)

    report_test_lda_pca = evaluate_predictions_generate_report(
                                                        y_true=y_test,
                                                        y_prob=y_test_pred,
                                                        threshold=0.5,
                                                        output_path=result_path_ess,
                                                        title='Test Set',
                                                        model_name='LDA-PCA')
    # Plot decision boundaries
    fig, ax = plt.subplots(figsize=(10, 6))
    DecisionBoundaryDisplay.from_estimator(
        lda, X_train,
        response_method="predict",
        ax=ax,
        cmap="coolwarm")

    # Scatter plot of the test data
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1],
                         c=y_test.values,
                         edgecolor='k',
                         cmap='coolwarm',
                         alpha=0.7)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Decision Boundary of LDA after PCA')
    legend = ax.legend(*scatter.legend_elements(), title="Diagnosis")
    ax.add_artist(legend)
    plt.tight_layout()
    plt.savefig(result_path_ess.joinpath('LDA_PCA_DecisionBoundary.png'), dpi=300)
    plt.show()




    # %% SVM
    # pca = PCA(n_components=2)
    # X_train, y_train, X_test, y_test = create_splits_under_sampler(
    #     df=df_model,
    #     features=col_features,
    #     target=col_target,
    #     stratify='diagnosis',
    #     output_path=config.get('results_path')
    # )
    #
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.fit_transform(X_test)
    #
    # # Train the SVM model
    # svm_model = SVC(kernel='linear')
    # svm_model.fit(X_train, y_train)
    #
    # # Predict and generate classification report
    # y_test_pred = svm_model.predict(X_test)
    # report = classification_report(y_test, y_test_pred)
    # print(report)
    #
    # # Plot decision boundaries
    # fig, ax = plt.subplots(figsize=(10, 6))
    # DecisionBoundaryDisplay.from_estimator(
    #     svm_model, X_train,
    #     response_method="predict",
    #     ax=ax,
    #     cmap="coolwarm")
    #
    # # Scatter plot of the test data
    # scatter = ax.scatter(X_test[:, 0],
    #                      X_test[:, 1],
    #                      c=y_test.values,
    #                      edgecolor='k',
    #                      cmap='coolwarm',
    #                      alpha=0.7)
    # # Adding labels and title
    # ax.set_xlabel('PCA Component 1')
    # ax.set_ylabel('PCA Component 2')
    # ax.set_title('Decision Boundary of SVM after PCA')
    # legend = ax.legend(*scatter.legend_elements(), title="Diagnosis")
    # ax.add_artist(legend)
    # plt.tight_layout()
    # plt.show()

    # %% XGBoost model
    X_train, y_train, X_test, y_test = create_splits_under_sampler(
        df=df_model,
        features=col_features,
        target=col_target,
        stratify='diagnosis',
        output_path=result_path_ess
    )

    params = {
        'objective': 'binary:logistic',  # Binary classification
        'eval_metric': 'logloss',  # Evaluation metric
        'eta': 0.1,  # Learning rate
        'max_depth': 6,  # Maximum depth of the trees
        'min_child_weight': 1,  # Minimum sum of instance weight (hessian) needed in a child
        'subsample': 0.8,  # Subsample ratio of the training instance
        'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree
        'alpha': 0.1,  # L1 regularization term on weights
        'lambda': 1  # L2 regularization term on weights
    }

    # Convert data into DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train,
                         label=y_train)
    dtest = xgb.DMatrix(X_test,
                        label=y_test)

    # Train the XGBoost model
    evals_result = {}
    num_rounds = 1000  # Number of boosting rounds
    bst = xgb.train(params,
                    dtrain,
                    num_rounds,
                    evals=[(dtrain, 'train')],
                    evals_result=evals_result,
                    verbose_eval=False)

    # Make predictions on the test set
    y_pred_test_proba = bst.predict(dtest)

    report_test_xgboost = evaluate_predictions_generate_report(
                                                        y_true=y_test,
                                                        y_prob=pd.Series(y_pred_test_proba, index=y_test.index),
                                                        threshold=0.5,
                                                        output_path=result_path_ess,
                                                        title='Test Set',
                                                        model_name='XGBoost')
    y_pred_train_proba = bst.predict(dtrain)
    report_train_xgboost = evaluate_predictions_generate_report(
                                                        y_true=y_train,
                                                        y_prob=pd.Series(y_pred_train_proba, index=y_train.index),
                                                        threshold=0.5,
                                                        output_path=result_path_ess,
                                                        title='Train Set',
                                                        model_name='XGBoost')

    # Extract training and validation log loss from evals_result
    train_logloss = evals_result['train']['logloss']
    # val_logloss = evals_result['eval']['logloss']

    # Plot the training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_logloss, label='Training Log Loss')
    # plt.plot(val_logloss, label='Validation Log Loss')
    plt.xlabel('Number of Boosting Rounds')
    plt.ylabel('Log Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(result_path_ess.joinpath(f'XGBoost_train_curve.png'), dpi=300)
    plt.show()


    # %% Finding the best cut-off
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(df_data['diagnosis'], df_data['ESS'])
    roc_auc = roc_auc_score(df_data['diagnosis'], df_data['ESS'])








