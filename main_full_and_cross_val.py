from config.config import config
import numpy as np
import pandas as pd
from library.ml_models import (compute_cross_validation,
                               models,
                               make_veto_dataset,
                               load_imputed_folds,
                               summarize_fold_consistency, compute_full_training)
from library.metrics_functions import apply_veto_rule_re_classifications
from library.plot_functions import (plot_model_metrics, plot_model_metrics_specific_columns,
                                    create_venn_diagram, plot_elastic_net_model_coefficients,
                                    plot_dcurves_per_fold, ppv_curve,multi_ppv_plot_combined, plot_calibration,
                                    multi_ppv_plot, multi_calibration_plot)
from library.metrics_functions import recompute_classification, extract_metrics
from typing import Optional, Tuple, List


if __name__ == '__main__':
    # %% Read data
    # df_data = pd.read_csv(config.get('data_pre_proc_files').get('ssq_ssqdx'))
    # df_data = df_data.loc[df_data['narcolepsy'] != 'pseudo narcolepsy']
    # df_data.rename(columns={'cataplexy_clear_cut': 'NT1'}, inplace=True)

    df_data = pd.read_csv(config.get('data_pre_proc_files').get('anic_okun'))

    # df_data.rename(columns={'cataplexy_clear_cut': 'NT1'}, inplace=True)

    # %% output paths
    TEST = True
    OVERWRITE = False
    PLOT_VEN = False
    test_flag = 'test_' if TEST else ''
    base_path = config.get('results_path').get('results')

    path_avg_metrics_config          = base_path.joinpath(f'{test_flag}avg_metrics_config.csv')
    path_classifications_config      = base_path.joinpath(f'{test_flag}classifications_config.csv')
    path_feature_importance          = base_path.joinpath(f'{test_flag}feature_importance.csv')
    path_pred_prob_elastic           = base_path.joinpath(f'{test_flag}pred_prob_elasticnet.csv')
    path_models_pred_prob            = base_path.joinpath(f'{test_flag}pred_prob_all_models.csv')

    # Paths for full‐dataset (no cross‐val) results
    path_full_metrics                = base_path.joinpath(f'{test_flag}full_dataset_metrics.csv')
    path_full_classifications        = base_path.joinpath(f'{test_flag}full_dataset_classifications.csv')
    path_full_elastic_params         = base_path.joinpath(f'{test_flag}full_dataset_elastic_params.csv')


    path_avg_metrics_veto = config.get('results_path').get('results').joinpath(f'{test_flag}avg_metrics_veto.csv')
    path_avg_classifications_config_veto = config.get('results_path').get('results').joinpath(f'{test_flag}classifications_config_veto.csv')
    path_avg_paper = config.get('results_path').get('results').joinpath(f'{test_flag}avg_paper.csv')
    path_model_metrics = config.get('results_path').get('results').joinpath(f'model_metrics.png')
    # path_avg_paper_veto = config.get('results_path').get('results').joinpath(f'avg_paper_veto.csv')
    path_plot_best_model = config.get('results_path').get('main_model')
    path_fp_fp_tn_fn_tab = config.get('results_path').get('results').joinpath('tab_fp_fp_tn_fn.csv')

    # full dataset model
    path_full_dataset_metrics = config.get('results_path').get('results').joinpath(f'{test_flag}full_dataset_metrics.csv')
    path_full_dataset_classifications = config.get('results_path').get('results').joinpath(f'{test_flag}full_elastic_params.csv')
    path_full_dataset_elastic_params = config.get('results_path').get('results').joinpath(f'{test_flag}full_elastic_params.csv')

    # pred and prob of full dataset concat with k-folds
    path_models_pred_pob = config.get('results_path').get('results').joinpath(f'{test_flag}pred_pob.csv')


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
    # %% configuration of different models to run
    # col_ukbb = ['Age', 'BMI', 'sex', 'ESS', 'JAW', 'HEAD', 'HAND',
    #             'SPEECH', 'JOKING', 'LAUGHING', 'ANGER',
    #             'QUICKVERBAL']
    col_ukbb = ['BMI', 'ESS', 'JAW', 'HEAD', 'HAND', 'KNEES',
                'SPEECH', 'JOKING', 'LAUGHING', 'ANGER',
                'QUICKVERBAL']

    # some columns are not available because of missigness e.g., BMI
    col_ukbb_avail = [col for col in df_data.columns if col in col_ukbb]

    configurations = {
        "questionnaire": {
            'features': [col for col in df_data.columns if not col in [target, 'DQB10602']],
            'target': target,
            'dqb': False,
        },

        'questionnaire_hla': {
            'features': [col for col in df_data.columns if col != target],
            'target': target,
            'dqb': True,
        },

        'ukbb': {
            'features': col_ukbb_avail,
            'target': target,
            'dqb': False,
        },

        'ukbb_hla': {
            'features': col_ukbb_avail + ['DQB10602'],
            'target': target,
            'dqb': True,
        },

    }

    feature_set_mapper = {
        f'questionnaire': f'Full feature set (k={len(configurations.get("questionnaire").get("features"))})',
        'questionnaire_hla': f'Full Feature Set + DQB1*06:02  (k={len(configurations.get("questionnaire_hla").get("features"))})',
        'ukbb': f'Reduced Feature Set (k={len(configurations.get("ukbb").get("features"))})',
        'ukbb_hla': f'Reduced Feature Set + DQB1*06:02 (k={len(configurations.get("ukbb_hla").get("features"))})',
    }
    for key, val in feature_set_mapper.items():
        print(f'\t{key}: {val}')

    for key, dictionary in configurations.items():
        num_feature = len(dictionary.get('features'))
        print(f'{key}: {num_feature}')

    if PLOT_VEN:
        # Create a Venn diagram comparing "questionnaire" and "ukbb"
        create_venn_diagram(
            configs=configurations,
            config1="questionnaire",
            config2="ukbb",
            figure_size=(9, 8),
            set_labels=("Full Questionnaire Feature Set", "Reduces Feature Set"),
            colors=('pink', 'orange'),
            alpha=0.7,
            title="Venn Diagram of Questionnaire vs. UKBB Features"
        )

    # run the analysis if the files do not exist or it overwrite is set to True
    if not (path_avg_metrics_config.is_file() and path_classifications_config.is_file()) or OVERWRITE:
        # Accumulators across all configurations
        df_all_classifications = pd.DataFrame()
        df_all_metrics         = pd.DataFrame()
        df_all_feature_imp     = pd.DataFrame()
        df_all_elastic_preds   = pd.DataFrame()
        df_all_full_metrics    = pd.DataFrame()
        df_all_full_classif    = pd.DataFrame()
        df_all_full_elastic    = pd.DataFrame()
        df_all_model_probs     = pd.DataFrame()
        k = 5  # number of folds
        for conf_name, conf_values in configurations.items():
            print(f'Running configuration: {conf_name}')

            # 1) Build the dataset (with/without DQB1*06:02) + optional veto logic
            df_model, df_dqb_veto = make_veto_dataset(
                df_data=df_data,
                use_dqb10602=conf_values['dqb']
            )

            # 2) FULL‐DATA (no cross‐validation) training
            (
                df_full_metrics,
                df_full_classifications,
                df_full_elastic_params,
                df_full_model_probs
            ) = compute_full_training(
                models=models,
                df_model=df_model,
                features=conf_values['features'],
                target=conf_values['target'],
            )

            # 3) K‐FOLD CROSS‐VALIDATION
            (
                df_avg_metrics,
                df_classifications,
                df_elastic_params,
                df_elastic_preds,
                df_fold_model_probs
            ) = compute_cross_validation(
                models=models,
                df_model=df_model,
                features=conf_values['features'],
                target=conf_values['target'],
                k=k
            )

            # Add metadata columns for configuration & HLA flag
            df_classifications['config'] = conf_name
            df_classifications['hla']    = conf_values['dqb']
            df_all_classifications = pd.concat([df_all_classifications, df_classifications], ignore_index=True)

            df_avg_metrics['config'] = conf_name
            df_all_metrics = pd.concat([df_all_metrics, df_avg_metrics], ignore_index=True)

            df_elastic_params['config'] = conf_name
            df_all_feature_imp = pd.concat([df_all_feature_imp, df_elastic_params], ignore_index=True)

            df_elastic_preds['config'] = conf_name
            df_all_elastic_preds = pd.concat([df_all_elastic_preds, df_elastic_preds], ignore_index=True)

            # Combine the fold & full‐data probability DataFrames
            df_fold_model_probs['config']      = conf_name
            df_fold_model_probs['hla']         = conf_values['dqb']
            df_full_model_probs['config']      = conf_name
            df_full_model_probs['hla']         = conf_values['dqb']
            df_combined_probs = pd.concat(
                [df_fold_model_probs, df_full_model_probs],
                ignore_index=True
            )
            df_all_model_probs = pd.concat([df_all_model_probs, df_combined_probs], ignore_index=True)

            # Collect full‐data results
            df_full_metrics['config']         = conf_name
            df_all_full_metrics = pd.concat([df_all_full_metrics, df_full_metrics], ignore_index=True)

            df_full_classifications['config'] = conf_name
            df_all_full_classif = pd.concat([df_all_full_classif, df_full_classifications], ignore_index=True)

            df_full_elastic_params['config']  = conf_name
            df_all_full_elastic = pd.concat([df_all_full_elastic, df_full_elastic_params], ignore_index=True)

        # ---------------------------
        # Post‐processing & saving
        # ---------------------------

        # use the index to place the HLA measure if present
        df_combined_probs['hla_results'] = (
            df_data['DQB10602'].to_numpy()[df_combined_probs['indices']]
        )
        df_classifications['hla_results'] = (
            df_data['DQB10602'].to_numpy()[df_classifications['index']]
        )
        df_full_classifications['hla_results'] = (
            df_data['DQB10602'].to_numpy()[df_full_classifications['index']]
        )

        df_all_model_probs['hla_results'] = (
            df_data['DQB10602'].to_numpy()[df_all_model_probs['indices']]
        )
        # 1) Finalize cross‐val metrics ordering
        model_order = df_all_metrics.loc[
            df_all_metrics['config'] == 'questionnaire'
        ].sort_values(by='specificity', ascending=False)['model'].values
        df_all_metrics['model'] = pd.Categorical(df_all_metrics['model'], categories=model_order, ordered=True)
        df_all_metrics = df_all_metrics.sort_values(
            by=['config', 'specificity'], ascending=[True, False]
        )

        # 2) Save cross‐validation results
        df_all_metrics.to_csv(path_avg_metrics_config, index=False)
        df_all_classifications.to_csv(path_classifications_config, index=False)
        df_all_feature_imp.to_csv(path_feature_importance, index=False)
        df_all_elastic_preds.to_csv(path_pred_prob_elastic, index=False)
        df_all_model_probs['config'] = df_all_model_probs['config'].map(feature_set_mapper)
        df_all_model_probs.to_csv(path_models_pred_prob, index=False)


        # 3) Save full‐data (no cross‐val) results
        df_all_full_metrics.to_csv(path_full_metrics, index=False)
        df_all_full_classif.to_csv(path_full_classifications, index=False)
        df_all_full_elastic.to_csv(path_full_elastic_params, index=False)
    else:
        df_metrics_configs = pd.read_csv(path_avg_metrics_config)
        df_classifications_configs = pd.read_csv(path_classifications_config)
        df_feature_importance = pd.read_csv(path_feature_importance)
        df_elastic_pred_config = pd.read_csv(path_pred_prob_elastic)

    # sort the frames
    df_metrics_configs = df_metrics_configs.sort_values(by=['model', 'config', 'specificity'],
                                   ascending=[False, True, False]
                                   )
    # %% Apply the veto
    (df_veto_classifications,
     df_veto_avg_metrics) = apply_veto_rule_re_classifications(df_classifications=df_classifications_configs.copy(),
                                                               df_data=df_data)

    df_veto_classifications.to_csv(path_avg_classifications_config_veto, index=False)
    df_veto_avg_metrics = df_veto_avg_metrics.sort_values(by=['config', 'model', 'specificity'],
                                   ascending=[False, True, False]
                                   )
    # df_veto_avg_metrics['config'].replace(feature_set_mapper, inplace=True)
    df_veto_avg_metrics.to_csv(path_avg_metrics_veto, index=False)

    # %% Merge the avg fold metrics before and after the veto to have all in one dataset. Merge by model and config pairs
    col_veto_mapper = {col: f'{col}_veto' for col in df_veto_avg_metrics.columns if not col in ['config', 'model']}
    df_veto_avg_metrics.rename(columns=col_veto_mapper, inplace=True)

    df_avg_metrics = pd.merge(left=df_metrics_configs,
                              right=df_veto_avg_metrics,
                              on=['model', 'config'],
                              how='left')

    new_order = ['config', 'model'] + [col for col in df_avg_metrics.columns if col not in ['model', 'config']]
    df_avg_metrics = df_avg_metrics[new_order]
    df_avg_metrics = df_avg_metrics.sort_values(by=['config', 'specificity'],
                                                        ascending=[True, False])  # ['config', 'model']

    df_avg_metrics['config'].replace(feature_set_mapper, inplace=True)
    df_avg_metrics.to_csv(path_avg_paper, index=False)


    # plot_model_metrics(df_avg_metrics, palette='muted', figsize=(16, 8))

    df_avg_metrics_copy = df_avg_metrics.copy()
    df_avg_metrics_copy = df_avg_metrics_copy.rename(columns={
        'ppv_apparent': 'PPV Apparent',
        'ppv': 'PPV',
        'f1_score': 'F1 Score',
        'specificity': 'Specificity',
        'sensitivity': 'Sensitivity',
    })
    columns_to_plot = ['Specificity', 'Sensitivity', 'F1 Score', 'PPV'] #  'f1_score', 'npv', 'ppv']  # Example list of columns
    plot_model_metrics_specific_columns(df_avg_metrics_copy,
                       columns=columns_to_plot,
                       palette='pastel',
                       figsize=(18, 4.5),
                        output_path=path_model_metrics)


    # TODO: INlcue the ROC cuvre with all the models in the same figure

    # %%
    # Extract classification metrics

    # Apply the function to the dataset
    df_veto_classifications = recompute_classification(df_veto_classifications)

    # Initialize a list to store results
    data_list = []

    # Iterate through unique folds and models
    for model in df_classifications_configs['model_name'].unique():
        # model = 'Elastic Net'
        for fold in np.sort(df_classifications_configs['fold'].unique()):
            # fold = 1
            # Filter for current fold and model
            df_hla_true = df_classifications_configs.loc[
                (df_classifications_configs['hla'] == True) &
                (df_classifications_configs['fold'] == fold) &
                (df_classifications_configs['model_name'] == model)
                ]

            df_hla_false = df_classifications_configs.loc[
                (df_classifications_configs['hla'] == False) &
                (df_classifications_configs['fold'] == fold) &
                (df_classifications_configs['model_name'] == model)
                ]
            #
            # df_veto_rule = df_classifications_configs.loc[
            #     (df_classifications_configs['fold'] == fold) &
            #     (df_classifications_configs['model_name'] == model)
            #     ]

            # Filter now the congifucarion results for the observations the veto rule was applied
            df_hla_true_veto = df_veto_classifications.loc[
                (df_veto_classifications['hla'] == True) &
                (df_veto_classifications['fold'] == fold) &
                (df_veto_classifications['model_name'] == model) &
                (df_veto_classifications['dataset_type'] == 'validation')
                ]

            df_hla_false_veto = df_veto_classifications.loc[
                (df_veto_classifications['hla'] == False) &
                (df_veto_classifications['fold'] == fold) &
                (df_veto_classifications['model_name'] == model) &
                (df_veto_classifications['dataset_type'] == 'validation')
                ]

            # df_veto_rule_veto = df_veto_classifications.loc[
            #     (df_veto_classifications['fold'] == fold) &
            #     (df_veto_classifications['model_name'] == model) &
            #     (df_veto_classifications['dataset_type'] == 'validation')
            #     ]

            # Extract classification metrics
            # Get classification counts
            metrics_hla_true = extract_metrics(df_hla_true)
            metrics_hla_false = extract_metrics(df_hla_false)
            metrics_hla_true_veto = extract_metrics(df_hla_true_veto)
            metrics_hla_false_veto = extract_metrics(df_hla_false_veto)


            # Append to data list
            data_list.append([
                model,
                fold,

                metrics_hla_true.iloc[0]['FN'], metrics_hla_true.iloc[0]['FP'], metrics_hla_true.iloc[0]['TN'],
                metrics_hla_true.iloc[0]['TP'],

                metrics_hla_false.iloc[0]['FN'], metrics_hla_false.iloc[0]['FP'], metrics_hla_false.iloc[0]['TN'],
                metrics_hla_false.iloc[0]['TP'],

                metrics_hla_true_veto.iloc[0]['FN'], metrics_hla_true_veto.iloc[0]['FP'], metrics_hla_true_veto.iloc[0]['TN'],
                metrics_hla_true_veto.iloc[0]['TP'],

                metrics_hla_false_veto.iloc[0]['FN'], metrics_hla_false_veto.iloc[0]['FP'], metrics_hla_false_veto.iloc[0]['TN'],
                metrics_hla_false_veto.iloc[0]['TP'],
            ])

    # Create final DataFrame
    df_final = pd.DataFrame(data_list, columns=[
        'model_name',
        'Fold',

        'FN_HLA_True', 'FP_HLA_True', 'TN_HLA_True', 'TP_HLA_True',

        'FN_HLA_False', 'FP_HLA_False', 'TN_HLA_False', 'TP_HLA_False',

        'FN_HLA_True_After_Veto', 'FP_HLA_True_After_Veto', 'TN_HLA_True_After_Veto', 'TP_HLA_True_After_Veto',

        'FN_HLA_False_After_Veto', 'FP_HLA_False_After_Veto', 'TN_HLA_False_After_Veto', 'TP_HLA_False_After_Veto'
    ])
    df_final.to_csv(path_fp_fp_tn_fn_tab, index=False)
    # %%
    #       Evaluation Best Model - Elastic net
    #       Using the true, pred, pred_prob obtained for each fold and for each feature set
    #
    # %% Feature importance plot
    # Re-import necessary libraries after execution state reset

    df_feature_importance['config'].replace(feature_set_mapper, inplace=True)


    # plot_elastic_net_model_coefficients(df_feature_importance[df_feature_importance['config'].isin([ 'Reduced Feature Set + DQB1*06:02 (k=11)', 'Reduced Feature Set (k=10)'])],
    #                                     figsize=(8,6),
    #                                     output_path=path_plot_best_model)

    plot_elastic_net_model_coefficients(df_feature_importance,
                                        # figsize=(8,6),
                                        output_path=path_plot_best_model)

    # Define the mapping of feature display names
    feature_names = {
        'ANGER': 'anger',
        'HEAD': 'Head drooping',
        'JAW': 'Jaw',
        'JOKING': 'Joking',
        'KNEES': 'Knee buckling',
        'LAUGHING': 'Laughing',
        'SPEECH': 'Speech slurring',
        'DQB10602': 'DQB10602'
    }

    # Define the columns (configurations) in the desired order
    config_order = [
        'Full feature set (k=26)',
        'Full Feature Set + DQB1*06:02  (k=27)',
        'Reduced Feature Set (k=10)',
        'Reduced Feature Set + DQB1*06:02 (k=11)'
    ]

    # Initialize a dictionary to build the DataFrame
    table_data = {col: [] for col in ['Feature'] + config_order}

    # Fill in the data
    for feature_code, display_name in feature_names.items():
        table_data['Feature'].append(display_name)
        for config in config_order:
            row = df_feature_importance[
                (df_feature_importance['Feature'] == feature_code) &
                (df_feature_importance['config'] == config)
                ]
            if not row.empty:
                mean = row['Mean Coefficient'].values[0]
                std = row['Standard Error'].values[0]
                value = f"{mean:.3f} ± {std:.3f}"
            else:
                value = "-"
            table_data[config].append(value)
    df_table = pd.DataFrame(table_data)

    # %% decison curve
    df_elastic_pred = pd.read_csv(path_pred_prob_elastic)
    # from dcurves import plot_graphs, my_plot_graphs
    df_elastic_pred_dcurve = df_elastic_pred.copy()

    df_elastic_pred_dcurve['configuration_formal'] = df_elastic_pred_dcurve['configuration'].map(feature_set_mapper)
    # Example usage:
    # Assume you have a DataFrame "results" with columns 'true_label' and 'pred_prob'.
    prevalence = 30 / 100000  # i.e., 0.0003
    for feature_set_config in df_elastic_pred_dcurve['configuration_formal'].unique():
        plot_dcurves_per_fold(df_results= df_elastic_pred_dcurve,
                              prevalence=prevalence,
                              configuration=feature_set_config,
                              output_path=path_plot_best_model)

    # %% Prevalance plot
    best_model = 'Elastic Net'
    # all in one figure
    # multi_ppv_plot(df_avg_metrics=df_avg_metrics,
    #                model_name=best_model,
    #                rows=2,
    #                figsize=(12,8),
    #                output_path=None)
    df_elastic_pred_config['configuration'].replace(feature_set_mapper, inplace=True)

    multi_ppv_plot_combined(df_predictions_model=df_elastic_pred_config,
                   figsize=(10,6),
                    population_prevalence = 30 / 100000,  # 0.0003
                   output_path=path_plot_best_model)



    # %% calibration plot

    # df_elastic_pred_config['configuration'].replace(feature_set_mapper, inplace=True)

    # all in one figure
    df_brier_scores = multi_calibration_plot(df_predictions=df_elastic_pred_config,
                           model_name=best_model,
                           rows=2,
                           output_path=path_plot_best_model)


    df_grouped = df_brier_scores.groupby('configuration').agg(
        brier_score=('brier_score', lambda x: f"{x.mean():.4f} ({x.std():.4f})"),
        log_loss=('log_loss', lambda x: f"{x.mean():.4f} ({x.std():.4f})"),
        auc=('auc', lambda x: f"{x.mean():.4f} ({x.std():.4f})")
    ).reset_index()

    df_grouped['configuration'].replace(feature_set_mapper, inplace=True)
    df_grouped.to_csv(path_plot_best_model.joinpath('loss_metrics_elastic_net.csv'), index=False)

    # %% Check feature distribution across the folds
    # Load folds
    imputed_folds = load_imputed_folds(save_path=r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\NarcCataplexyQuestionnaire\data')

    # Extract features from one fold
    df_sample = imputed_folds[0]['train_data']
    target_name = 'target'  # Replace with your actual target column name
    # get feature names
    feature_names = [col for col in df_sample.columns if col != target_name]

    # Run summary
    df_fold_summary = summarize_fold_consistency(imputed_folds, feature_names, target_name)

    # Show first few rows
    print(df_fold_summary.head())

    # Optional: save to CSV
    df_fold_summary.to_csv('fold_consistency_summary.csv', index=False)
