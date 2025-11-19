from config.config import config
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from library.ml_questionnaire.training import run_nested_cv_with_optuna_parallel
import optuna
from pathlib import Path
import matplotlib.pyplot as plt
from library.ml_questionnaire.veto_rule import apply_veto_and_recompute
from library.ml_questionnaire.visualization import plot_hla_vs_nonhla_roc_and_cm, \
    plot_hla_vs_nonhla_roc_and_cm_with_veto
from tabulate import  tabulate

def _load_all_results(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mets, preds = [], []
    for sub in base_dir.iterdir():
        if not sub.is_dir():
            continue
        m = sub / "metrics_outer_folds.csv"
        p = sub / "predictions_outer_folds.csv"
        if m.exists(): mets.append(pd.read_csv(m))
        if p.exists(): preds.append(pd.read_csv(p))
    df_m = pd.concat(mets, ignore_index=True) if mets else pd.DataFrame()
    df_p = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    return df_m, df_p


def downsample_cases_relative_to_controls(df, target_col, case_to_control_frac=0.23, random_seed=42) -> pd.DataFrame:
        """
        Downsample cases so that their count equals case_to_control_frac * number of controls.
        Returns the balanced dataframe and a comparison table for Age, BMI, and sex.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing target, Age, BMI, and sex.
        target_col : str
            Column name for the target variable (1 = case, 0 = control).
        case_to_control_frac : float, default=0.23
            Desired ratio of cases relative to controls (e.g., 0.23 means cases = 23% of controls).
        random_seed : int, default=42
            Random seed for reproducibility.
        """
        print(f'\nDownsampling cases relative to controls: {df[target_col].value_counts()}')
        np.random.seed(random_seed)

        # --- Separate cases and controls ---
        cases = df[df[target_col] == 1]
        controls = df[df[target_col] == 0]

        # --- Number of cases to keep ---
        desired_cases_n = int(case_to_control_frac * len(controls))

        # --- Downsample cases ---
        cases_downsampled = cases.sample(n=desired_cases_n, random_state=random_seed)

        # --- Combine sampled cases with all controls ---
        df_balanced = pd.concat([cases_downsampled, controls], axis=0) \
            .sample(frac=1, random_state=random_seed) \
            .reset_index(drop=True)
        print(f'\nNumber of cases downsampled to: {df_balanced[target_col].value_counts()}')

        return df_balanced

model_types = [
                "random_forest",
               "lightgbm",
               "xgboost",
               "elastic_net",
               "logistic_regression",
               "SVM",
               "LDA"]

if __name__ == '__main__':
    # %% Read data
    df_data = pd.read_csv(config.get('data_pre_proc_files').get('anic_okun'))

    # %%
    OVERWRITE = True
    result_dir = config.get('results_path').get('results').joinpath(f'main_cv_less_cases_complete_train')
    result_dir.mkdir(parents=True, exist_ok=True)
    # %% input
    random_seed = 42  # Random seed for reproducibility
    # Cross-validation parameters (for evaluation, optimization, and fine-tuning)
    cv_folds_eval = 10  # Folds for evaluating models
    cv_folds_optimization = 5  # Folds used in cross_val_score during optimization
    optuna_trials = 250 # 250  # Number of trials to run during optimization
    optuna_direction = "maximize"  # Optimization direction

    optuna_sampler = optuna.samplers.TPESampler()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # %% Select columns and drop columns with nans
    target = 'NT1 ICSD3 - TR'
    target_nt2 = target.replace('1', '2')
    df_data['subject_id'] = df_data.index
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
    columns = columns + ['subject_id']
    df_data = df_data.loc[:, columns]
    # df_data = df_data.dropna(axis=1)
    cols_with_many_nans = df_data.columns[df_data.isna().sum() > 15]
    df_data.drop(cols_with_many_nans, axis=1, inplace=True)
    df_data.reset_index(drop=True, inplace=True)
    df_data = df_data.reindex(sorted(df_data.columns), axis=1)
    print(f'Dataset dimension: {df_data.shape}')

    # %%
    df_data = downsample_cases_relative_to_controls(df_data, target_col=target, case_to_control_frac=0.23)

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
        'ess': {
            'features':  ['ESS'],
            'target': target,
            'dqb': False,
        },
        'ess_hla': {
            'features': ['ESS'] + ['DQB10602'],
            'target': target,
            'dqb': True,
        },

    }
    feature_set_mapper = {
        f'questionnaire': f'Full feature set (k={len(configurations.get("questionnaire").get("features"))})',
        'questionnaire_hla': f'Full Feature Set + DQB1*06:02  (k={len(configurations.get("questionnaire_hla").get("features"))})',
        'ukbb': f'Reduced Feature Set (k={len(configurations.get("ukbb").get("features"))})',
        'ukbb_hla': f'Reduced Feature Set + DQB1*06:02 (k={len(configurations.get("ukbb_hla").get("features"))})',
        'ess_hla': f'ESS + HLA',
        'ess': f'ESS',
    }
    # %%
    # %% Run nested CV
    if OVERWRITE:
        for conf_name, conf_values in configurations.items():
            print(f'Running configuration: {conf_name}')

            # create output path for the current feature configuration
            dir_config = result_dir.joinpath(conf_name)
            dir_config.mkdir(parents=True, exist_ok=True)

            # get the features and target of the current configuration
            feature_cols = conf_values['features']
            target_col = conf_values['target']

            # get the formal labels for meta-data
            conf_label = feature_set_mapper.get(conf_name, conf_name)
            k_feats = len(conf_values["features"])
            dqb_flag = conf_values["dqb"]

            df_metrics, df_predictions, df_inner_val_records = run_nested_cv_with_optuna_parallel(
                df=df_data,
                target_col=conf_values['target'],
                feature_cols=conf_values['features'],
                col_id="subject_id",  # fixed: was group_col
                results_dir=dir_config,
                continuous_cols=['ESS'] if 'ESS' in conf_values['features'] else None,
                model_types=model_types,  # will default to ["random_forest", "lightgbm",    "xgboost", "elastic_net"]
                random_seed=42,
                procerssor=True,  # fixed: was preprocessor=True
                n_outer_splits=cv_folds_eval,
                n_inner_splits=cv_folds_optimization,
                n_trials=optuna_trials,
                min_sens=0.6,
                maximize="spec"
            )

            # after run_nested_cv_with_optuna_parallel returns (df_metrics, df_predictions, df_inner_val_records)
            # include the meta data
            for _df in (df_metrics, df_predictions, df_inner_val_records):
                _df["config"] = conf_name
                _df["feature_set_label"] = conf_label
                _df["k_features"] = k_feats
                _df["dqb"] = bool(dqb_flag)

            # save
            df_metrics.to_csv(dir_config / "metrics_outer_folds.csv", index=False)
            df_predictions.to_csv(dir_config / "predictions_outer_folds.csv", index=False)
            df_inner_val_records.to_csv(dir_config / "inner_val_records.csv", index=False)

    # Collection: Build master tables (whether we just ran or are reusing disk)
    master_dir = result_dir / "master"
    master_dir.mkdir(exist_ok=True)

    df_metrics_all, df_predictions_all = _load_all_results(result_dir)
    df_metrics_all.to_parquet(master_dir / "metrics_all.parquet", index=False)
    df_predictions_all.to_parquet(master_dir / "predictions_all.parquet", index=False)

    df_predictions_all = pd.merge(df_predictions_all,
                                  df_data[['subject_id', 'DQB10602']],
                                  on="subject_id",
                                  how="left")


    # %% compute the veto predictions
    path_veto_pred = master_dir / "df_predictions_with_veto.parquet"
    path_veto_metrics = master_dir / "df_metrics_with_veto.parquet"
    if not path_veto_pred and not path_veto_metrics:
        df_predictions_with_veto, df_metrics_with_veto, df_veto_counts = apply_veto_and_recompute(
                df_predictions_all=df_predictions_all,
                df_metrics_all=df_metrics_all,
                subject_col= "subject_id",
                hla_col= "DQB10602",
                dqb_flag_col= "dqb",  # boolean flag if present
                config_col= "config",
                model_col = "model_type",
                opt_col = "optimization",
                fold_col= "outer_fold",
                thr_type_col = "threshold_type",
                thr_value_col = "threshold_value",
                ytrue_col = "y_true",
                yscore_col= "y_score",
                ypred_col= "y_pred",)

        df_predictions_with_veto.to_parquet(path_veto_pred, index=False)
        df_metrics_with_veto.to_parquet(path_veto_metrics, index=False)
        df_veto_counts.to_csv(master_dir.joinpath('veto_count.csv'), index=False)
    else:
        print(f'Reading all predictions and metrics..."')
        df_predictions_with_veto = pd.read_parquet(path_veto_pred)
        df_metrics_with_veto = pd.read_parquet(path_veto_metrics)
        print(f'Read completed')

    # double check the formal names mapping
    df_predictions_with_veto['feature_set_label'] = df_predictions_with_veto['config'].map(feature_set_mapper)
    df_metrics_with_veto['feature_set_label'] = df_metrics_with_veto['config'].map(feature_set_mapper)

    assert df_predictions_with_veto['feature_set_label'].isna().sum() == 0
    # %%  Generate plots
    pair_map = {"questionnaire": "questionnaire_hla",
                "ukbb": "ukbb_hla",
                "ess": "ess_hla"}
    RUN_MOST_SPECIFIC = True
    RUN_MOST_SPECIFIC_BALANCE = False
    if RUN_MOST_SPECIFIC:
        path_most_specific = master_dir.joinpath('most_specific')
        path_most_specific.mkdir(exist_ok=True, parents=True)
        df_selected_best_sp, df_per_model_best_sp = plot_hla_vs_nonhla_roc_and_cm_with_veto(
            df_predictions_all=df_predictions_with_veto,
            df_metrics_all=df_metrics_with_veto,
            feature_set_mapper=feature_set_mapper,
            class_name={0: "CL", 1: "NT1"},
            pair_map=pair_map,
            prefer_thresholds=None,
            min_spec=99.0,
            min_sens=60.0,
            xlim=(-0.01, 0.6),
            ylim=(0.6, 1.01),
            title="Best Models ROC Per Feature Set & Classifications\nVeto Rule Implementation Feature Set with No HLA During Training",
            fonts={"scale": 1.5},
            palette='Dark2',
            # fig_size=(12, 10)
            output_path=path_most_specific.joinpath('hla_vs_non_hla_roc_and_cm_with_veto_sp99_se_60.png'),
        )
        df_selected_best_sp.to_csv(path_most_specific / 'df_selected_best_sp.csv', index=False)
        df_per_model_best_sp.to_csv(path_most_specific / 'df_per_model_best_sp.csv', index=False)

    if RUN_MOST_SPECIFIC_BALANCE:
        # this requiers the sort by function to include the auc and prc so it balance better the Se and Sp
        # in function select_best_model_type_per_config()
        path_spec_but_balance = master_dir.joinpath('spec_but_balance')
        path_spec_but_balance.mkdir(exist_ok=True, parents=True)

        df_selected_best_bl, df_per_model_best_bl = plot_hla_vs_nonhla_roc_and_cm_with_veto(
            df_predictions_all=df_predictions_with_veto,
            df_metrics_all=df_metrics_with_veto,
            feature_set_mapper=feature_set_mapper,
            class_name={0: "CL", 1: "NT1"},
            pair_map=pair_map,
            prefer_thresholds=None,
            min_spec=90.0,
            min_sens=90.0,
            xlim=(-0.01, 0.6),
            ylim=(0.6, 1.01),
            title="Best Models ROC Per Feature Set & Classifications\nVeto Rule Implementation Feature Set with No HLA During Training",
            fonts={"scale": 1.5},
            palette='Dark2',
            # fig_size=(12, 10)
            output_path=path_spec_but_balance.joinpath('hla_vs_non_hla_roc_and_cm_with_veto_sp90_se_90.png'),
        )
        df_selected_best_bl.to_csv(path_spec_but_balance / 'df_selected_best_bl.csv', index=False)
        df_per_model_best_bl.to_csv(path_spec_but_balance / 'df_per_model_best_sp.csv', index=False)


    # %% Feature Importance, XGBOOST















