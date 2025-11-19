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
import seaborn as sns

# %%

# path = r"C:\Users\riccig01\OneDrive\Projects\MignotLab\CataplexyQuestionnaire\data\pproc\for_publication\pp_anic_okun.xlsx"
# df_final = pd.read_excel(path)
#
# df_csf = pd.read_excel(r"C:\Users\riccig01\OneDrive\Projects\MignotLab\CataplexyQuestionnaire\data\pproc\for_publication\csf_records.xlsx")
# df_csf = df_csf[~df_csf['CSF'].isna()]
#
# # --- Identify overlapping columns ---
# common_cols = list(set(df_csf.columns).intersection(df_final.columns))
# print("Common columns:", common_cols)
#
# df_csf = df_csf[common_cols + ['CSF']]
# df_final = df_final[common_cols]

# --- normalize columns
# def clean_soremp(series: pd.Series) -> pd.Series:
#     """
#     Clean and normalize 'SOREMP' column.
#     Extracts number before ' of' if present, handles NaN, numeric, and malformed values.
#     If 'of' not found, returns the original value.
#
#     Parameters
#     ----------
#     series : pd.Series
#         Input column (e.g., '3 of 5', 4, NaN).
#
#     Returns
#     -------
#     pd.Series
#         Cleaned numeric or original values.
#     """
#     def parse_value(x):
#         if pd.isna(x):
#             return np.nan
#         if isinstance(x, (int, float)):
#             return x
#         try:
#             x_str = str(x).strip().lower()
#             if "of" in x_str:
#                 val = x_str.split("of")[0].replace(",", ".").strip()
#                 return float(val)
#             # no 'of' → return numeric if possible, else original string
#             val = float(x_str) if x_str.replace(".", "", 1).isdigit() else x_str
#             return val
#         except Exception:
#             return x  # fallback to original if parsing fails
#
#     return series.apply(parse_value)
#
# df_csf["SOREMP"] = clean_soremp(df_csf["SOREMP"])
# df_final["SOREMP"] = clean_soremp(df_final["SOREMP"])
#
#
# df_csf['SOREMP'] = df_csf['SOREMP'].astype(float)
# df_final['SOREMP'] = df_final['SOREMP'].astype(float)
#
#
# df_csf['MSLT'] = df_csf['MSLT'].astype(float)
# df_final['MSLT'] = df_final['MSLT'].astype(float)
#
# df_csf['BMI'] = df_csf['BMI'].fillna(-9).astype(float)
# df_final['BMI'] = df_final['BMI'].fillna(-9).astype(float)
#
#
#
# # --- Merge using matching fields ---
# df_merged = pd.merge(
#     df_final,
#     df_csf,
#     on=common_cols,
#     how="left",
#     suffixes=("", "_CSF")
# )
#
# merged_rows = []
# for idx in range(len(df_csf)):
#     mask = (
#         (df_final["Age"] == df_csf.loc[idx, "Age"]) &
#         (df_final["sex"] == df_csf.loc[idx, "sex"]) &
#         (df_final["DQB10602"] == df_csf.loc[idx, "DQB10602"]) &
#         (df_final["ESS"] == df_csf.loc[idx, "ESS"]) &
#         (df_final["NT1 ICSD3 - TR"] == df_csf.loc[idx, "NT1 ICSD3 - TR"])
#     )
#
#     matches = df_final[mask]
#
#     if not matches.empty:
#         print(f"{idx}: {len(matches)} match(es) found.")
#
#         # Reset indexes to avoid InvalidIndexError
#         match_reset = matches.reset_index(drop=True)
#         csf_row = df_csf.loc[[idx]].reset_index(drop=True)
#
#         # Concatenate horizontally, ignoring index alignment
#         merged_row = pd.concat([match_reset, csf_row.add_suffix("_CSF")], axis=1)
#
#         merged_rows.append(merged_row)
#
# # Combine all results
# df_manual_merge = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
#
# print(f"\nTotal merged pairs: {len(df_manual_merge)}")
# print(df_manual_merge.head())
#
# print(f"\nTotal merged pairs: {len(df_manual_merge)}")
# print(df_manual_merge.head().T)
#
#
# print(f"Matched pairs with CSF values: {len(df_merged)}")
# print(df_merged.head())




# %%
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


model_types = [
                "random_forest",
               "lightgbm",
               "xgboost",
               "elastic_net",
               "logistic_regression",
               "SVM",
               "LDA"]


def plot_ppv_vs_prevalence(df,
                           prevalence_range=(1e-4, 0.5),
                           figsize=(8,6),
                           prevalence_current_dataset: float = None,
                           scale: float = 1.0,
                           output_path:Path = None,):
    def compute_ppv(sens, spec, prev):
        return (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))

    prevalence_grid = np.logspace(np.log10(prevalence_range[0]),
                                  np.log10(prevalence_range[1]), 400)

    # Map messy names to clean groups
    def map_group(name):
        if "Full" in name:
            return "Full Feature Set"
        elif "Reduced" in name:
            return "Reduced Feature Set"
        elif "ESS" in name:
            return "ESS"
        else:
            return name.strip()

    df = df.copy()
    df["group"] = df["feature_set"].apply(map_group)

    # Assign one color per group (journal friendly)
    group_colors = dict(zip(df["group"].unique(),
                            sns.color_palette("Set1", n_colors=df["group"].nunique())))

    # Line styles by type
    line_styles = {
        "non-HLA": "solid",
        "HLA": (0, (5, 3)),     # long dash
        "Veto": (0, (2, 2)),    # short dash
    }

    # Sort by group and type order
    type_order = {"non-HLA": 0, "HLA": 1, "Veto": 2}
    df_sorted = df.sort_values(by=["group","type"], key=lambda x: x.map(type_order))

    plt.figure(figsize=figsize)

    for _, row in df_sorted.iterrows():
        sens, spec = row["sensitivity"], row["specificity"]
        ppv_curve = compute_ppv(sens, spec, prevalence_grid)

        group = row["group"]
        linestyle = line_styles[row["type"]]
        color = group_colors[group]

        # Legend: show group + type, clean formatting
        fs_name = row["feature_set"].replace("\n", " ")
        plt.plot(prevalence_grid, ppv_curve,
                 label=f"{fs_name} ({row['type']})",
                 color=color,
                 linestyle=linestyle,
                 linewidth=2)

    # Reference lines
    plt.axvline(3e-4, color="black", linestyle=":", label="General population (~30/100k)")
    if prevalence_current_dataset:
        plt.axvline(prevalence_current_dataset, color="red", linestyle="--",
                    label=f"Study Dataset ({prevalence_current_dataset:.2%})")

    plt.xscale("log")
    plt.ylim(0, 1.07)
    plt.xlim(prevalence_grid.min(), prevalence_grid.max())
    # Apply scale to font sizes
    plt.xlabel("Prevalence (log scale)", fontsize=12*scale)
    plt.ylabel("Positive Predictive Value (PPV)", fontsize=12*scale)
    plt.title("PPV as a Function of Prevalence", fontsize=14*scale)
    plt.xticks(fontsize=10*scale)
    plt.yticks(fontsize=10*scale)
    plt.grid(alpha=0.7)
    plt.legend(frameon=True, fontsize=8*scale, loc="lower right")
    sns.despine()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path.joinpath("ppv.png"),  dpi=300)
    plt.show()





if __name__ == '__main__':
    # %% Read data
    df_data = pd.read_csv(config.get('data_pre_proc_files').get('anic_okun'))

    # %%
    OVERWRITE = False
    result_dir = config.get('results_path').get('results').joinpath(f'main_cv_complete_train')
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
    # df_metrics_all.to_parquet(master_dir / "metrics_all.parquet", index=False)
    # df_predictions_all.to_parquet(master_dir / "predictions_all.parquet", index=False)

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

    RUN_MOST_SPECIFIC = False
    RUN_MOST_SPECIFIC_BALANCE = True
    if RUN_MOST_SPECIFIC:
        path_most_specific = master_dir.joinpath('most_specific')
        path_most_specific.mkdir(exist_ok=True, parents=True)
        path_most_specific_selected = path_most_specific / 'df_selected_best_sp.csv'
        path_most_specific_model_best = path_most_specific / 'df_per_model_best_sp.csv'
        path_most_specific_model_best_veto = path_most_specific / 'df_per_model_best_sp_veto.csv'
        path_most_specific_selected_prevalence = path_most_specific / 'df_per_model_best_sp_prevalence.csv'
        if not path_most_specific_selected.exists() or not path_most_specific_model_best.exists():
            (df_selected_best_sp,
             df_per_model_best_sp,
             df_metrics_prevalence,
             df_per_model_best_veto) = plot_hla_vs_nonhla_roc_and_cm_with_veto(
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
            df_selected_best_sp.to_csv(path_most_specific_selected, index=False)
            df_per_model_best_sp.to_csv(path_most_specific_model_best, index=False)
            df_per_model_best_veto.to_csv(path_most_specific_selected_prevalence, index=False)

            df_metrics_prevalence['sensitivity'] = df_metrics_prevalence['sensitivity'] * 100
            df_metrics_prevalence['specificity'] = df_metrics_prevalence['specificity'] * 100
            for col in ['sensitivity', 'specificity', 'PPV', 'NPV']:
                df_metrics_prevalence[col] = df_metrics_prevalence[col].round(3)
            df_metrics_prevalence.to_csv(path_most_specific_selected_prevalence, index=False)
            print(
                tabulate(
                    df_metrics_prevalence,
                    headers=list(df_metrics_prevalence.columns),
                    tablefmt='psql'
                )
            )
            plot_ppv_vs_prevalence(df=df_metrics_prevalence,
                                   prevalence_range = (1e-4, 0.5),
                                   figsize=(11, 6),
                                   output_path=path_most_specific,
                                   prevalence_current_dataset=(df_data[target] == 1).sum()/df_data.shape[0],
                                   scale=1.2)

        else:
            df_selected_best_sp = pd.read_csv(path_most_specific_selected)
            df_per_model_best_sp = pd.read_csv(path_most_specific_model_best)

    if RUN_MOST_SPECIFIC_BALANCE:
        # this requiers the sort by function to include the auc and prc so it balance better the Se and Sp
        # in function select_best_model_type_per_config()
        path_spec_but_balance = master_dir.joinpath('spec_but_balance')
        path_spec_but_balance.mkdir(exist_ok=True, parents=True)

        (df_selected_best_bl,
         df_per_model_best_bl,
         df_metrics_prevalence,
         df_per_model_best_bl_veto) = plot_hla_vs_nonhla_roc_and_cm_with_veto(
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
        df_per_model_best_bl.to_csv(path_spec_but_balance / 'df_per_model_best_bl.csv', index=False)
        df_per_model_best_bl.to_csv(path_spec_but_balance / 'df_per_model_best_bl_veto.csv', index=False)
        df_metrics_prevalence.to_csv(path_spec_but_balance / 'df_metrics_prevalence.csv', index=False)

        #
        # # Load Excel file
        # file_path = path_spec_but_balance / 'df_per_model_best_bl_cell_break.xlsx'
        # df = pd.read_excel(file_path)
        #
        #
        # # Function to insert line break before '('
        # def insert_linebreak(text):
        #     if isinstance(text, str):
        #         if "\n" not in text:
        #             return text.replace("(", "\n(")
        #         else:
        #             return text
        #     return text
        #
        #
        # # Apply to all cells
        # df = df.map(insert_linebreak)
        #
        # # Save updated file
        # output_path = path_spec_but_balance / 'df_per_model_best_bl_cell_break.xlsx'
        # df.to_excel(output_path, index=False)
        #
        plot_ppv_vs_prevalence(df=df_metrics_prevalence,
                               prevalence_range=(1e-4, 0.5),
                               figsize=(11, 6),
                               output_path=path_spec_but_balance,
                               prevalence_current_dataset=(df_data[target] == 1).sum() / df_data.shape[0],
                               scale=1.2)


    # %% Feature Importance, XGBOOST
    import pandas as pd
    path = r"C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\CataplexyQuestionnaire\data\pproc\dataset.xlsx"
    df = pd.read_excel(path)
    df = df[df['Included'] == 1]
    path_new = r"C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\CataplexyQuestionnaire\data\pproc\dataset_for_publication.xlsx"
    df.to_excel(path_new)













