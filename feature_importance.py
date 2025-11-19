from config.config import config
import pandas as pd
from library.ml_questionnaire.shap_im_pipeline import (get_data_and_configurations,
                                                       get_available_models,
                                                        normalize_shap_importances,
                                                       compute_shap_importance_from_folds)
import matplotlib as mpl
import matplotlib.pyplot as plt
import textwrap
from typing import Optional, Tuple
import pathlib


def plot_shap_importances_by_config(
        df_plot,
        output_path: Optional[pathlib.Path] = None,
        figsize: Optional[Tuple[int, int]] = None,
):
    """
    Plot normalized SHAP importances for the selected best model of each configuration.
    Each subplot corresponds to one configuration.

    Parameters
    ----------
    df_plot : pd.DataFrame
        Must contain columns: 'Feature', 'Rel |SHAP| (%)', 'config', 'formal_label'
    output_path : pathlib.Path, optional
        Directory to save the plot
    figsize : tuple, optional
        Figure size (width, height)
    """

    # --- Styling ---
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.titlesize'] = 15
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 14

    # --- Get configs ---
    configurations = df_plot["config"].unique()
    n_configs = len(configurations)

    # --- Global max for consistent x-limits ---
    global_max = df_plot["Rel |SHAP| (%)"].max()

    # --- Assign colors per feature ---
    unique_features = df_plot["Feature"].unique()
    cmap = plt.get_cmap("tab20")
    feature_to_color = {feat: cmap(i % cmap.N) for i, feat in enumerate(unique_features)}

    # --- Create subplots ---
    if figsize is None:
        figsize = (4 * n_configs, 8)

    fig, axes = plt.subplots(1, n_configs, figsize=figsize, sharey=True)
    if n_configs == 1:
        axes = [axes]

    for ax, config in zip(axes, configurations):
        df_sub = df_plot[df_plot["config"] == config].copy()

        # Sort features within subplot
        df_sub = df_sub.sort_values("Rel |SHAP| (%)", ascending=True)

        # Color features consistently across subplots
        colors = [feature_to_color[f] for f in df_sub["Feature"]]

        # Plot horizontal bars
        ax.barh(
            df_sub["Feature"],
            df_sub["Rel |SHAP| (%)"],
            color=colors
        )

        # Axis formatting
        ax.set_xlim(0, global_max * 1.1)
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.7)

        # Title = formal label
        label = df_sub["formal_label"].iloc[0]
        ax.set_title(textwrap.fill(label, width=25))

    # Shared labels
    fig.supxlabel("Relative SHAP Importance (%)", fontsize=13)
    fig.supylabel("Feature", fontsize=13)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == '__main__':
    # %% Read inpu data
    df_data = pd.read_csv(config.get('data_pre_proc_files').get('anic_okun'))
    dir_model_path = config.get('results_path').get('results').joinpath(f'main_cv_complete_train')
    results_dir = dir_model_path.joinpath('shap_importance')
    results_dir.mkdir(parents=True, exist_ok=True)
    # %% Define the features used in the model
    path_shap_importances = results_dir / "shap_importances.parquet"

    df, config_feats, feature_set_mapper = get_data_and_configurations(
        path_data=config.get('data_pre_proc_files').get('anic_okun'),
        target='NT1 ICSD3 - TR')

    if not path_shap_importances.exists():
        results_all = []  # collect everything for parquet
        for config_, meta_ in config_feats.items():
            print(f"\n=== Processing configuration: {config_} ===")

            # path to models for this configuration
            models_root = dir_model_path / config_ / "models"
            print(f'\t models root folder: {models_root}')
            assert models_root.exists()

            # path to outer folds file
            folds_path = dir_model_path / config_  / "outer_cv.pkl"
            print(f'\t folds path: {folds_path}')
            assert folds_path.is_file()

            # select features and target
            target = meta_["target"]
            features = [feat for feat in meta_["features"] if feat != "subject_id"]
            X = df[features]
            y = df[target]
            feature_names = features

            # loop over available models
            available_models = get_available_models(models_root)
            for model_name in available_models:
                # model_name = 'xgboost'
                # model_name = 'random_forest'
                print(f"  -> Model: {model_name}")
                model_dir = models_root / model_name
                assert model_dir.exists()
                # loop over optimizations
                for optimization in ["auc", "maxspec", "youden"]:
                    print(f"     Optimization: {optimization}")

                    df_imp = compute_shap_importance_from_folds(
                        model_dir=model_dir,
                        folds_path=folds_path,
                        X=X,
                        y=y,
                        continuous_cols = ['ESS'],
                        feature_names=feature_names,
                        optimization=optimization,
                    )

                    # ---- Option 1: nested dirs (human browsing) ----
                    out_dir = results_dir / config_ / model_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    df_imp.to_csv(out_dir / f"{optimization}_shap.csv", index=False)

                    # ---- Option 2: keep for single parquet (programmatic use) ----
                    df_imp = df_imp.assign(
                        config=config_,
                        model=model_name,
                        optimization=optimization,
                    )
                    results_all.append(df_imp)

        # ---- Save everything once at the end ----
        if results_all:
            df_all = pd.concat(results_all, ignore_index=True)
            parquet_path = results_dir / "shap_importances.parquet"
            df_all.to_parquet(parquet_path, index=False)
            print(f"Saved consolidated results to {parquet_path}")
            df_norm = normalize_shap_importances(df_all)  # normalized shap DataFrame
            df_norm.to_parquet(results_dir / "shap_importances_normalized.parquet")

    else:
        df_all = pd.read_parquet(results_dir / "shap_importances.parquet")
        df_norm = pd.read_parquet(results_dir / "shap_importances_normalized.parquet")

    # %% Normalize shap values to compare
    df_all = pd.read_parquet(results_dir / "shap_importances.parquet")
    df_norm = normalize_shap_importances(df_all)  # normalized shap DataFrame
    df_norm.to_parquet(results_dir/ "shap_importances_normalized.parquet")

    # %% ==================== Importance for best specific
    # for the feature importance let's use the best model for each feature set configuration
    # one per configuration
    df_metrics = results_dir.parents[0].joinpath('master', 'most_specific', 'df_selected_best_sp.csv')
    assert df_metrics.exists()
    df_models = pd.read_csv(df_metrics)

    df_plot = pd.DataFrame()
    for idx, model_ in df_models.iterrows():
        config_ = model_["config"]
        if config_ in ['ess_hla', 'ess']:
            continue

        model_type = model_['model_type']
        optimization = model_['optimization']
        label = model_['feature_set_label']

        df_slice = df_norm.loc[(df_norm.config == config_) &
                               (df_norm.model == model_type) &
                               (df_norm.optimization == optimization), ['Feature', 'Rel |SHAP| (%)', 'config'] ]
        df_slice['formal_label'] = label
        df_plot = pd.concat([df_plot, df_slice])

    plot_shap_importances_by_config(df_plot=df_plot,
                                    output_path=results_dir.joinpath(f'plt_shap_selected_best_sp.png'),
                                    figsize=None)


    plot_shap_importances_by_config(df_plot=df_plot.loc[df_plot['config'].isin(['ukbb_hla', 'ukbb'])],
                                    output_path=results_dir.joinpath(f'plt_shap_selected_best_asbtrct.png'),
                                    figsize=None)

    # %% ==================== Importance for best balanced
    df_metrics = results_dir.parents[0].joinpath('master', 'spec_but_balance', 'df_selected_best_bl.csv')
    assert df_metrics.exists()
    df_models = pd.read_csv(df_metrics)

    df_plot = pd.DataFrame()
    for idx, model_ in df_models.iterrows():
        config_ = model_["config"]
        if config_ in ['ess_hla', 'ess']:
            continue
        model_type = model_['model_type']
        optimization = model_['optimization']
        label = model_['feature_set_label']

        df_slice = df_norm.loc[(df_norm.config == config_) &
                               (df_norm.model == model_type) &
                               (df_norm.optimization == optimization), ['Feature', 'Rel |SHAP| (%)', 'config'] ]
        df_slice['formal_label'] = label
        df_plot = pd.concat([df_plot, df_slice])

    plot_shap_importances_by_config(df_plot=df_plot,
                                    output_path=results_dir.joinpath(f'plt_shap_selected_best_bl'),
                                    figsize=None)



















