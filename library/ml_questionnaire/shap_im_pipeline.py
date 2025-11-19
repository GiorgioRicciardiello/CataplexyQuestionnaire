import shap, joblib, pickle
import numpy as np
import pandas as pd
from config.config import config
from typing import Optional, Tuple, List
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def infer_model_type(folder_name: str) -> str:
    """
    Infer the model family based on folder name.
    Returns one of: "tree", "linear", "kernel".
    """
    tree_models = {"random_forest", "lightgbm", "xgboost"}
    linear_models = {"logistic_regression", "elastic_net"}
    kernel_models = {"SVM", "LDA"}  # anything not tree/linear, fallback to kernel

    if folder_name in tree_models:
        return "tree"
    elif folder_name in linear_models:
        return "linear"
    elif folder_name in kernel_models:
        return "kernel"
    else:
        return "kernel"  # safe fallback

def get_available_models(models_root: Path) -> List[str]:
    """
    List all available model types based on folder names inside models/.
    """
    return [f.name for f in models_root.iterdir() if f.is_dir()]


def normalize_shap_importances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-hoc normalize SHAP importances within each (config, model, optimization).
    Adds relative importance (%) and feature ranking.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe of SHAP importances with at least:
        ['Feature', 'Mean |SHAP|', 'Std', 'config', 'model', 'optimization']

    Returns
    -------
    pd.DataFrame
        Input dataframe with two new columns:
        - 'Rel |SHAP| (%)': feature importance normalized to 100% within each group
        - 'Rank': feature rank within each group (1 = most important)
    """
    df_norm = df.copy()

    # normalize within each (config, model, optimization)
    df_norm["Rel |SHAP| (%)"] = (
        df_norm.groupby(["config", "model", "optimization"])["Mean |SHAP|"]
        .transform(lambda x: 100 * x / x.sum())
    )

    # rank features (1 = top)
    df_norm["Rank"] = (
        df_norm.groupby(["config", "model", "optimization"])["Mean |SHAP|"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return df_norm



def compute_shap_importance_from_folds(
        model_dir: Path,
        folds_path: Path,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        continuous_cols: List[str] = None,
        optimization: str = "auc",
        n_jobs: int = -1,
        max_samples: int = None
    ) -> pd.DataFrame:
    """
    Compute mean ± std SHAP importance across outer folds for one optimization.
    Parallelized across folds with progress bar.
    Missing values imputed using training medians.
    Continuous variables standardized using training mean/std.
    """
    with open(folds_path, "rb") as f:
        folds = pickle.load(f)

    def _compute_for_fold(i, train_idx, test_idx):
        model_path = model_dir / f"outer{i}_{optimization}.pkl"
        model = joblib.load(model_path)

        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx].copy(), y.iloc[test_idx]

        # --- Median imputation (train stats only) ---
        medians = X_train.median(numeric_only=True)
        X_train[medians.index] = X_train[medians.index].fillna(medians)
        X_test[medians.index] = X_test[medians.index].fillna(medians)

        # --- Standardize continuous columns (train stats only) ---
        if continuous_cols:
            scaler = StandardScaler()
            X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
            X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

        # --- Optional subsampling for test set ---
        if max_samples is not None and len(X_test) > max_samples:
            X_test = X_test.sample(n=max_samples, random_state=0)
            y_test = y_test.loc[X_test.index]

        # --- Pick explainer ---
        folder_name = model_dir.name
        model_type = infer_model_type(folder_name)

        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X_test)
        else:  # kernel models like SVM, LDA
            background = shap.sample(X_train, 50, random_state=0)  # smaller background
            explainer = shap.KernelExplainer(model.predict_proba, background, link="logit")

        # --- SHAP values ---
        shap_values = explainer.shap_values(X_test)
        shap_values = np.array(shap_values)

        if shap_values.ndim == 2:        # regression
            shap_for_use = shap_values
        elif shap_values.ndim == 3:      # binary/multiclass
            shap_for_use = shap_values[:, :, 1]  # positive class only
        else:
            raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")

        # --- Aggregate feature importances ---
        mean_abs_shap = np.abs(shap_for_use).mean(axis=0)
        mean_abs_shap = np.nan_to_num(mean_abs_shap, nan=0.0)

        return mean_abs_shap

    # Run in parallel across folds
    all_importances = Parallel(n_jobs=n_jobs)(
        delayed(_compute_for_fold)(i, train_idx, test_idx)
        for i, (train_idx, test_idx) in tqdm(
            enumerate(folds, start=1),
            total=len(folds),
            desc=f"Computing SHAP ({optimization})"
        )
    )

    arr = np.vstack(all_importances)  # (n_folds, n_features)

    return pd.DataFrame({
        "Feature": feature_names,
        "Mean |SHAP|": arr.mean(axis=0),
        "Std": arr.std(axis=0)
    }).sort_values("Mean |SHAP|", ascending=False)


def get_data_and_configurations(path_data:Path, target:str= 'NT1 ICSD3 - TR'):
    df_data = pd.read_csv(path_data)
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

    col_ukbb = ['BMI', 'ESS', 'JAW', 'HEAD', 'HAND', 'KNEES',
                'SPEECH', 'JOKING', 'LAUGHING', 'ANGER',
                'QUICKVERBAL']

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
            'features': ['ESS'],
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
    # use the formal names only
    # configurations = {
    #     feature_set_mapper.get(key): val
    #     for key, val in configurations.items()
    # }

    return df_data, configurations, feature_set_mapper




def plot_feature_importance_comparison(df_compare: pd.DataFrame,
                                       feature_sets: list,
                                       title: str = "Feature Importance Comparison",
                                       figsize: tuple = (14, 8),
                                       darken_for_reduced: list = None):
    """
    Plot side-by-side feature importance across multiple feature sets.

    Parameters
    ----------
    df_compare : pd.DataFrame
        Wide dataframe: Feature + one column per feature set (importance values).
    feature_sets : list
        Ordered list of feature set names (columns in df_compare).
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    darken_for_reduced : list
        List of feature set names that represent reduced sets (will darken colors for features unique to them).
    """

    df_long = df_compare.melt(id_vars="Feature", value_vars=feature_sets,
                              var_name="FeatureSet", value_name="Importance")

    # consistent color per feature
    palette = sns.color_palette("tab20", n_colors=df_long["Feature"].nunique())
    feature_colors = {feat: palette[i % len(palette)]
                      for i, feat in enumerate(sorted(df_long["Feature"].unique()))}

    # optional darkening for features only in reduced sets
    if darken_for_reduced:
        reduced_feats = set()
        for fs in darken_for_reduced:
            feats_in_fs = df_compare.loc[df_compare[fs].notna(), "Feature"].tolist()
            reduced_feats.update(feats_in_fs)

        def adjust_color(c, factor=0.6):
            return tuple(np.array(c) * factor)

        for f in reduced_feats:
            feature_colors[f] = adjust_color(feature_colors[f])

    # sort features by average importance across sets
    avg_importance = (
        df_long.groupby("Feature")["Importance"].mean().sort_values(ascending=True)
    )
    sorted_features = avg_importance.index.tolist()

    # plotting
    fig, axes = plt.subplots(1, len(feature_sets), figsize=figsize, sharey=True)
    if len(feature_sets) == 1:
        axes = [axes]

    for ax, fs in zip(axes, feature_sets):
        sub = df_long[df_long["FeatureSet"] == fs]
        sub = sub.set_index("Feature").loc[sorted_features].reset_index()

        bars = ax.barh(sub["Feature"], sub["Importance"],
                       color=[feature_colors[f] for f in sub["Feature"]])

        ax.set_title(fs)
        ax.set_xlabel("Relative SHAP importance (%)")

    axes[0].set_ylabel("Feature")
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
