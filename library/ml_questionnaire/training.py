"""
Nested Cross-Validation with Optuna Hyperparameter Optimization

This script runs nested cross-validation with multiple model types,
scoring strategies, and Optuna for inner-loop hyperparameter search.
Outputs include metrics CSVs, per-sample classification CSVs, and figures.

"""
from typing import Sequence, Tuple, Union, Optional, List, Dict, Any, Callable
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from collections.abc import Iterable
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import joblib
import pickle
from tabulate import tabulate
from library.ml_questionnaire.scoring import (youdens_j_from_scores,
                                              metrics_at_threshold,
                                              compute_ci_from_folds_average)
from library.ml_questionnaire.scoring import generate_ci
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from pandas import DataFrame
from sklearn.base import clone
from typing import Optional, List, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from scipy import stats


# %% XGBoost optimization for a singular metrics
class XGBSensWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, params: dict, alpha: float = 2.0, num_boost_round: int = 200):
        self.params = params
        self.alpha = alpha
        self.num_boost_round = num_boost_round
        self.booster = None
        self.feature_names = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        self.feature_names = (
            X.columns.tolist() if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        )
        # copy params and strip out n_estimators
        train_params = {k: v for k, v in self.params.items() if k != "n_estimators"}

        self.booster = xgb.train(
            train_params,
            dtrain,
            num_boost_round=self.num_boost_round,   # use it here
            obj=lambda preds, dtrain: self.sensitivity_loss(preds, dtrain, alpha=self.alpha)
        )
        return self

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        preds = self.booster.predict(dtest)
        return np.column_stack([1 - preds, preds])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {
            "params": self.params,
            "alpha": self.alpha,
            "num_boost_round": self.num_boost_round
        }

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    @staticmethod
    def sensitivity_loss(preds, dtrain, alpha: float = 2.0):
        """
        Custom loss to emphasize sensitivity (recall for positives).
        Penalizes false negatives alpha× more strongly.
        """
        labels = dtrain.get_label()
        probs = 1.0 / (1.0 + np.exp(-preds))  # sigmoid

        grad = np.where(labels == 1,
                        -alpha * (1.0 - probs),  # positives weighted alpha×
                        probs)
        hess = np.where(labels == 1,
                        alpha * probs * (1.0 - probs),
                        probs * (1.0 - probs))
        return grad, hess

class XGBSpecWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, params: dict, beta: float = 2.0, num_boost_round: int = 200):
        self.params = params
        self.beta = beta
        self.num_boost_round = num_boost_round
        self.booster = None
        self.feature_names = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        self.feature_names = (
            X.columns.tolist() if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
        )

        train_params = {k: v for k, v in self.params.items() if k != "n_estimators"}

        self.booster = xgb.train(
            train_params,
            dtrain,
            num_boost_round=self.num_boost_round,
            obj=lambda preds, dtrain: self.specificity_loss(preds, dtrain, beta=self.beta)
        )
        return self

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        preds = self.booster.predict(dtest)
        return np.column_stack([1 - preds, preds])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {
            "params": self.params,
            "beta": self.beta,
            "num_boost_round": self.num_boost_round
        }

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    @staticmethod
    def specificity_loss(preds, dtrain, beta: float = 2.0):
        """
        Custom loss to emphasize specificity (true negatives).
        Penalizes false positives beta× more strongly.
        """
        labels = dtrain.get_label()
        probs = 1.0 / (1.0 + np.exp(-preds))  # sigmoid

        grad = np.where(labels == 0,
                        beta * probs,  # negatives penalized beta× if misclassified
                        -(1.0 - probs))
        hess = np.where(labels == 0,
                        beta * probs * (1.0 - probs),
                        probs * (1.0 - probs))
        return grad, hess


#%% helpers
def _get_model_and_space(model_type,
                         trial,
                         pos_weight_ratio: float | None = None,
                         random_seed:Optional[int]=42):
    """Return (estimator, params_dict_used) for a given model_type using Optuna trial."""
    params = {}
    if model_type == "random_forest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),  # smaller range
            "max_depth": trial.suggest_int("max_depth", 2, 5),  # shallow trees
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"])
        }
        model = RandomForestClassifier(**params, random_state=random_seed, n_jobs=-1)

    elif model_type == "lightgbm":
        low  = (pos_weight_ratio or 1.0) * 0.8
        high = (pos_weight_ratio or 1.0) * 1.2
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 4, 15),  # lower because few unique values
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, 15),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),  # almost full data
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight",low,high)
        }
        # params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 20.0)

        model = lgb.LGBMClassifier(**params,
                                   random_state=random_seed,
                                   verbose=-1,
                                   # device="gpu",  # <- use GPU
                                   # gpu_platform_id=0,  # optional: select OpenCL platform
                                   # gpu_device_id=0  # optional: select GPU device

                                   )

    elif model_type == "xgboost":
        low  = (pos_weight_ratio or 1.0) * 0.8
        high = (pos_weight_ratio or 1.0) * 1.2
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "n_estimators": trial.suggest_int("n_estimators", 10, 400),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", low, high)
        }
        model = xgb.XGBClassifier(**params,
                                  random_state=random_seed,
                                  eval_metric="logloss",
                                  device="cuda",
                                  predictor="gpu_predictor",
                                  verbosity=0)
    elif model_type == "elastic_net":
        params = {
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),  # 0=L2, 1=L1
            "class_weight": trial.suggest_categorical(
                                    "class_weight",
                                    [None,
                                     "balanced",
                                     ]
                                )
        }
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",             # only saga supports elasticnet
            max_iter=5000,             # ensure convergence
            random_state=random_seed,
            n_jobs=-1,
            **params
        )

    elif model_type == "logistic_regression":
        params = {
            "C": 1.0,
            "penalty": None,
            "solver": "lbfgs"
        }
        model = LogisticRegression(penalty=None,  # Model (b)
                           solver=params.get('solver'),
                           max_iter=1000,
                           C=params.get('C'))

    elif model_type == "SVM":
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly", "sigmoid"])
        params = {
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
            "probability": True,
        }
        if kernel == "rbf":
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
        elif kernel == "linear":
            pass  # no extra params
        elif kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            params["coef0"] = trial.suggest_float("coef0", 0.0, 1.0)
        elif kernel == "sigmoid":
            params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
            params["coef0"] = trial.suggest_float("coef0", 0.0, 1.0)
        model = SVC(kernel=kernel, **params)

    elif model_type == "LDA":
        # Enable shrinkage (regularized LDA) via non-SVD solvers
        solver = trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"])
        params = {"solver": solver, "tol": trial.suggest_float("tol", 1e-6, 1e-2, log=True)}
        if solver in ("lsqr", "eigen"):
            # shrinkage may be 'auto' or a float (0,1)
            use_auto = trial.suggest_categorical("use_auto_shrinkage", [True, False])
            if use_auto:
                params["shrinkage"] = "auto"
            else:
                params["shrinkage"] = trial.suggest_float("shrinkage", 0.0 + 1e-6, 1.0 - 1e-6)
        model = LinearDiscriminantAnalysis(**params)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, params


def _pos_weight_ratio(y: np.ndarray) -> float:
    # n_neg / n_pos (guard against division by zero)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    return float(n_neg / max(n_pos, 1))

def _save_model_and_importances(model,
                               model_type: str,
                                scoring_key: str,
                               feature_names,
                               results_dir: Path,
                               fold_name: str):
    """
    Save trained model and its feature importance to disk.

    results_dir/
    └── model_performance/
        ├── random_forest/
        │   ├── auc/
        │   │   ├── models/
        │   │   │   ├── outer1.pkl
        │   │   │   ├── outer2.pkl
        │   │   └── feature_importances/
        │   │       ├── outer1.csv
        │   │       └── outer2.csv
        │   │   └── random_forest_auc_summary.csv
        │   ├── youden/
        │   └── ...
        ├── lightgbm/
        │   ├── auc/
        │   └── ...
        └── ...

    Parameters
    ----------
    model : fitted estimator
        The trained model.
    model_type : str
        One of ["random_forest", "lightgbm", "xgboost", "elastic_net"].
    feature_names : list[str]
        Names of input features.
    results_dir : Path
        Root directory for experiment results.
    fold_name : str
        Identifier for the current fold (e.g., "outer1_inner3").
    """
    base_dir = Path(results_dir) / "model_performance" / model_type / scoring_key
    models_dir = base_dir / "models"
    feats_dir = base_dir / "feature_importances"
    models_dir.mkdir(parents=True, exist_ok=True)
    feats_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_dir / f"{fold_name}.pkl"
    joblib.dump(model, model_path)

    # Extract feature importances
    if model_type in ["random_forest", "lightgbm", "xgboost"]:
        importances = model.feature_importances_
    elif model_type == "elastic_net":
        importances = model.coef_.ravel()

    elif model_type == "xgboost_sens":
        # booster-based importance
        score_dict = model.booster.get_score(importance_type="gain")  # or "weight"
        # convert to aligned array with feature order
        importances = np.array([score_dict.get(f, 0.0) for f in feature_names])

    else:
        raise ValueError(f"Feature importances not implemented for {model_type}")

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    imp_path = feats_dir / f"{fold_name}_feature_importance.csv"
    df_imp.to_csv(imp_path, index=False)


def _sensitivity_score(y_true, y_score, min_spec: float = 0.6) -> Tuple[float, float]:
    """
    Compute the maximum sensitivity achievable at or above a minimum specificity.
    min_spec : float, default=0.6 Minimum specificity required.
    Returns (best_sens, best_tau).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1 - fpr

    # Filter thresholds where specificity >= min_spec
    valid = np.where(spec >= min_spec)[0]
    if len(valid) == 0:
        return 0.0, 0.5  # no feasible threshold

    # Among valid thresholds, pick one with maximum sensitivity
    best_idx = valid[np.argmax(tpr[valid])]
    return tpr[best_idx], thresholds[best_idx]


def _best_sens_or_spec_thr(y_true,
                           y_score,
                           min_sens: float = 0.7,
                           min_spec: float = 0.6,
                           maximize: str = "spec"):
    """
    Flexible threshold finder.

    maximize="spec" → maximize specificity subject to min sensitivity
    maximize="sens" → maximize sensitivity subject to min specificity
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    spec = 1 - fpr
    sens = tpr

    if maximize == "spec":
        valid = np.where(sens >= min_sens)[0]
        if len(valid) == 0:
            return 0.0, 0.0, 0.5
        best_idx = valid[np.argmax(spec[valid])]
    elif maximize == "sens":
        valid = np.where(spec >= min_spec)[0]
        if len(valid) == 0:
            return 0.0, 0.0, 0.5
        best_idx = valid[np.argmax(sens[valid])]
    else:
        raise ValueError("maximize must be either 'spec' or 'sens'")
    return spec[best_idx], sens[best_idx], thresholds[best_idx]

def _save_or_load_folds(path: Path, cv, X, y, random_seed=42):
    """
    Save folds (train/test indices) if not present; load them otherwise.
    Returns list of (train_idx, test_idx).
    """
    path = Path(path)
    if path.exists():
        with open(path, "rb") as f:
            folds = pickle.load(f)
        print(f"Loaded folds from {path}")
    else:
        folds = list(cv.split(X, y))
        with open(path, "wb") as f:
            pickle.dump(folds, f)
        print(f"Saved folds to {path}")
    return folds


# %% Sort results
def aggregate_feature_importances(model_type: str,
                            scoring_key: str,
                                  results_dir: Path,
                                  force: bool = False):
    """
    Aggregate feature importance CSVs across folds into mean + std summary.
    """
    base_dir = Path(results_dir) / "model_performance" / model_type / scoring_key
    feats_dir = base_dir / "feature_importances"
    summary_path = base_dir / f"{model_type}_{scoring_key}_feature_importance_summary.csv"

    # if summary_path.exists() and not force:
    #     print(f"Summary already exists: {summary_path}")
    #     return summary_path, pd.read_csv(summary_path)

    files = list(feats_dir.glob("*_feature_importance.csv"))
    if not files:
        raise FileNotFoundError(f"No feature importance CSVs found in {feats_dir}")

    dfs = [pd.read_csv(f) for f in files]
    df_all = pd.concat(dfs, axis=0)

    summary = (
        df_all.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    print(f'Summary saved to {summary_path}')

    return summary_path, summary

def _save_model(model,
                feature_names,
                results_dir: Path,
                model_type: str,
                opt_type: str,
                outer_idx: int):
    model_dir = results_dir / "models" / model_type / f"outer{outer_idx:02d}"
    model_dir.mkdir(parents=True, exist_ok=True)

    path = model_dir / f"{opt_type}.pkl"
    payload = {
        "model": model,
        "feature_names": feature_names
    }
    joblib.dump(payload, path)
    print(f"Saved model + features → {path}")
    return path

# %% Main called
def run_nested_cv_with_optuna_parallel(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    col_id: str,
    results_dir: Path,
    continuous_cols: Optional[List[str]] = None,
    model_types: Optional[List[str]] = None,
    random_seed: int = 42,
    procerssor: bool = True,
    n_outer_splits: int = 10,
    n_inner_splits: int = 5,
    n_trials: int = 50,
    study_sampler=None,
    pos_weight: bool = False,
    min_sens: float = 0.6,
    min_spec: float = 0.6,
    maximize: str = "spec",                       # "spec" or "sens"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nested CV with Optuna hyperparameter tuning.

    Preprocessing policy:
      - Impute ALL `feature_cols` that may have NaNs.
      - Standardize ONLY the `continuous_cols` subset.
      - Keep ALL features (no column dropping).
      - Return transformed X with columns in the SAME ORDER as `feature_cols`.

    Implementation details:
      - Build a *template* ColumnTransformer; CLONE & FIT it per split to avoid leakage.
      - Keep X as DataFrame so ColumnTransformer can select columns by name.
      - Cache preprocessed inner folds once per outer split for Optuna trials (speed).
      - Compute thresholds from inner OOF predictions for the chosen best trial for each objective;
        then evaluate on the outer test fold.
      - Fail fast if any NaNs remain post-transform.
    """

    # -------------------------------
    # utilities
    # -------------------------------
    def _assert_no_nans(arr: np.ndarray, where: str):
        n = np.isnan(arr).sum()
        if n > 0:
            raise ValueError(f"NaNs remain after preprocessing in {where}: {n} NaNs")

    def _normalize_cols(cols, all_cols: Iterable[str]) -> Optional[List[str]]:
        """Normalize/validate a list of column names. Returns None if cols is None."""
        if cols is None:
            return None
        if isinstance(cols, (str, np.str_)):  # single name → list
            cols = [cols]
        cols = list(cols)
        missing = set(cols) - set(all_cols)
        if missing:
            raise ValueError(f"Unknown columns in continuous_cols: {sorted(missing)}")
        return cols

    def _make_preprocessor(all_features: List[str], cont_cols: Optional[List[str]]) -> ColumnTransformer:
        """
        Build a ColumnTransformer that:
          - imputes + scales ONLY `cont_cols` (if provided),
          - imputes the remaining `all_features` (no scaling),
          - keeps all features in the output (remainder='passthrough'),
          - uses concise feature names (no prefixes) when possible.
        NOTE: Ensure all features are numeric by this point (encode categoricals upstream),
              or split `other_cols` into numeric/categorical and add an encoder if needed.
        """
        numeric_cols = cont_cols or []  # columns to scale
        other_cols = [c for c in all_features if c not in numeric_cols]

        transformers = []
        if numeric_cols:
            transformers.append((
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numeric_cols
            ))
        if other_cols:
            transformers.append((
                "other",
                SimpleImputer(strategy="most_frequent"),
                other_cols
            ))

        ct = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False  # try to keep original names for 1-to-1 transforms
        )
        # If sklearn supports DataFrame output, enable it (helps naming/dtypes)
        try:
            ct.set_output(transform="pandas")
        except Exception:
            pass
        return ct

    def _compute_tau_youden(inner_df: pd.DataFrame, col_true="y_true", col_score="y_score"):
        if inner_df.empty:
            return 0.0, 0.5
        return youdens_j_from_scores(inner_df[col_true].values, inner_df[col_score].values)


    def _extract_params_per_run(df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep exactly one params row per (outer_fold, model_type, optimization)
        :param df:
        :return:
        """
        # Make threshold ordering deterministic (any '_max' last)
        thr_order = {"youden": 0, "0p5": 1}

        def _thr_rank(s: str) -> int:
            if s in thr_order: return thr_order[s]
            return 2 if str(s).endswith("_max") else 9

        d = df.copy()
        d["_thr_rank"] = d["threshold"].map(_thr_rank)

        out = (
            d.sort_values(["outer_fold", "model_type", "optimization", "_thr_rank"])
            .drop_duplicates(subset=["outer_fold", "model_type", "optimization"], keep="first")
            [["outer_fold", "model_type", "optimization", "best_params_json"]]
            .reset_index(drop=True)
        )
        return out

    def _mk_row(threshold_name:str, threshold_value, met_dict:Dict[str, float]):
        """
        Metrics in the long from
        :param threshold_name:
        :param threshold_value:
        :param met_dict:
        :return:
        """
        return {
            "outer_fold": outer_idx,
            "model_type": model_type,
            "optimization": opt_type,
            "threshold": threshold_name,  # 'youden' | '0p5' | 'spec_max'/'sens_max'
            "threshold_value": float(threshold_value),
            # core metrics (keep names unsuffixed)
            "auc_score": met_dict.get("auc_score", np.nan),
            "prc_score": met_dict.get("prc_score", np.nan),
            "sensitivity": met_dict.get("sensitivity", np.nan),
            "specificity": met_dict.get("specificity", np.nan),
            # keep per-trial params for traceability (optional)
            "best_params_json": json.dumps(params),
        }
    # -------------------------------
    # helper: fit on train, transform valid/test, and REORDER to original feature order
    # -------------------------------
    def _fit_transform_split(
        X_tr_df: pd.DataFrame,
        y_tr_ser: pd.Series,
        X_va_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        if preprocessor_template is None:
            # No preprocessing requested; already in original order
            return X_tr_df.to_numpy(), X_va_df.to_numpy(), list(X_tr_df.columns)

        preproc = clone(preprocessor_template)

        # Fit on TRAIN (imputer/scaler learn only from training fold)
        X_tr_tx = preproc.fit_transform(X_tr_df, y_tr_ser)
        X_va_tx = preproc.transform(X_va_df)

        # Normalize to DataFrames with names
        if isinstance(X_tr_tx, pd.DataFrame):
            X_tr_df_tx = X_tr_tx.copy()
            X_va_df_tx = X_va_tx.copy()
        else:
            # Older sklearn: build DataFrames using feature names
            try:
                feat_names_all = list(preproc.get_feature_names_out(input_features=X_tr_df.columns))
            except Exception:
                feat_names_all = list(X_tr_df.columns)  # best-effort
            X_tr_df_tx = pd.DataFrame(X_tr_tx, index=X_tr_df.index, columns=feat_names_all)
            X_va_df_tx = pd.DataFrame(X_va_tx, index=X_va_df.index, columns=feat_names_all)

        # Ensure no NaNs remain
        _assert_no_nans(X_tr_df_tx.to_numpy(), "train")
        _assert_no_nans(X_va_df_tx.to_numpy(), "valid/test")

        # REORDER to the original feature order
        missing_in_tx = [c for c in feature_cols if c not in X_tr_df_tx.columns]
        if missing_in_tx:
            # With verbose_feature_names_out=False and 1-to-1 transforms, names should match;
            # if not, show what we actually have for debugging.
            raise ValueError(
                f"Transformed features missing expected columns: {missing_in_tx}. "
                f"Available: {list(X_tr_df_tx.columns)}"
            )
        X_tr_df_tx = X_tr_df_tx[feature_cols]
        X_va_df_tx = X_va_df_tx[feature_cols]

        # final sanity checks
        if X_tr_df_tx.shape[1] != expected_out_dim or X_va_df_tx.shape[1] != expected_out_dim:
            raise ValueError(
                f"Unexpected transformed width. Got train {X_tr_df_tx.shape[1]}, "
                f"valid {X_va_df_tx.shape[1]}, expected {expected_out_dim}."
            )

        return X_tr_df_tx.to_numpy(), X_va_df_tx.to_numpy(), list(feature_cols)

    # -------------------------------
    # helper: Optuna with cached inner folds (preprocessed once per outer split)
    # -------------------------------
    def _run_optuna_with_cache(
        model_type: str,
        X_tr_df: pd.DataFrame, y_tr_ser: pd.Series,
        tr_idx: np.ndarray, outer_idx: int
    ) -> Tuple[optuna.Study, Callable[[Dict[str, Any]], pd.DataFrame]]:
        """
        Returns:
          - study: the optuna study after optimization
          - inner_oof_for_trial(params): function that recomputes concatenated inner OOF predictions
                                         for a fixed trial (used to compute thresholds per trial)
        """
        # Build/cache inner folds ONCE per outer split (fit preproc on inner-train only)
        # Each item: (Xt_proc, Xv_proc, yt_np, yv_np, inner_va_indices)
        fold_cache: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for inner_idx, (inner_tr, inner_va) in enumerate(inner_cv.split(X_tr_df, y_tr_ser), start=1):
            Xt = X_tr_df.iloc[inner_tr]
            Xv = X_tr_df.iloc[inner_va]
            yt = y_tr_ser.iloc[inner_tr]
            yv = y_tr_ser.iloc[inner_va]

            Xt_proc, Xv_proc, _ = _fit_transform_split(Xt, yt, Xv)
            fold_cache.append((Xt_proc, Xv_proc, yt.to_numpy(), yv.to_numpy(), inner_va))

        pos_weight_ratio = _pos_weight_ratio(y_tr_ser) if pos_weight else None

        def objective(trial):
            model, _ = _get_model_and_space(model_type, trial, pos_weight_ratio=pos_weight_ratio)
            scores_auc, scores_youden, sens_spec = [], [], []

            for (Xt_proc, Xv_proc, yt_np, yv_np, _) in fold_cache:
                mdl = clone(model).fit(Xt_proc, yt_np)
                if hasattr(mdl, "predict_proba"):
                    y_score = mdl.predict_proba(Xv_proc)[:, 1]
                elif hasattr(mdl, "decision_function"):
                    y_score = mdl.decision_function(Xv_proc)
                else:
                    y_score = mdl.predict(Xv_proc).astype(float)

                scores_auc.append(roc_auc_score(yv_np, y_score))
                j, _ = youdens_j_from_scores(yv_np, y_score)
                scores_youden.append(j)
                spec, sens, _ = _best_sens_or_spec_thr(yv_np, y_score, min_spec=min_spec)
                sens_spec.append(sens if maximize == "sens" else spec)

            return (np.mean(scores_auc), np.mean(scores_youden), np.mean(sens_spec))

        # -------------------------
        # Model that do not need optimization
        # -------------------------
        fixed_models = {"logistic_regression"}
        this_n_trials = 1 if model_type in fixed_models else n_trials

        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            sampler=study_sampler
        )
        study.optimize(objective, n_trials=this_n_trials, n_jobs=-1, show_progress_bar=True)

        # helper: concatenated inner OOF predictions for a fixed trial
        def inner_oof_for_trial(trial_params: Dict[str, Any]) -> pd.DataFrame:
            model, _ = _get_model_and_space(
                model_type=model_type,
                trial=optuna.trial.FixedTrial(trial_params),
                pos_weight_ratio=pos_weight_ratio
            )
            y_true_all, y_score_all, subj_ids_all = [], [], []
            for (Xt_proc, Xv_proc, yt_np, yv_np, inner_va) in fold_cache:
                mdl = clone(model).fit(Xt_proc, yt_np)
                if hasattr(mdl, "predict_proba"):
                    y_score = mdl.predict_proba(Xv_proc)[:, 1]
                elif hasattr(mdl, "decision_function"):
                    y_score = mdl.decision_function(Xv_proc)
                else:
                    y_score = mdl.predict(Xv_proc).astype(float)

                y_true_all.append(yv_np.astype(int))
                y_score_all.append(y_score.astype(float))
                subj_ids_all.append(ids_np[tr_idx][inner_va])

            return pd.DataFrame({
                "subject_id": np.concatenate(subj_ids_all),
                "y_true":     np.concatenate(y_true_all),
                "y_score":    np.concatenate(y_score_all),
            })

        # For traceability, store inner OOF for the AUC-best trial
        best_auc_trial = max(study.best_trials, key=lambda t: t.values[0])
        df_trace = inner_oof_for_trial(best_auc_trial.params).rename(
            columns={"y_true": "y_val_true", "y_score": "y_val_score"}
        )
        df_trace["outer_fold"] = outer_idx
        df_trace["model_type"] = model_type
        nonlocal df_inner_val_records
        df_inner_val_records = pd.concat([df_inner_val_records, df_trace], ignore_index=True)

        return study, inner_oof_for_trial

    # -------------------------------
    # setup
    # -------------------------------
    if model_types is None:
        model_types = ["random_forest", "lightgbm", "xgboost", "elastic_net", "logistic_regression", "SVM", "LDA"]

    # Remove id from features if present
    feature_cols = [c for c in feature_cols if c != col_id]

    results_dir.mkdir(parents=True, exist_ok=True)

    # Keep as DataFrame/Series to preserve column names for ColumnTransformer
    X = df[feature_cols]
    y = df[target_col]
    ids_np = df[col_id].to_numpy()

    # Build preprocessor *template* (structure only). We'll clone & fit per split.
    preprocessor_template = None
    if procerssor:
        cont_cols = _normalize_cols(continuous_cols, X.columns)  # may be None
        preprocessor_template = _make_preprocessor(all_features=feature_cols, cont_cols=cont_cols)

    # We expect to keep ALL feature columns in the output (no dropping)
    expected_out_dim = len(feature_cols)

    # CV splitters
    outer_cv = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_seed)
    _ = _save_or_load_folds(results_dir / "outer_cv.pkl", outer_cv, X.to_numpy(), y.to_numpy())

    inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_seed)

    # outputs
    df_outer_metrics_records = pd.DataFrame()
    df_outer_predictions_records = pd.DataFrame()
    df_inner_val_records = pd.DataFrame()  # traceability: inner-CV predictions
    df_pretty_records = pd.DataFrame()


    # ------ define directory
    metrics_dir = results_dir / "metrics"
    preds_dir = results_dir / "predictions"
    params_dir = results_dir / "params"
    logs_dir = results_dir / "logs"
    models_dir = results_dir / "models"
    for d in [metrics_dir, preds_dir, params_dir, logs_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)
    (params_dir / "best_params").mkdir(parents=True, exist_ok=True)
    # -------------------------------
    # Outer CV
    # -------------------------------
    for model_type in model_types:
        for outer_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y), start=1):
            print(f"\n=== Outer {outer_idx}/{n_outer_splits} | {model_type}  | Feature Set k={len(feature_cols)} ===")
            X_tr = X.iloc[tr_idx]
            X_te = X.iloc[te_idx]
            y_tr = y.iloc[tr_idx]
            y_te = y.iloc[te_idx]
            ids_te = ids_np[te_idx]

            # ---- Optuna on inner CV (with cached preprocessed folds)
            study, inner_oof_for_trial = _run_optuna_with_cache(
                model_type=model_type,
                X_tr_df=X_tr, y_tr_ser=y_tr,
                tr_idx=tr_idx, outer_idx=outer_idx
            )

            # pick best trials (one per objective)
            best_trials = {
                "auc": max(study.best_trials, key=lambda t: t.values[0]),
                "youden": max(study.best_trials, key=lambda t: t.values[1]),
                f"max{maximize}": max(study.best_trials, key=lambda t: t.values[2]),
            }

            # ---- Evaluate each chosen trial on the outer test fold
            for n_scoring, (opt_type, trial) in enumerate(best_trials.items(), start=0):
                params = trial.params
                print(f"[Outer {outer_idx}] {model_type} | Best {opt_type}: {trial.values[n_scoring]:2f}")
                print("Params:")
                for k, v in params.items():
                    print(f"  - {k}: {v}")

                # thresholds computed from inner OOF predictions for THIS trial
                inner_preds_df = inner_oof_for_trial(params)
                youden_score, youden_threshold = _compute_tau_youden(inner_preds_df, "y_true", "y_score")
                _, _, thr_sens_spec_max = _best_sens_or_spec_thr(
                    y_true=inner_preds_df["y_true"],
                    y_score=inner_preds_df["y_score"],
                    min_sens=min_sens,
                    min_spec=min_spec,
                    maximize=maximize,
                )

                # train best model on full outer-train with a fresh fit/transform
                best_model, _ = _get_model_and_space(
                    model_type=model_type,
                    trial=optuna.trial.FixedTrial(params),
                    pos_weight_ratio=_pos_weight_ratio(y_tr) if pos_weight else None,
                )

                X_tr_proc, X_te_proc, feature_names = _fit_transform_split(X_tr, y_tr, X_te)
                best_model.fit(X_tr_proc, y_tr.to_numpy())

                # predict on outer test
                if hasattr(best_model, "predict_proba"):
                    y_score_te = best_model.predict_proba(X_te_proc)[:, 1]
                elif hasattr(best_model, "decision_function"):
                    y_score_te = best_model.decision_function(X_te_proc)
                else:
                    y_score_te = best_model.predict(X_te_proc).astype(float)

                model_out = models_dir / model_type
                model_out.mkdir(parents=True, exist_ok=True)
                joblib.dump(best_model, model_out / f"outer{outer_idx}_{opt_type}.pkl")

                # ---------- Apply Thresholds ----------
                met_tau = metrics_at_threshold(y_te, y_score_te, youden_threshold)
                met_05 = metrics_at_threshold(y_te, y_score_te, 0.5)
                met_sens_spec = metrics_at_threshold(y_te, y_score_te, thr_sens_spec_max)

                # ---------- Predictions in long format ----------
                pred_dfs = []

                # (1) Youden
                pred_dfs.append(pd.DataFrame({
                    "outer_fold": outer_idx,
                    "model_type": model_type,
                    "optimization": opt_type,
                    "subject_id": ids_te,
                    "y_true": y_te.astype(int).to_numpy(),
                    "y_score": y_score_te.astype(float),
                    "threshold_type": "youden",
                    "threshold_value": float(youden_threshold),
                    "y_pred": (y_score_te >= youden_threshold).astype(int),
                }))

                # (2) Fixed 0.5
                pred_dfs.append(pd.DataFrame({
                    "outer_fold": outer_idx,
                    "model_type": model_type,
                    "optimization": opt_type,
                    "subject_id": ids_te,
                    "y_true": y_te.astype(int).to_numpy(),
                    "y_score": y_score_te.astype(float),
                    "threshold_type": "0p5",
                    "threshold_value": 0.5,
                    "y_pred": (y_score_te >= 0.5).astype(int),
                }))

                # (3) Sens/Spec max
                pred_dfs.append(pd.DataFrame({
                    "outer_fold": outer_idx,
                    "model_type": model_type,
                    "optimization": opt_type,
                    "subject_id": ids_te,
                    "y_true": y_te.astype(int).to_numpy(),
                    "y_score": y_score_te.astype(float),
                    "threshold_type": f"{maximize}_max",
                    "threshold_value": float(thr_sens_spec_max),
                    "y_pred": (y_score_te >= thr_sens_spec_max).astype(int),
                }))

                # Combine them
                records_df = pd.concat(pred_dfs, ignore_index=True)

                # Append to global DataFrame
                df_outer_predictions_records = pd.concat(
                    [df_outer_predictions_records, records_df], ignore_index=True
                )

                # ---------- Metrics ----------
                rows_long = [
                    _mk_row("youden", youden_threshold, met_tau),
                    _mk_row("0p5", 0.5, met_05),
                    _mk_row(f"{maximize}_max", thr_sens_spec_max, met_sens_spec),
                ]

                df_outer_metrics_records = pd.concat(
                    [df_outer_metrics_records, pd.DataFrame(rows_long)],
                    ignore_index=True
                )


                # ---------- Pretty Printing ----------
                pretty_rows = [
                    {
                        "Outer": outer_idx,
                        "Model": model_type,
                        "Opt": opt_type,
                        "Threshold": "Youden",
                        "Thr_Val": youden_threshold,
                        "AUC": met_tau.get("auc_score"),
                        "PRC": met_tau.get("prc_score"),
                        "Sens": met_tau.get("sensitivity"),
                        "Spec": met_tau.get("specificity"),
                    },
                    {
                        "Outer": outer_idx,
                        "Model": model_type,
                        "Opt": opt_type,
                        "Threshold": "0.5",
                        "Thr_Val": 0.5,
                        "AUC": met_05.get("auc_score"),
                        "PRC": met_05.get("prc_score"),
                        "Sens": met_05.get("sensitivity"),
                        "Spec": met_05.get("specificity"),
                    },
                    {
                        "Outer": outer_idx,
                        "Model": model_type,
                        "Opt": opt_type,
                        "Threshold": f"{maximize}_max",
                        "Thr_Val": thr_sens_spec_max,
                        "AUC": met_sens_spec.get("auc_score"),
                        "PRC": met_sens_spec.get("prc_score"),
                        "Sens": met_sens_spec.get("sensitivity"),
                        "Spec": met_sens_spec.get("specificity"),
                    }
                ]
                # Convert to dataframe
                df_pretty = pd.DataFrame(pretty_rows)
                df_pretty.to_csv(
                    logs_dir / f"log_outer{outer_idx}_{model_type}_opt_{opt_type}.csv",
                    index=False
                )
                # Print with tabulate
                print("\n" + tabulate(df_pretty, headers="keys", tablefmt="psql", showindex=False))
                df_pretty_records = pd.concat([df_pretty_records, df_pretty], ignore_index=True)

                with open(params_dir / "best_params" / f"{model_type}_outer{outer_idx}_{opt_type}.json", "w") as f:
                    json.dump(params, f, indent=2)
    # ---------- Confidence Intervals, sort, save ----------
    metric_cols = ['auc_score', 'prc_score', 'sensitivity', 'specificity', 'threshold_value']
    group_cols = ["model_type", "optimization", "threshold"]

    # compute the metrics per fold per model_type and per optimization -> get one metric per each OK
    df_outer_metrics_records_ci = compute_ci_from_folds_average(
                                                df_metrics_per_fold=df_outer_metrics_records,
                                                  group_cols=group_cols,
                                                  col_metrics=metric_cols,
                                                  suffix='_ci'
                                                  )

    # 1. Extract params per run
    params_per_run = _extract_params_per_run(df_outer_metrics_records)

    # 2) Per-fold representative params (ONE row per (outer_fold, model_type))
    per_fold_param = (
        df_outer_metrics_records
        .sort_values(by=["optimization", "threshold", "model_type"])
        .drop_duplicates(subset=["optimization", "threshold", "model_type"], keep="first")
        [group_cols + ["best_params_json"]]
    )
    params_per_run.to_csv(params_dir / "params_per_run.csv", index=False)
    per_fold_param.to_csv(params_dir / "per_fold_param.csv", index=False)

    df_outer_metrics_records_ci.sort_values(
        by=["model_type", "optimization", "threshold"],
        ascending=[True, True, True],
        inplace=True
    )

    # ---------- Confidence Intervals, sort, save ----------
    df_outer_metrics_records.to_csv(metrics_dir   / "metrics_outer_folds.csv", index=False)

    df_outer_predictions_records.to_csv(preds_dir   / "predictions_outer_folds.csv", index=False)

    df_inner_val_records.to_csv(preds_dir   / "inner_val_records.csv", index=False)

    params_per_run.to_csv(params_dir   / "params_per_run.csv", index=False)

    per_fold_param.to_csv(params_dir   / "per_fold_param.csv", index=False)

    print(f"\nSaved outer metrics and predictions in {results_dir.resolve()}")
    return df_outer_metrics_records_ci, df_outer_predictions_records, df_inner_val_records

