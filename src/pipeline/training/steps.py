"""
Pure feature engineering functions for the training pipeline.

All functions are stateless — they accept data and encoders as parameters
and return transformed data or fitted encoders. No side effects.
"""

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataframe into train/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def fit_target_encoder(X_train: pd.DataFrame, y_train: pd.Series, target_encode_col: str) -> dict:
    """Fit target encoder on training data only. Returns mapping dict with global mean fallback."""
    mapping = (
        X_train.groupby(target_encode_col)
        .apply(lambda x: y_train.loc[x.index].mean())
        .to_dict()
    )
    return {"map": mapping, "global_mean": float(y_train.mean())}


def apply_target_encoding(X: pd.DataFrame, target_encoder: dict, target_encode_col: str) -> pd.DataFrame:
    """Apply target encoding to the specified column."""
    X = X.copy()
    encoded_col = target_encode_col.replace(" ", "_") + "_Encoded"
    X[encoded_col] = (
        X[target_encode_col].map(target_encoder["map"]).fillna(target_encoder["global_mean"])
    )
    return X


def fit_ohe(X_train: pd.DataFrame, cat_cols: list[str]) -> OneHotEncoder:
    """Fit OneHotEncoder on training data only."""
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    ohe.fit(X_train[cat_cols])
    return ohe


def apply_ohe(X: pd.DataFrame, ohe: OneHotEncoder, cat_cols: list[str], target_encode_col: str) -> pd.DataFrame:
    """Apply one-hot encoding and drop original categorical columns."""
    ohe_array = ohe.transform(X[cat_cols])
    ohe_df = pd.DataFrame(
        ohe_array,
        columns=ohe.get_feature_names_out(cat_cols),
        index=X.index,
    )
    return X.drop(columns=[target_encode_col] + cat_cols).join(ohe_df)


def cross_val_rmse(
    model_cfg: DictConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """K-fold CV on training set only. Returns mean RMSE across folds."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rmses = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = instantiate(model_cfg)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
    return float(np.mean(rmses))


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate model on test set. Returns RMSE, MAE, R²."""
    y_pred = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(np.mean(np.abs(y_test.values - y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
