"""LightGBM (MIT licence) pair classifier with grouped cross-validation."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import LGB_PARAMS, NUM_BOOST_ROUND, EARLY_STOP, N_FOLDS


def make_folds(groups: np.ndarray, mode: str, s1_country: np.ndarray | None = None) -> np.ndarray:
    """fold id per row.  group  : GroupKFold on the S1 record (no S1 split across folds)
                         country: leave-one-country-out (simulates the unseen-country test)."""
    if mode == "country":
        _, fold = np.unique(s1_country, return_inverse=True)
        return fold
    fold = np.zeros(len(groups), int)
    for f, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(groups, groups=groups)):
        fold[va] = f
    return fold


def cv_oof(X: pd.DataFrame, y: np.ndarray, fold: np.ndarray, tag="stage1"):
    oof = np.zeros(len(y), np.float32)
    iters, imps = [], []
    for f in np.unique(fold):
        tr, va = fold != f, fold == f
        dtr = lgb.Dataset(X[tr], y[tr])
        dva = lgb.Dataset(X[va], y[va])
        m = lgb.train(LGB_PARAMS, dtr, NUM_BOOST_ROUND, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        imps.append(pd.Series(m.feature_importance("gain"), index=X.columns))
        print(f"[{tag}] fold {f}: best_iter={m.best_iteration} logloss={m.best_score['valid_0']['binary_logloss']:.4f}")
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    return oof, int(np.mean(iters) * 1.1) + 1, imp


def fit_full(X: pd.DataFrame, y: np.ndarray, n_rounds: int):
    return lgb.train(LGB_PARAMS, lgb.Dataset(X, y), n_rounds)
