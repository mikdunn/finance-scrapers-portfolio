"""Train ML models on technical-indicator features.

This script is the practical starting point for your ideas:
- RandomForest (baseline)
- Gradient-boosted trees (sklearn HistGradientBoosting)

It consumes CSVs produced by `projects/market_analyzer.py`.

Outputs:
- model.joblib
- metrics.json
- feature_importance.csv (when available)

Research/education only; not financial advice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.ml_features import LabelSpec, build_features, make_labels


def _asset_name_from_path(p: Path) -> str:
    # Prefer the filename stem; market analyzer outputs are like: SYMBOL_period_interval.csv
    # For multi-asset training, we only need a stable identifier.
    stem = p.stem
    # Heuristic: symbol is before the first underscore.
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def _drop_all_nan_train_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    keep_at_least: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop columns that are entirely NaN in the training slice.

    This prevents noisy warnings from imputers/models when a feature has no
    observed values in a given fold (common with long lookback indicators).
    """
    if X_train.empty:
        return X_train, X_test

    keep_cols = X_train.columns[X_train.notna().any(axis=0)]
    if keep_cols.size < keep_at_least:
        # Be conservative: if we dropped too much, keep whatever is available.
        # (This also avoids puzzling "0 features" errors.)
        keep_cols = X_train.columns
    return X_train.loc[:, keep_cols], X_test.loc[:, keep_cols]


def _load_asset_dataset(path: Path, *, task: str, horizon: int, threshold: float) -> tuple[str, pd.DataFrame, pd.Series]:
    df = _load_csv(path)

    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        missing = sorted(needed - set(df.columns))
        raise SystemExit(f"Input CSV '{path.name}' missing required OHLC columns: {missing}")

    X = build_features(df)
    # Drop features that are always-missing across the entire asset history.
    X = X.dropna(axis=1, how="all")
    y = make_labels(
        df["Close"],
        LabelSpec(horizon=int(horizon), task=task, threshold=float(threshold)),
    )
    return _asset_name_from_path(path), X, y


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # market_analyzer writes Date as the index in CSV; when saved it becomes a column.
    # Prefer common variants.
    date_col = None
    for c in ("Date", "Datetime", "date", "datetime"):
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        # If pandas wrote an unnamed index column
        if df.columns.size and str(df.columns[0]).lower().startswith("unnamed"):
            date_col = df.columns[0]

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    return df


def _gather_inputs(in_csv: str | None, in_dir: str | None) -> list[Path]:
    if in_csv:
        p = Path(in_csv)
        if not p.exists():
            raise FileNotFoundError(str(p))
        return [p]

    if not in_dir:
        raise ValueError("Provide --in-csv or --in-dir")

    d = Path(in_dir)
    if not d.exists():
        raise FileNotFoundError(str(d))

    # Directory inputs may contain helper artifacts (e.g., macro feature caches)
    # that are not per-asset OHLCV datasets.
    exclude_names = {
        "summary.json",
        "macro_features.csv",
        "extra_features_raw.csv",
    }

    def _is_dataset_csv(p: Path) -> bool:
        n = p.name.lower()
        if n in exclude_names:
            return False
        if n.endswith("_importance.csv"):
            return False
        return n.endswith(".csv")

    def _list_csvs_shallow(root: Path) -> list[Path]:
        """List CSVs from a directory, plus one sharding level.

        This intentionally avoids a deep rglob(). Some output folders can be
        huge (charts, per-run artifacts), and a recursive walk can dominate
        runtime on Windows.
        """
        out: list[Path] = []
        out.extend([p for p in root.glob("*.csv") if _is_dataset_csv(p)])
        # Common sharded layout: root/<prefix>/*.csv
        try:
            for child in root.iterdir():
                if child.is_dir():
                    out.extend([p for p in child.glob("*.csv") if _is_dataset_csv(p)])
        except FileNotFoundError:
            pass
        return out

    # Prefer a dedicated assets subfolder if present.
    candidates: list[Path] = []
    assets = d / "assets"
    if assets.exists() and assets.is_dir():
        candidates = _list_csvs_shallow(assets)
    if not candidates:
        candidates = _list_csvs_shallow(d)

    return sorted(candidates)


def _time_split(n: int, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    if not (0.05 <= test_size <= 0.8):
        raise ValueError("test_size should be between 0.05 and 0.8")

    # Enforce a sane minimum so we don't end up with tiny test sets.
    min_train = 30
    min_test = 10
    cut = int(np.floor(n * (1.0 - test_size)))
    cut = max(min_train, min(cut, n - min_test))
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


def _walkforward_splits(
    n: int,
    *,
    n_splits: int,
    test_window: int,
    purge: int,
    min_train: int = 60,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward splits.

    IMPORTANT: `purge` removes the last N samples of each training window to
    prevent label leakage when labels use future returns (shift(-horizon)).
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if test_window < 5:
        raise ValueError("test_window must be >= 5")
    if purge < 0:
        raise ValueError("purge must be >= 0")

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    # Put the first test window after a minimum training span.
    first_test_start = max(min_train, purge)

    # Compute an even-ish spacing of test windows, ending at n.
    last_possible_start = n - test_window
    if last_possible_start <= first_test_start:
        raise ValueError(
            f"Not enough rows for walk-forward CV (n={n}). "
            "Try a longer period, smaller test_window, or fewer splits."
        )

    starts = np.linspace(first_test_start, last_possible_start, num=n_splits, dtype=int)
    starts = np.unique(starts)

    for s in starts:
        train_end = s - purge
        if train_end <= 0:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(s, min(n, s + test_window))
        if len(train_idx) < min_train or len(test_idx) < 5:
            continue
        splits.append((train_idx, test_idx))

    if len(splits) < 2:
        raise ValueError(
            "Walk-forward split generation produced too few usable folds. "
            "Try: --period 1y, --test-window 10, --n-splits 5."
        )
    return splits


def _make_pipeline(task: str, model_name: str, random_state: int):
    """Create an sklearn Pipeline with an imputer + model."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    task = (task or "classification").strip().lower()
    model_name = (model_name or "hgb").strip().lower()

    model = None
    if task in {"regression", "reg", "return"}:
        if model_name in {"rf", "random_forest"}:
            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=random_state,
            )
        elif model_name in {"xgb", "xgboost"}:
            try:
                import importlib

                XGBRegressor = importlib.import_module("xgboost").XGBRegressor
            except Exception as e:
                raise RuntimeError(
                    "xgboost is not installed. Install it with: pip install xgboost"
                ) from e

            model = XGBRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=random_state,
                tree_method="hist",
            )
        else:
            # Strong GBDT baseline for tabular data
            from sklearn.ensemble import HistGradientBoostingRegressor

            model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=6,
                max_iter=800,
                random_state=random_state,
            )
    else:
        # Classification
        if model_name in {"rf", "random_forest"}:
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(
                n_estimators=800,
                max_depth=None,
                min_samples_leaf=3,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=random_state,
            )
        elif model_name in {"xgb", "xgboost"}:
            try:
                import importlib

                XGBClassifier = importlib.import_module("xgboost").XGBClassifier
            except Exception as e:
                raise RuntimeError(
                    "xgboost is not installed. Install it with: pip install xgboost"
                ) from e

            # 3-class: -1,0,1 (sell/hold/buy). We'll map to 0..2 internally.
            model = XGBClassifier(
                n_estimators=1200,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=random_state,
                tree_method="hist",
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=3,
            )
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_depth=6,
                max_iter=800,
                random_state=random_state,
            )

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model),
        ]
    )


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
        "labels": {"-1": "sell", "0": "hold", "1": "buy"},
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _get_feature_importance(
    pipe,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    task: str,
    method: str,
    random_state: int,
) -> pd.DataFrame | None:
    """Return a (feature, importance) dataframe.

    - method="model": uses model.feature_importances_ when available
    - method="permutation": uses sklearn permutation importance
    """
    method = (method or "model").strip().lower()

    if method in {"model", "tree"}:
        try:
            m = pipe.named_steps["model"]
            if hasattr(m, "feature_importances_"):
                imp = np.asarray(m.feature_importances_, dtype=float)
                return (
                    pd.DataFrame({"feature": X_test.columns, "importance": imp})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )
        except Exception:
            return None

    if method in {"permutation", "perm"}:
        from sklearn.inspection import permutation_importance

        # If we trained an XGBoost multi-class classifier, the pipeline predicts
        # classes 0..K-1. Our canonical labels are -1,0,1, so map y accordingly
        # for scoring.
        y_for_scoring = y_test
        try:
            m = pipe.named_steps.get("model")
            task_norm = (task or "classification").strip().lower()
            if task_norm not in {"regression", "reg", "return"}:
                cls_name = getattr(m, "__class__", type("x", (), {})).__name__
                if "XGB" in cls_name:
                    uniq = set(pd.Series(y_test).dropna().astype(int).unique().tolist())
                    if uniq.issubset({-1, 0, 1}):
                        y_for_scoring = pd.Series(y_test).replace({-1: 0, 0: 1, 1: 2}).astype(int)
        except Exception:
            y_for_scoring = y_test

        scoring = "r2" if (task or "classification").strip().lower() in {"regression", "reg", "return"} else "f1_macro"
        r = permutation_importance(
            pipe,
            X_test,
            y_for_scoring,
            n_repeats=8,
            random_state=random_state,
            scoring=scoring,
        )
        imp = np.asarray(r.importances_mean, dtype=float)
        return (
            pd.DataFrame({"feature": X_test.columns, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    return None


def train_supervised(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: str,
    model_name: str,
    test_size: float,
    random_state: int,
) -> tuple[object, dict, pd.DataFrame | None]:
    """Train a supervised model (classification/regression) with a time-based holdout."""
    task = (task or "classification").strip().lower()
    model_name = (model_name or "rf").strip().lower()

    # Filter rows with a label
    mask = ~pd.isna(y)
    X = X.loc[mask]
    y = y.loc[mask]

    # Also remove rows where all features are NaN
    X = X.dropna(how="all")
    y = y.loc[X.index]

    n = len(X)
    if n < 80:
        raise ValueError(
            f"Not enough rows after cleaning for ML training (n={n}). "
            "Try a longer --period (e.g., 6mo/1y) or a faster interval."
        )

    train_idx, test_idx = _time_split(n, test_size)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    # Fold-specific cleanup: drop columns that are all-NaN in the training slice.
    X_train, X_test = _drop_all_nan_train_columns(X_train, X_test)

    pipe = _make_pipeline(task, model_name, random_state)

    if task in {"regression", "reg", "return"}:
        y_train = y.iloc[train_idx].astype(float)
        y_test = y.iloc[test_idx].astype(float)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = {
            "task": "regression",
            "model": model_name,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            **_regression_metrics(y_test.to_numpy(dtype=float), np.asarray(pred, dtype=float)),
        }
        return pipe, metrics, _get_feature_importance(pipe, X_test, y_test, task=task, method="model", random_state=random_state)

    # Classification
    y_train = y.iloc[train_idx].astype(int)
    y_test = y.iloc[test_idx].astype(int)

    # XGBoost expects classes 0..K-1. Map -1,0,1 -> 0,1,2 when needed.
    y_train_fit = y_train
    y_test_eval = y_test
    if model_name in {"xgb", "xgboost"}:
        y_train_fit = y_train.replace({-1: 0, 0: 1, 1: 2}).astype(int)
        y_test_eval = y_test.replace({-1: 0, 0: 1, 1: 2}).astype(int)

    pipe.fit(X_train, y_train_fit)
    pred = pipe.predict(X_test)
    pred = np.asarray(pred)
    if model_name in {"xgb", "xgboost"}:
        # Map back 0,1,2 -> -1,0,1
        pred = pd.Series(pred).replace({0: -1, 1: 0, 2: 1}).to_numpy(dtype=int)
        y_test_eval = pd.Series(y_test_eval).replace({0: -1, 1: 0, 2: 1}).to_numpy(dtype=int)
    else:
        y_test_eval = y_test.to_numpy(dtype=int)

    metrics = {
        "task": "classification",
        "model": model_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        **_classification_metrics(y_test_eval, pred.astype(int)),
    }

    importance_df = _get_feature_importance(pipe, X_test, y_test, task=task, method="model", random_state=random_state)
    return pipe, metrics, importance_df


def train_supervised_multi(
    datasets: list[tuple[str, pd.DataFrame, pd.Series]],
    *,
    task: str,
    model_name: str,
    test_size: float,
    random_state: int,
) -> tuple[object, dict, pd.DataFrame | None]:
    """Train one model on multiple assets using per-asset time-based holdout.

    We split each asset in time, then concatenate all train slices and all test
    slices. We also add one-hot asset ID features.
    """
    task = (task or "classification").strip().lower()
    model_name = (model_name or "hgb").strip().lower()

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    y_train_parts: list[pd.Series] = []
    y_test_parts: list[pd.Series] = []

    for asset, X, y in datasets:
        mask = ~pd.isna(y)
        Xa = X.loc[mask].copy()
        ya = y.loc[mask].copy()
        Xa = Xa.dropna(how="all")
        ya = ya.loc[Xa.index]

        n = len(Xa)
        if n < 80:
            raise ValueError(
                f"Not enough rows for ML training for asset '{asset}' (n={n}). "
                "Try a longer --period (e.g., 6mo/1y) or a faster interval."
            )

        tr_idx, te_idx = _time_split(n, test_size)
        Xtr = Xa.iloc[tr_idx]
        Xte = Xa.iloc[te_idx]
        ytr = ya.iloc[tr_idx]
        yte = ya.iloc[te_idx]

        Xtr = Xtr.copy()
        Xte = Xte.copy()
        Xtr["asset"] = asset
        Xte["asset"] = asset

        train_parts.append(Xtr)
        test_parts.append(Xte)
        y_train_parts.append(ytr)
        y_test_parts.append(yte)

    X_train = pd.concat(train_parts, axis=0).sort_index()
    X_test = pd.concat(test_parts, axis=0).sort_index()
    y_train = pd.concat(y_train_parts, axis=0).sort_index()
    y_test = pd.concat(y_test_parts, axis=0).sort_index()

    # One-hot asset identifier
    X_all = pd.concat([X_train, X_test], axis=0)
    X_all = pd.get_dummies(X_all, columns=["asset"], prefix="asset", dtype=float)
    X_train = X_all.iloc[: len(X_train)]
    X_test = X_all.iloc[len(X_train) :]

    X_train, X_test = _drop_all_nan_train_columns(X_train, X_test)

    pipe = _make_pipeline(task, model_name, random_state)

    if task in {"regression", "reg", "return"}:
        y_train_f = y_train.astype(float)
        y_test_f = y_test.astype(float)
        pipe.fit(X_train, y_train_f)
        pred = pipe.predict(X_test)
        metrics = {
            "task": "regression",
            "model": model_name,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_assets": int(len(datasets)),
            "assets": [a for a, _, _ in datasets],
            **_regression_metrics(y_test_f.to_numpy(dtype=float), np.asarray(pred, dtype=float)),
        }
        return pipe, metrics, _get_feature_importance(pipe, X_test, y_test_f, task=task, method="model", random_state=random_state)

    y_train_c = y_train.astype(int)
    y_test_c = y_test.astype(int)

    y_train_fit = y_train_c
    y_test_eval = y_test_c
    if model_name in {"xgb", "xgboost"}:
        y_train_fit = y_train_c.replace({-1: 0, 0: 1, 1: 2}).astype(int)
        y_test_eval = y_test_c.replace({-1: 0, 0: 1, 1: 2}).astype(int)

    pipe.fit(X_train, y_train_fit)
    pred = np.asarray(pipe.predict(X_test))
    if model_name in {"xgb", "xgboost"}:
        pred = pd.Series(pred).replace({0: -1, 1: 0, 2: 1}).to_numpy(dtype=int)
        y_test_eval = pd.Series(y_test_eval).replace({0: -1, 1: 0, 2: 1}).to_numpy(dtype=int)
    else:
        y_test_eval = y_test_eval.to_numpy(dtype=int)

    metrics = {
        "task": "classification",
        "model": model_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_assets": int(len(datasets)),
        "assets": [a for a, _, _ in datasets],
        **_classification_metrics(y_test_eval, pred.astype(int)),
    }

    importance_df = _get_feature_importance(pipe, X_test, y_test_c, task=task, method="model", random_state=random_state)
    return pipe, metrics, importance_df


def walkforward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    task: str,
    model_name: str,
    n_splits: int,
    test_window: int,
    purge: int,
    importance_method: str,
    random_state: int,
) -> tuple[list[dict], pd.DataFrame | None]:
    """Run walk-forward CV and return per-fold metrics + mean importance."""
    task = (task or "classification").strip().lower()
    model_name = (model_name or "hgb").strip().lower()

    # Filter rows with a label
    mask = ~pd.isna(y)
    X = X.loc[mask]
    y = y.loc[mask]
    X = X.dropna(how="all")
    y = y.loc[X.index]

    n = len(X)
    if n < 120:
        raise ValueError(
            f"Not enough rows for walk-forward CV (n={n}). "
            "Try a longer --period (e.g., 1y) or a faster interval."
        )

    splits = _walkforward_splits(n, n_splits=n_splits, test_window=test_window, purge=purge)
    fold_metrics: list[dict] = []
    importances: list[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        # Fold-specific cleanup: drop all-NaN columns in this fold's training window.
        X_train, X_test = _drop_all_nan_train_columns(X_train, X_test)

        if task in {"regression", "reg", "return"}:
            y_train = y.iloc[train_idx].astype(float)
            y_test = y.iloc[test_idx].astype(float)
            pipe = _make_pipeline(task, model_name, random_state + fold)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            m = {
                "fold": fold,
                "train_end": str(X_train.index[-1]),
                "test_start": str(X_test.index[0]),
                "test_end": str(X_test.index[-1]),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                **_regression_metrics(y_test.to_numpy(dtype=float), np.asarray(pred, dtype=float)),
            }
            fold_metrics.append(m)
            imp_df = _get_feature_importance(
                pipe,
                X_test,
                y_test,
                task=task,
                method=importance_method,
                random_state=random_state + fold,
            )
        else:
            y_train = y.iloc[train_idx].astype(int)
            y_test = y.iloc[test_idx].astype(int)
            pipe = _make_pipeline(task, model_name, random_state + fold)

            y_train_fit = y_train
            if model_name in {"xgb", "xgboost"}:
                y_train_fit = y_train.replace({-1: 0, 0: 1, 1: 2}).astype(int)

            pipe.fit(X_train, y_train_fit)
            pred = pipe.predict(X_test)
            pred = np.asarray(pred)
            if model_name in {"xgb", "xgboost"}:
                pred = pd.Series(pred).replace({0: -1, 1: 0, 2: 1}).to_numpy(dtype=int)
            m = {
                "fold": fold,
                "train_end": str(X_train.index[-1]),
                "test_start": str(X_test.index[0]),
                "test_end": str(X_test.index[-1]),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                **_classification_metrics(y_test.to_numpy(dtype=int), pred.astype(int)),
            }
            fold_metrics.append(m)
            imp_df = _get_feature_importance(
                pipe,
                X_test,
                y_test,
                task=task,
                method=importance_method,
                random_state=random_state + fold,
            )

        if imp_df is not None:
            imp_df = imp_df.copy()
            imp_df["fold"] = fold
            importances.append(imp_df)

    if not importances:
        return fold_metrics, None

    all_imp = pd.concat(importances, ignore_index=True)
    mean_imp = (
        all_imp.groupby("feature", as_index=False)["importance"].mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return fold_metrics, mean_imp


def walkforward_cv_multi(
    datasets: list[tuple[str, pd.DataFrame, pd.Series]],
    *,
    task: str,
    model_name: str,
    n_splits: int,
    test_window: int,
    purge: int,
    importance_method: str,
    random_state: int,
) -> tuple[list[dict], pd.DataFrame | None]:
    """Walk-forward CV for a multi-asset panel.

    Strategy: generate walk-forward folds independently per asset, then for each
    fold index, concatenate the train/test slices across assets. This keeps the
    split logic time-respecting within each asset.
    """
    task = (task or "classification").strip().lower()
    model_name = (model_name or "hgb").strip().lower()

    per_asset: list[dict] = []
    for asset, X, y in datasets:
        mask = ~pd.isna(y)
        Xa = X.loc[mask].copy()
        ya = y.loc[mask].copy()
        Xa = Xa.dropna(how="all")
        ya = ya.loc[Xa.index]
        n = len(Xa)
        if n < 120:
            raise ValueError(
                f"Not enough rows for walk-forward CV for asset '{asset}' (n={n}). "
                "Try a longer --period (e.g., 1y) or a faster interval."
            )
        splits = _walkforward_splits(n, n_splits=n_splits, test_window=test_window, purge=purge)
        per_asset.append({"asset": asset, "X": Xa, "y": ya, "splits": splits})

    n_folds = min(len(d["splits"]) for d in per_asset)
    if n_folds < 2:
        raise ValueError("Multi-asset walk-forward produced too few usable folds across assets.")

    fold_metrics: list[dict] = []
    importances: list[pd.DataFrame] = []

    for fold in range(1, n_folds + 1):
        train_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []
        y_train_parts: list[pd.Series] = []
        y_test_parts: list[pd.Series] = []

        fold_train_ends: list[str] = []
        fold_test_starts: list[str] = []
        fold_test_ends: list[str] = []

        for d in per_asset:
            asset = d["asset"]
            Xa: pd.DataFrame = d["X"]
            ya: pd.Series = d["y"]
            tr_idx, te_idx = d["splits"][fold - 1]
            Xtr = Xa.iloc[tr_idx].copy()
            Xte = Xa.iloc[te_idx].copy()
            ytr = ya.iloc[tr_idx]
            yte = ya.iloc[te_idx]
            Xtr["asset"] = asset
            Xte["asset"] = asset
            train_parts.append(Xtr)
            test_parts.append(Xte)
            y_train_parts.append(ytr)
            y_test_parts.append(yte)
            fold_train_ends.append(str(Xtr.index[-1]))
            fold_test_starts.append(str(Xte.index[0]))
            fold_test_ends.append(str(Xte.index[-1]))

        X_train = pd.concat(train_parts, axis=0).sort_index()
        X_test = pd.concat(test_parts, axis=0).sort_index()
        y_train = pd.concat(y_train_parts, axis=0).sort_index()
        y_test = pd.concat(y_test_parts, axis=0).sort_index()

        X_all = pd.concat([X_train, X_test], axis=0)
        X_all = pd.get_dummies(X_all, columns=["asset"], prefix="asset", dtype=float)
        X_train = X_all.iloc[: len(X_train)]
        X_test = X_all.iloc[len(X_train) :]

        X_train, X_test = _drop_all_nan_train_columns(X_train, X_test)

        pipe = _make_pipeline(task, model_name, random_state + fold)

        if task in {"regression", "reg", "return"}:
            ytr = y_train.astype(float)
            yte = y_test.astype(float)
            pipe.fit(X_train, ytr)
            pred = pipe.predict(X_test)
            m = {
                "fold": fold,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "n_assets": int(len(datasets)),
                "train_end_min": min(fold_train_ends),
                "train_end_max": max(fold_train_ends),
                "test_start_min": min(fold_test_starts),
                "test_start_max": max(fold_test_starts),
                "test_end_min": min(fold_test_ends),
                "test_end_max": max(fold_test_ends),
                **_regression_metrics(yte.to_numpy(dtype=float), np.asarray(pred, dtype=float)),
            }
            fold_metrics.append(m)
            imp_df = _get_feature_importance(
                pipe,
                X_test,
                yte,
                task=task,
                method=importance_method,
                random_state=random_state + fold,
            )
        else:
            ytr = y_train.astype(int)
            yte = y_test.astype(int)
            ytr_fit = ytr
            if model_name in {"xgb", "xgboost"}:
                ytr_fit = ytr.replace({-1: 0, 0: 1, 1: 2}).astype(int)

            pipe.fit(X_train, ytr_fit)
            pred = np.asarray(pipe.predict(X_test))
            if model_name in {"xgb", "xgboost"}:
                pred = pd.Series(pred).replace({0: -1, 1: 0, 2: 1}).to_numpy(dtype=int)

            m = {
                "fold": fold,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "n_assets": int(len(datasets)),
                "train_end_min": min(fold_train_ends),
                "train_end_max": max(fold_train_ends),
                "test_start_min": min(fold_test_starts),
                "test_start_max": max(fold_test_starts),
                "test_end_min": min(fold_test_ends),
                "test_end_max": max(fold_test_ends),
                **_classification_metrics(yte.to_numpy(dtype=int), pred.astype(int)),
            }
            fold_metrics.append(m)
            imp_df = _get_feature_importance(
                pipe,
                X_test,
                yte,
                task=task,
                method=importance_method,
                random_state=random_state + fold,
            )

        if imp_df is not None:
            imp_df = imp_df.copy()
            imp_df["fold"] = fold
            importances.append(imp_df)

    if not importances:
        return fold_metrics, None

    all_imp = pd.concat(importances, ignore_index=True)
    mean_imp = (
        all_imp.groupby("feature", as_index=False)["importance"].mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return fold_metrics, mean_imp


def _write_importance_plot(importance_df: pd.DataFrame, out_html: Path, *, title: str, top_n: int = 30) -> None:
    import plotly.express as px

    df = importance_df.head(int(top_n)).copy()
    fig = px.bar(df[::-1], x="importance", y="feature", orientation="h", title=title)
    fig.update_layout(template="plotly_white", height=max(500, 18 * len(df) + 120))
    fig.write_html(str(out_html))


def _write_cv_plot(metrics: list[dict], out_html: Path, *, task: str) -> None:
    import plotly.express as px

    df = pd.DataFrame(metrics)
    task = (task or "classification").strip().lower()
    if task in {"regression", "reg", "return"}:
        y_col = "rmse" if "rmse" in df.columns else df.columns[-1]
        fig = px.line(df, x="fold", y=y_col, markers=True, title=f"Walk-forward CV: {y_col} by fold")
    else:
        y_col = "f1_macro" if "f1_macro" in df.columns else "accuracy"
        fig = px.line(df, x="fold", y=y_col, markers=True, title=f"Walk-forward CV: {y_col} by fold")
    fig.update_layout(template="plotly_white")
    fig.write_html(str(out_html))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ML models from market analyzer CSV outputs")
    parser.add_argument("--in-csv", default=None, help="Single CSV path (from market analyzer)")
    parser.add_argument("--in-dir", default=None, help="Directory of market analyzer CSVs")
    parser.add_argument("--model", default="hgb", help="rf | hgb | xgb")
    parser.add_argument("--task", default="classification", help="classification | regression")
    parser.add_argument("--horizon", type=int, default=5, help="Label horizon in candles")
    parser.add_argument("--threshold", type=float, default=0.002, help="Classification threshold on future return")
    parser.add_argument("--cv", default="holdout", help="holdout | walkforward")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction for --cv holdout")
    parser.add_argument("--n-splits", type=int, default=6, help="Walk-forward: number of folds")
    parser.add_argument("--test-window", type=int, default=20, help="Walk-forward: test window size (candles)")
    parser.add_argument("--purge", type=int, default=None, help="Walk-forward: purge size to prevent label leakage (default: horizon)")
    parser.add_argument("--importance", default="model", help="Feature importance: model | permutation")
    parser.add_argument("--top-features", type=int, default=30, help="How many features to include in plots")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", default="ml_outputs", help="Output directory")
    parser.add_argument(
        "--multi-asset",
        action="store_true",
        help="Train on all CSVs in --in-dir by combining assets (adds one-hot asset features).",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _gather_inputs(args.in_csv, args.in_dir)
    if not paths:
        raise SystemExit("No CSVs found.")

    datasets: list[tuple[str, pd.DataFrame, pd.Series]] = []
    if args.multi_asset and args.in_dir:
        for pth in paths:
            datasets.append(
                _load_asset_dataset(
                    pth,
                    task=args.task,
                    horizon=int(args.horizon),
                    threshold=float(args.threshold),
                )
            )
    else:
        # Single-asset baseline: train on first CSV.
        p = paths[0]
        datasets.append(
            _load_asset_dataset(
                p,
                task=args.task,
                horizon=int(args.horizon),
                threshold=float(args.threshold),
            )
        )

    # For metrics.json input field
    p = paths[0]

    cv = (args.cv or "holdout").strip().lower()
    purge = int(args.purge) if args.purge is not None else int(args.horizon)

    single_asset_mode = (len(datasets) == 1)

    if single_asset_mode:
        _, X, y = datasets[0]

    if cv in {"walkforward", "walk_forward", "walk-forward", "wf"}:
        if single_asset_mode:
            fold_metrics, mean_imp = walkforward_cv(
                X,
                y,
                task=args.task,
                model_name=args.model,
                n_splits=int(args.n_splits),
                test_window=int(args.test_window),
                purge=purge,
                importance_method=args.importance,
                random_state=int(args.random_state),
            )
        else:
            fold_metrics, mean_imp = walkforward_cv_multi(
                datasets,
                task=args.task,
                model_name=args.model,
                n_splits=int(args.n_splits),
                test_window=int(args.test_window),
                purge=purge,
                importance_method=args.importance,
                random_state=int(args.random_state),
            )

        # Save CV results
        cv_metrics_path = out_dir / "cv_metrics.csv"
        pd.DataFrame(fold_metrics).to_csv(cv_metrics_path, index=False)
        _write_cv_plot(fold_metrics, out_dir / "cv_plot.html", task=args.task)

        importance_df = mean_imp
        if importance_df is not None:
            importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
            _write_importance_plot(
                importance_df,
                out_dir / "feature_importance.html",
                title=f"Feature importance (mean across folds) - {args.model}",
                top_n=int(args.top_features),
            )

        metrics = {
            "mode": "walkforward",
            "task": (args.task or "classification"),
            "model": (args.model or "hgb"),
            "n_folds": int(len(fold_metrics)),
            "purge": int(purge),
            "test_window": int(args.test_window),
            "cv_metrics_csv": str(cv_metrics_path),
        }

        if not single_asset_mode:
            metrics["n_assets"] = int(len(datasets))
            metrics["assets"] = [a for a, _, _ in datasets]

        # Train a final model on the full dataset (minus last purge) so you have an artifact to deploy.
        # NOTE: For true deployment you'd retrain periodically and re-evaluate out-of-sample.
        # Train a final model artifact.
        if single_asset_mode:
            X_fit = X.copy()
            y_fit = y.copy()
            if purge > 0 and len(X_fit) > purge:
                X_fit = X_fit.iloc[:-purge]
                y_fit = y_fit.iloc[:-purge]
            model, _, _ = train_supervised(
                X_fit,
                y_fit,
                task=args.task,
                model_name=args.model,
                test_size=0.2,
                random_state=args.random_state,
            )
        else:
            trimmed: list[tuple[str, pd.DataFrame, pd.Series]] = []
            for asset, Xa, ya in datasets:
                if purge > 0 and len(Xa) > purge:
                    trimmed.append((asset, Xa.iloc[:-purge], ya.iloc[:-purge]))
                else:
                    trimmed.append((asset, Xa, ya))
            model, _, _ = train_supervised_multi(
                trimmed,
                task=args.task,
                model_name=args.model,
                test_size=0.2,
                random_state=args.random_state,
            )
    else:
        if single_asset_mode:
            model, metrics, importance_df = train_supervised(
                X,
                y,
                task=args.task,
                model_name=args.model,
                test_size=args.test_size,
                random_state=args.random_state,
            )
        else:
            model, metrics, importance_df = train_supervised_multi(
                datasets,
                task=args.task,
                model_name=args.model,
                test_size=args.test_size,
                random_state=args.random_state,
            )

        if importance_df is not None:
            importance_df.to_csv(out_dir / "feature_importance.csv", index=False)
            _write_importance_plot(
                importance_df,
                out_dir / "feature_importance.html",
                title=f"Feature importance - {args.model}",
                top_n=int(args.top_features),
            )

    # Persist
    from joblib import dump

    model_path = out_dir / "model.joblib"
    dump(model, model_path)

    # Best-effort feature count after any fold-specific column dropping / asset dummies.
    resolved_feature_count = None
    try:
        if hasattr(model, "named_steps") and "imputer" in model.named_steps:
            resolved_feature_count = int(getattr(model.named_steps["imputer"], "statistics_", np.array([])).shape[0])
        elif hasattr(model, "named_steps") and "model" in model.named_steps:
            resolved_feature_count = int(getattr(model.named_steps["model"], "n_features_in_", 0))
    except Exception:
        resolved_feature_count = None

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "input": str(p) if single_asset_mode else None,
            "inputs": [str(pp) for pp in paths] if not single_asset_mode else None,
            "feature_count": resolved_feature_count,
            **metrics,
        }, f, indent=2)

    print(f"Wrote: {model_path}")
    print(f"Wrote: {metrics_path}")
    if (out_dir / "feature_importance.csv").exists():
        print(f"Wrote: {out_dir / 'feature_importance.csv'}")
    if (out_dir / "feature_importance.html").exists():
        print(f"Wrote: {out_dir / 'feature_importance.html'}")
    if (out_dir / "cv_metrics.csv").exists():
        print(f"Wrote: {out_dir / 'cv_metrics.csv'}")
    if (out_dir / "cv_plot.html").exists():
        print(f"Wrote: {out_dir / 'cv_plot.html'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
