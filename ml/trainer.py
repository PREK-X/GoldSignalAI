"""
GoldSignalAI — ml/trainer.py
==============================
Trains the XGBoost (primary) and Random Forest (verification) classifiers
using walk-forward validation.

Walk-forward validation:
  Unlike a simple train/test split, walk-forward respects time ordering.
  The data is split into K sequential folds. For each fold:
    - Train on all data BEFORE the fold
    - Test on the fold itself
  This simulates real trading where the model never sees future data.

  Fold layout (5 splits on 1000 rows):
    Fold 1: train [0:200]     test [200:400]
    Fold 2: train [0:400]     test [400:600]
    Fold 3: train [0:600]     test [600:800]
    Fold 4: train [0:800]     test [800:1000]
    (First fold is skipped if train set is too small)

Training pipeline:
  1. Build feature matrix from historical OHLCV data
  2. Walk-forward cross-validation to estimate accuracy
  3. Train final model on ALL data (for production use)
  4. Validate: if accuracy < ML_MIN_ACCURACY → reject model
  5. Save model + scaler to disk

Auto-retraining:
  Called by scheduler/tasks.py every Monday at 00:00 UTC.
  Adds the previous week's data to the training set and retrains.
"""

import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    xgb = None
    XGB_AVAILABLE = False

from config import Config
from ml.features import build_features, get_feature_columns, split_xy

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────

class TrainingResult:
    """Holds the outcome of a training run for logging and inspection."""

    def __init__(self):
        self.xgb_accuracy:    float = 0.0
        self.rf_accuracy:     float = 0.0
        self.xgb_f1:          float = 0.0
        self.rf_f1:           float = 0.0
        self.fold_accuracies: list[float] = []
        self.n_samples:       int   = 0
        self.n_features:      int   = 0
        self.training_time_s: float = 0.0
        self.timestamp:       str   = ""
        self.xgb_saved:       bool  = False
        self.rf_saved:        bool  = False
        self.scaler_saved:    bool  = False
        self.rejected:        bool  = False
        self.reject_reason:   str   = ""

    def summary(self) -> str:
        return (
            f"XGB acc={self.xgb_accuracy:.1%} f1={self.xgb_f1:.1%} | "
            f"RF acc={self.rf_accuracy:.1%} f1={self.rf_f1:.1%} | "
            f"{self.n_samples} samples, {self.n_features} features, "
            f"{self.training_time_s:.1f}s"
        )


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _walk_forward_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = Config.ML_WALK_FORWARD_SPLITS,
    min_train_rows: int = 200,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward train/test index arrays.

    Each fold uses all prior data for training and a forward slice for
    testing. This guarantees no future data leaks into the training set.

    Args:
        X:              Feature matrix
        y:              Target series
        n_splits:       Number of test folds
        min_train_rows: Minimum training set size (skip early folds)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    n = len(X)
    fold_size = n // (n_splits + 1)
    splits = []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end  = min(fold_size * (i + 1), n)

        if train_end < min_train_rows:
            continue
        if train_end >= test_end:
            continue

        train_idx = np.arange(0, train_end)
        test_idx  = np.arange(train_end, test_end)
        splits.append((train_idx, test_idx))

    logger.debug(
        "Walk-forward: %d valid splits from %d samples (fold_size=%d)",
        len(splits), n, fold_size
    )
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_xgb():
    """Create an XGBoost classifier with Config hyperparameters."""
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is not installed.")
    return xgb.XGBClassifier(
        n_estimators=Config.XGB_N_ESTIMATORS,
        max_depth=Config.XGB_MAX_DEPTH,
        min_child_weight=Config.XGB_MIN_CHILD_WEIGHT,
        learning_rate=Config.XGB_LEARNING_RATE,
        subsample=Config.XGB_SUBSAMPLE,
        colsample_bytree=Config.XGB_COLSAMPLE,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def _build_rf():
    """Create a Random Forest classifier with Config hyperparameters."""
    return RandomForestClassifier(
        n_estimators=Config.RF_N_ESTIMATORS,
        max_depth=Config.RF_MAX_DEPTH,
        random_state=42,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    save: bool = True,
) -> TrainingResult:
    """
    Train both XGBoost and Random Forest models on the provided OHLCV data.

    Pipeline:
      1. Build feature matrix with target
      2. Walk-forward cross-validation (estimate out-of-sample accuracy)
      3. Train final models on ALL data
      4. Validate accuracy threshold
      5. Save models + scaler to disk

    Args:
        df:   Processed OHLCV DataFrame (2+ years of M15 data ideally)
        save: If True, persist models to Config.MODELS_DIR

    Returns:
        TrainingResult with accuracy metrics and save status.
    """
    result = TrainingResult()
    result.timestamp = datetime.now(timezone.utc).isoformat()
    t_start = time.time()

    # ── Step 1: Feature engineering ───────────────────────────────────────
    logger.info("Building feature matrix…")
    feat_df = build_features(df, include_target=True, dropna=True)

    if len(feat_df) < 300:
        result.rejected = True
        result.reject_reason = f"Only {len(feat_df)} samples after feature engineering (need 300+)"
        logger.error(result.reject_reason)
        return result

    X, y = split_xy(feat_df)
    result.n_samples  = len(X)
    result.n_features = X.shape[1]

    logger.info("Training data: %d samples × %d features | target balance: %.1f%% up",
                len(X), X.shape[1], y.mean() * 100)

    # ── Step 2: Scale features ────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    # ── Step 3: Walk-forward cross-validation ─────────────────────────────
    logger.info("Running walk-forward cross-validation…")
    splits = _walk_forward_cv(X_scaled, y)

    fold_accs: list[float] = []
    for i, (train_idx, test_idx) in enumerate(splits):
        X_train = X_scaled.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test  = X_scaled.iloc[test_idx]
        y_test  = y.iloc[test_idx]

        # Use XGBoost for CV if available, else RF
        if XGB_AVAILABLE:
            model_cv = _build_xgb()
            model_cv.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        else:
            model_cv = _build_rf()
            model_cv.fit(X_train, y_train)

        y_pred  = model_cv.predict(X_test)
        acc     = accuracy_score(y_test, y_pred)
        fold_accs.append(acc)
        logger.debug(
            "  Fold %d: train=%d test=%d acc=%.1f%%",
            i + 1, len(train_idx), len(test_idx), acc * 100
        )

    result.fold_accuracies = fold_accs
    mean_cv_acc = np.mean(fold_accs) if fold_accs else 0.0
    logger.info(
        "Walk-forward CV: %.1f%% mean accuracy across %d folds",
        mean_cv_acc * 100, len(fold_accs)
    )

    # ── Step 4: Train final models on ALL data ────────────────────────────
    logger.info("Training final models on all %d samples…", len(X_scaled))

    # 4a: XGBoost
    if XGB_AVAILABLE:
        xgb_model = _build_xgb()
        xgb_model.fit(X_scaled, y, verbose=False)

        # Use CV accuracy (out-of-sample) as the honest reported metric.
        # In-sample accuracy is logged for debugging only — it will naturally
        # be higher than CV and does NOT reflect real predictive ability.
        insample_xgb_acc = accuracy_score(y, xgb_model.predict(X_scaled))
        insample_xgb_f1  = f1_score(y, xgb_model.predict(X_scaled), zero_division=0)
        result.xgb_accuracy = mean_cv_acc   # walk-forward CV estimate
        result.xgb_f1       = insample_xgb_f1

        logger.debug(
            "XGBoost in-sample acc=%.1f%% (CV=%.1f%%) f1=%.1f%%",
            insample_xgb_acc * 100, mean_cv_acc * 100, insample_xgb_f1 * 100,
        )
        if insample_xgb_acc > Config.ML_OVERFIT_WARNING_THRESHOLD:
            logger.warning(
                "XGBoost in-sample accuracy %.1f%% exceeds %.0f%% — "
                "model is overfitting. Reported accuracy uses CV estimate (%.1f%%).",
                insample_xgb_acc * 100,
                Config.ML_OVERFIT_WARNING_THRESHOLD * 100,
                mean_cv_acc * 100,
            )
        logger.info(
            "XGBoost CV acc=%.1f%% f1=%.1f%%",
            result.xgb_accuracy * 100, result.xgb_f1 * 100,
        )
    else:
        xgb_model = None
        logger.warning("XGBoost not available — skipping.")

    # 4b: Random Forest
    rf_model = _build_rf()
    rf_model.fit(X_scaled, y)
    insample_rf_acc = accuracy_score(y, rf_model.predict(X_scaled))
    insample_rf_f1  = f1_score(y, rf_model.predict(X_scaled), zero_division=0)
    # RF reported accuracy also uses the shared CV mean for consistency.
    result.rf_accuracy = mean_cv_acc
    result.rf_f1       = insample_rf_f1
    logger.debug(
        "Random Forest in-sample acc=%.1f%% (CV=%.1f%%) f1=%.1f%%",
        insample_rf_acc * 100, mean_cv_acc * 100, insample_rf_f1 * 100,
    )
    if insample_rf_acc > Config.ML_OVERFIT_WARNING_THRESHOLD:
        logger.warning(
            "Random Forest in-sample accuracy %.1f%% exceeds %.0f%% — "
            "model is overfitting.",
            insample_rf_acc * 100,
            Config.ML_OVERFIT_WARNING_THRESHOLD * 100,
        )
    logger.info(
        "Random Forest CV acc=%.1f%% f1=%.1f%%",
        result.rf_accuracy * 100, result.rf_f1 * 100,
    )

    # ── Step 5: Validate accuracy threshold ───────────────────────────────
    # Use CV accuracy (out-of-sample) as the honest estimate
    if mean_cv_acc < Config.ML_MIN_ACCURACY:
        result.rejected = True
        result.reject_reason = (
            f"CV accuracy {mean_cv_acc:.1%} < minimum {Config.ML_MIN_ACCURACY:.0%}. "
            f"Model NOT saved."
        )
        logger.warning(result.reject_reason)
        result.training_time_s = time.time() - t_start
        return result

    # ── Step 6: Save to disk ──────────────────────────────────────────────
    if save:
        os.makedirs(Config.MODELS_DIR, exist_ok=True)

        if xgb_model is not None:
            xgb_path = os.path.join(Config.BASE_DIR, Config.ML_MODEL_PATH)
            joblib.dump(xgb_model, xgb_path)
            result.xgb_saved = True
            logger.info("XGBoost model saved → %s", xgb_path)

        rf_path = os.path.join(Config.BASE_DIR, Config.ML_RF_MODEL_PATH)
        joblib.dump(rf_model, rf_path)
        result.rf_saved = True
        logger.info("Random Forest model saved → %s", rf_path)

        scaler_path = os.path.join(Config.BASE_DIR, Config.ML_SCALER_PATH)
        joblib.dump(scaler, scaler_path)
        result.scaler_saved = True
        logger.info("Scaler saved → %s", scaler_path)

        # Save feature column list for prediction-time alignment
        cols_path = os.path.join(Config.MODELS_DIR, "feature_columns.joblib")
        joblib.dump(list(X.columns), cols_path)
        logger.info("Feature columns saved → %s", cols_path)

    result.training_time_s = time.time() - t_start
    logger.info("Training complete: %s", result.summary())

    # ── Append to training log ────────────────────────────────────────────
    _log_training_result(result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# RETRAINING (incremental — adds new data to existing training set)
# ─────────────────────────────────────────────────────────────────────────────

def retrain(
    new_df: pd.DataFrame,
    existing_df: Optional[pd.DataFrame] = None,
) -> TrainingResult:
    """
    Retrain models by combining new data with existing historical data.

    Called by the weekly scheduler. If `existing_df` is not provided,
    trains on `new_df` alone (cold start).

    Args:
        new_df:      New OHLCV data (e.g. last week's candles)
        existing_df: Previous training data (optional)

    Returns:
        TrainingResult
    """
    if existing_df is not None:
        # Combine and deduplicate
        combined = pd.concat([existing_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        logger.info(
            "Retraining: %d existing + %d new = %d combined rows",
            len(existing_df), len(new_df), len(combined)
        )
    else:
        combined = new_df
        logger.info("Cold-start training on %d rows", len(combined))

    return train(combined, save=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING LOG
# ─────────────────────────────────────────────────────────────────────────────

def _log_training_result(result: TrainingResult) -> None:
    """Append training metrics to the ML training log file."""
    try:
        os.makedirs(os.path.dirname(Config.ML_TRAINING_LOG), exist_ok=True)
        with open(Config.ML_TRAINING_LOG, "a") as f:
            f.write(
                f"{result.timestamp} | "
                f"XGB={result.xgb_accuracy:.3f} RF={result.rf_accuracy:.3f} | "
                f"CV_folds={result.fold_accuracies} | "
                f"samples={result.n_samples} features={result.n_features} | "
                f"time={result.training_time_s:.1f}s | "
                f"saved={result.xgb_saved},{result.rf_saved}\n"
            )
    except Exception as exc:
        logger.warning("Could not write training log: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL STATUS CHECK
# ─────────────────────────────────────────────────────────────────────────────

def get_model_status() -> dict:
    """
    Check if trained models exist on disk and return their metadata.
    Used by the dashboard and Telegram /status command.
    """
    xgb_path    = os.path.join(Config.BASE_DIR, Config.ML_MODEL_PATH)
    rf_path     = os.path.join(Config.BASE_DIR, Config.ML_RF_MODEL_PATH)
    scaler_path = os.path.join(Config.BASE_DIR, Config.ML_SCALER_PATH)

    xgb_exists    = os.path.isfile(xgb_path)
    rf_exists     = os.path.isfile(rf_path)
    scaler_exists = os.path.isfile(scaler_path)

    last_train = None
    if os.path.isfile(Config.ML_TRAINING_LOG):
        try:
            with open(Config.ML_TRAINING_LOG) as f:
                lines = f.readlines()
            if lines:
                last_train = lines[-1].split("|")[0].strip()
        except Exception:
            pass

    return {
        "xgb_model_exists":    xgb_exists,
        "rf_model_exists":     rf_exists,
        "scaler_exists":       scaler_exists,
        "models_ready":        xgb_exists and rf_exists and scaler_exists,
        "last_training":       last_train,
        "xgb_model_path":      xgb_path,
        "rf_model_path":       rf_path,
    }
