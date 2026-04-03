"""
GoldSignalAI — ml/deep_predictor.py
=====================================
Inference for the CNN-BiLSTM deep model (Stage 7).

Loads the saved model and scaler at startup, then runs inference
on single bars or full batches for backtesting.

Used by:
  - signals/generator.py (single bar inference in live mode)
  - backtest/engine.py (batch prediction for all bars)
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import joblib

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class DeepPrediction:
    """Prediction result from the CNN-BiLSTM model."""
    probability: float = 0.5  # P(price goes up)
    available: bool = False
    reason: str = ""

    def confirms(self, signal_direction: str) -> bool:
        """
        Check if deep model confirms a signal direction.

        For BUY: probability must be >= DEEP_MIN_CONFIDENCE (bullish)
        For SELL: probability must be <= (1 - DEEP_MIN_CONFIDENCE) (bearish)
        If not available, don't block.
        """
        if not self.available:
            return True  # don't block if model unavailable
        if signal_direction == "BUY":
            return self.probability >= Config.DEEP_MIN_CONFIDENCE
        if signal_direction == "SELL":
            return self.probability <= (1.0 - Config.DEEP_MIN_CONFIDENCE)
        return True

    def summary(self) -> str:
        return f"Deep: P(up)={self.probability:.1%} | threshold={Config.DEEP_MIN_CONFIDENCE:.0%}"


class _DeepCache:
    """In-memory cache for the deep model and scaler."""
    model = None
    scaler = None
    loaded: bool = False

    @classmethod
    def load(cls) -> bool:
        model_path = os.path.join(Config.BASE_DIR, Config.DEEP_MODEL_PATH)
        scaler_path = os.path.join(Config.BASE_DIR, Config.DEEP_SCALER_PATH)

        try:
            if not os.path.isfile(model_path):
                logger.debug("Deep model not found at %s", model_path)
                cls.loaded = False
                return False

            from ml.deep_model import load_model
            cls.model = load_model(model_path)
            if cls.model is None:
                cls.loaded = False
                return False

            if os.path.isfile(scaler_path):
                cls.scaler = joblib.load(scaler_path)
            else:
                cls.scaler = None  # model was trained without scaler

            cls.loaded = True
            logger.info("Deep model loaded")
            return True

        except Exception as exc:
            logger.exception("Failed to load deep model: %s", exc)
            cls.loaded = False
            return False

    @classmethod
    def ensure_loaded(cls) -> bool:
        if cls.loaded:
            return True
        return cls.load()

    @classmethod
    def reload(cls) -> bool:
        cls.loaded = False
        return cls.load()


def is_deep_ready() -> bool:
    """Check if deep model is loaded and ready."""
    return _DeepCache.ensure_loaded()


def reload_deep_model() -> bool:
    """Reload CNN-BiLSTM from disk without restarting bot. Called after retrain."""
    return _DeepCache.reload()


def predict_deep(feat_df: pd.DataFrame, bar_index: int) -> DeepPrediction:
    """
    Predict direction probability for a single bar.

    Args:
        feat_df: Full feature DataFrame from build_deep_features().
        bar_index: Integer index of the bar to predict.

    Returns:
        DeepPrediction with probability and availability.
    """
    if not _DeepCache.ensure_loaded():
        return DeepPrediction(available=False, reason="Deep model not trained")

    lookback = Config.DEEP_LOOKBACK
    if bar_index < lookback:
        return DeepPrediction(available=False, reason=f"Insufficient bars ({bar_index} < {lookback})")

    from ml.deep_features import build_inference_window, DEEP_FEATURE_NAMES
    window = build_inference_window(feat_df, bar_index, lookback)
    if window is None:
        return DeepPrediction(available=False, reason="Window contains NaN")

    # Apply scaler if available
    if _DeepCache.scaler is not None:
        shape = window.shape  # (1, lookback, n_features)
        flat = window.reshape(-1, shape[2])
        flat = _DeepCache.scaler.transform(flat)
        window = flat.reshape(shape)

    try:
        prob = float(_DeepCache.model.predict(window, verbose=0)[0, 0])
    except Exception as exc:
        return DeepPrediction(available=False, reason=f"Prediction error: {exc}")

    return DeepPrediction(probability=prob, available=True, reason=f"P(up)={prob:.3f}")


def predict_deep_batch(
    feat_df: pd.DataFrame,
    close_series: pd.Series,
    lookback: int = 60,
) -> pd.DataFrame:
    """
    Batch prediction for all bars (for backtesting).

    Returns DataFrame with columns: deep_prob, deep_pass_buy, deep_pass_sell
    indexed same as feat_df.
    """
    if not _DeepCache.ensure_loaded():
        logger.error("Cannot run deep batch prediction — model not loaded.")
        return pd.DataFrame()

    from ml.deep_features import DEEP_FEATURE_NAMES

    cols = [c for c in DEEP_FEATURE_NAMES if c in feat_df.columns]
    values = feat_df[cols].values.astype(np.float32)
    n_bars = len(values)

    result = pd.DataFrame(index=feat_df.index)
    result["deep_prob"] = np.nan
    result["deep_pass_buy"] = False
    result["deep_pass_sell"] = False

    # Build all valid windows
    windows = []
    valid_indices = []
    for i in range(lookback, n_bars):
        window = values[i - lookback:i]
        if np.isnan(window).any():
            continue
        windows.append(window)
        valid_indices.append(i)

    if not windows:
        return result

    X = np.stack(windows)

    # Apply scaler if available
    if _DeepCache.scaler is not None:
        shape = X.shape
        flat = X.reshape(-1, shape[2])
        flat = _DeepCache.scaler.transform(flat)
        X = flat.reshape(shape).astype(np.float32)

    # Batch predict
    probs = _DeepCache.model.predict(X, batch_size=256, verbose=0).flatten()

    # Fill results
    min_conf = Config.DEEP_MIN_CONFIDENCE
    for idx, prob in zip(valid_indices, probs):
        result.iloc[idx, result.columns.get_loc("deep_prob")] = float(prob)
        result.iloc[idx, result.columns.get_loc("deep_pass_buy")] = prob >= min_conf
        result.iloc[idx, result.columns.get_loc("deep_pass_sell")] = prob <= (1.0 - min_conf)

    n_valid = len(valid_indices)
    n_buy = result["deep_pass_buy"].sum()
    n_sell = result["deep_pass_sell"].sum()
    logger.info(
        "Deep batch: %d/%d predicted | buy_pass=%d sell_pass=%d",
        n_valid, n_bars, n_buy, n_sell,
    )
    return result


def invalidate_deep_cache() -> bool:
    """Force-reload deep model from disk."""
    return _DeepCache.reload()
