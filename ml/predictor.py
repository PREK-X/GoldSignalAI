"""
GoldSignalAI — ml/predictor.py
================================
Live prediction using the trained XGBoost and Random Forest models.

This module loads saved models from disk, processes the latest candle
data into features, and outputs a prediction with confidence.

Prediction flow:
  1. Load models + scaler + feature columns from disk (cached in memory)
  2. Build features from the latest OHLCV DataFrame (no target)
  3. Scale features using the saved scaler
  4. Predict with XGBoost (primary) and Random Forest (verification)
  5. Both must agree for the prediction to be used
  6. Return MLPrediction with direction, probability, and agreement flag

Confirmation rule used by signals/generator.py:
  - ML must AGREE with the technical signal
  - If ML says UP but technicals say SELL → force WAIT
  - ML prediction is displayed separately in every signal output
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import Config
from ml.features import build_features, build_lgbm_features

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MLPrediction:
    """
    Prediction result from the ML models.

    Attributes:
        direction       : "UP" | "DOWN" | "UNCERTAIN"
        xgb_probability : XGBoost probability of UP (0.0–1.0)
        rf_probability  : Random Forest probability of UP (0.0–1.0)
        avg_probability : Average of both probabilities
        models_agree    : True if both models predict same direction
        confidence_pct  : Confidence as a percentage (0–100)
        available       : True if models are loaded and prediction succeeded
        reason          : Explanation string
    """
    direction:       str    = "UNCERTAIN"
    xgb_probability: float  = 0.5
    rf_probability:  float  = 0.5
    avg_probability: float  = 0.5
    models_agree:    bool   = False
    confidence_pct:  float  = 50.0
    available:       bool   = False
    reason:          str    = ""

    @property
    def confirms_buy(self) -> bool:
        """True if ML predicts UP with agreement."""
        return self.direction == "UP" and self.models_agree

    @property
    def confirms_sell(self) -> bool:
        """True if ML predicts DOWN with agreement."""
        return self.direction == "DOWN" and self.models_agree

    def confirms(self, signal_direction: str) -> bool:
        """
        Check if ML confirms a technical signal direction.

        Args:
            signal_direction: "BUY" or "SELL" from the scoring engine.

        Returns:
            True if ML agrees with the technical signal.
        """
        if not self.available or not self.models_agree:
            return False
        if signal_direction == "BUY" and self.direction == "UP":
            return True
        if signal_direction == "SELL" and self.direction == "DOWN":
            return True
        return False

    def summary(self) -> str:
        return (
            f"ML: {self.direction} ({self.confidence_pct:.0f}%) | "
            f"XGB={self.xgb_probability:.1%} RF={self.rf_probability:.1%} | "
            f"agree={self.models_agree}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────────────────────────────────────────

class _ModelCache:
    """
    In-memory cache for loaded models. Avoids disk I/O on every
    prediction cycle (every 15 minutes). Models are reloaded when
    `reload()` is called (after a retrain).
    """
    xgb_model:       object = None
    rf_model:        object = None
    scaler:          Optional[StandardScaler] = None
    feature_columns: Optional[list[str]] = None
    loaded:          bool = False

    @classmethod
    def load(cls) -> bool:
        """
        Load all model artifacts from disk into memory.
        Returns True if all artifacts loaded successfully.
        """
        xgb_path    = os.path.join(Config.BASE_DIR, Config.ML_MODEL_PATH)
        rf_path     = os.path.join(Config.BASE_DIR, Config.ML_RF_MODEL_PATH)
        scaler_path = os.path.join(Config.BASE_DIR, Config.ML_SCALER_PATH)
        cols_path   = os.path.join(Config.MODELS_DIR, "feature_columns.joblib")

        try:
            if not all(os.path.isfile(p) for p in [xgb_path, rf_path, scaler_path, cols_path]):
                missing = [p for p in [xgb_path, rf_path, scaler_path, cols_path]
                           if not os.path.isfile(p)]
                logger.warning("ML model files missing: %s", missing)
                cls.loaded = False
                return False

            cls.xgb_model       = joblib.load(xgb_path)
            cls.rf_model        = joblib.load(rf_path)
            cls.scaler          = joblib.load(scaler_path)
            cls.feature_columns = joblib.load(cols_path)
            cls.loaded          = True

            logger.info(
                "ML models loaded: XGB=%s RF=%s | %d features",
                type(cls.xgb_model).__name__,
                type(cls.rf_model).__name__,
                len(cls.feature_columns),
            )
            return True

        except Exception as exc:
            logger.exception("Failed to load ML models: %s", exc)
            cls.loaded = False
            return False

    @classmethod
    def reload(cls) -> bool:
        """Force reload from disk (called after retraining)."""
        cls.loaded = False
        return cls.load()

    @classmethod
    def ensure_loaded(cls) -> bool:
        """Load if not already loaded. Returns True if ready."""
        if cls.loaded:
            return True
        return cls.load()


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame) -> MLPrediction:
    """
    Generate an ML prediction for the latest candle in the DataFrame.

    This is the primary function called by signals/generator.py on
    every signal cycle.

    Args:
        df: Processed OHLCV DataFrame (same as used by indicators.py).
            Must have enough rows for feature engineering (~250+).

    Returns:
        MLPrediction with direction, probabilities, and agreement flag.
        If models aren't available or prediction fails, returns a
        prediction with available=False and direction="UNCERTAIN".
    """
    # ── Load models ───────────────────────────────────────────────────────
    if not _ModelCache.ensure_loaded():
        return MLPrediction(
            available=False,
            reason="ML models not trained yet — run trainer.py first"
        )

    # ── Build features for the latest candle ──────────────────────────────
    try:
        feat_df = build_features(df, include_target=False, dropna=True)
    except Exception as exc:
        logger.exception("ML feature engineering failed: %s", exc)
        return MLPrediction(available=False, reason=f"Feature error: {exc}")

    if feat_df.empty:
        return MLPrediction(
            available=False,
            reason="Feature matrix empty — insufficient data for ML"
        )

    # ── Align columns to match training order ─────────────────────────────
    expected_cols = _ModelCache.feature_columns
    missing = set(expected_cols) - set(feat_df.columns)
    if missing:
        logger.warning("ML prediction: missing features %s — filling with 0", missing)
        for col in missing:
            feat_df[col] = 0.0

    # Select and order columns exactly as during training
    X = feat_df[expected_cols].iloc[-1:]   # last row only

    # ── Scale features ────────────────────────────────────────────────────
    try:
        X_scaled = pd.DataFrame(
            _ModelCache.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )
    except Exception as exc:
        logger.exception("ML scaling failed: %s", exc)
        return MLPrediction(available=False, reason=f"Scaler error: {exc}")

    # ── Predict with both models ──────────────────────────────────────────
    try:
        xgb_proba = _ModelCache.xgb_model.predict_proba(X_scaled)[0]
        rf_proba  = _ModelCache.rf_model.predict_proba(X_scaled)[0]

        # proba[0] = probability of class 0 (DOWN)
        # proba[1] = probability of class 1 (UP)
        xgb_up = float(xgb_proba[1])
        rf_up  = float(rf_proba[1])

        xgb_dir = "UP" if xgb_up > 0.5 else "DOWN"
        rf_dir  = "UP" if rf_up  > 0.5 else "DOWN"

        agree  = (xgb_dir == rf_dir)
        avg_up = (xgb_up + rf_up) / 2

        # Direction: use XGBoost as primary, but only if RF agrees
        if agree:
            direction = xgb_dir
            confidence = abs(avg_up - 0.5) * 200   # 0–100 scale
        else:
            direction = "UNCERTAIN"
            confidence = 0.0

        reason = (
            f"XGB: {xgb_dir} ({xgb_up:.1%}) | RF: {rf_dir} ({rf_up:.1%})"
            f"{' — models agree' if agree else ' — models DISAGREE'}"
        )

    except Exception as exc:
        logger.exception("ML prediction failed: %s", exc)
        return MLPrediction(available=False, reason=f"Prediction error: {exc}")

    result = MLPrediction(
        direction=direction,
        xgb_probability=xgb_up,
        rf_probability=rf_up,
        avg_probability=avg_up,
        models_agree=agree,
        confidence_pct=confidence,
        available=True,
        reason=reason,
    )

    logger.info("ML prediction: %s", result.summary())
    return result


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for ALL rows in the DataFrame (for backtesting).

    Returns a DataFrame with columns:
      xgb_prob, rf_prob, ml_direction, ml_agree

    Rows where features can't be computed are filled with NaN / "UNCERTAIN".
    """
    if not _ModelCache.ensure_loaded():
        logger.error("Cannot run batch prediction — models not loaded.")
        return pd.DataFrame()

    feat_df = build_features(df, include_target=False, dropna=False)

    # Drop rows with NaN for prediction, but track their indices
    valid_mask = ~feat_df.isnull().any(axis=1)
    feat_valid = feat_df[valid_mask].copy()

    if feat_valid.empty:
        logger.warning("Batch prediction: no valid rows after feature engineering.")
        return pd.DataFrame()

    expected_cols = _ModelCache.feature_columns
    missing = set(expected_cols) - set(feat_valid.columns)
    for col in missing:
        feat_valid[col] = 0.0
    X = feat_valid[expected_cols]

    X_scaled = pd.DataFrame(
        _ModelCache.scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )

    xgb_proba = _ModelCache.xgb_model.predict_proba(X_scaled)[:, 1]
    rf_proba  = _ModelCache.rf_model.predict_proba(X_scaled)[:, 1]

    result = pd.DataFrame(index=df.index)
    result["xgb_prob"]     = np.nan
    result["rf_prob"]      = np.nan
    result["ml_direction"] = "UNCERTAIN"
    result["ml_agree"]     = False

    result.loc[feat_valid.index, "xgb_prob"] = xgb_proba
    result.loc[feat_valid.index, "rf_prob"]  = rf_proba

    xgb_dir = np.where(xgb_proba > 0.5, "UP", "DOWN")
    rf_dir  = np.where(rf_proba > 0.5, "UP", "DOWN")
    agree   = xgb_dir == rf_dir

    directions = np.where(agree, xgb_dir, "UNCERTAIN")
    result.loc[feat_valid.index, "ml_direction"] = directions
    result.loc[feat_valid.index, "ml_agree"]     = agree

    logger.info(
        "Batch prediction: %d/%d rows predicted | UP=%d DOWN=%d UNCERTAIN=%d",
        len(feat_valid), len(df),
        (directions == "UP").sum(),
        (directions == "DOWN").sum(),
        (directions == "UNCERTAIN").sum(),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def is_model_ready() -> bool:
    """Check if ML models are loaded and ready for prediction."""
    return _ModelCache.ensure_loaded()


def get_model_info() -> dict:
    """Return model metadata for the dashboard."""
    _ModelCache.ensure_loaded()
    return {
        "loaded":          _ModelCache.loaded,
        "xgb_type":        type(_ModelCache.xgb_model).__name__ if _ModelCache.xgb_model else None,
        "rf_type":         type(_ModelCache.rf_model).__name__  if _ModelCache.rf_model  else None,
        "n_features":      len(_ModelCache.feature_columns) if _ModelCache.feature_columns else 0,
        "feature_columns": _ModelCache.feature_columns or [],
    }


def invalidate_cache() -> bool:
    """
    Force-reload ML models from disk. Called by scheduler after retraining.
    Returns True if models reloaded successfully.
    """
    return _ModelCache.reload()


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5: LIGHTGBM DIRECTION CLASSIFIER — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LGBMPrediction:
    """
    Prediction result from the LightGBM direction classifier.

    Attributes:
        direction:    "BUY_OK" | "SELL_OK" | "SKIP"
        probability:  P(profitable) from LightGBM [0.0–1.0]
        available:    True if model loaded and prediction succeeded
        reason:       Explanation string
    """
    direction:   str   = "SKIP"
    probability: float = 0.5
    available:   bool  = False
    reason:      str   = ""

    def confirms(self, signal_direction: str) -> bool:
        """
        Check if LGBM confirms a technical signal direction.
        LGBM doesn't predict direction — it predicts whether the trade
        will be profitable. So it confirms if probability >= threshold.
        """
        if not self.available:
            return True  # If not available, don't block
        return self.probability >= Config.LGBM_MIN_PROBABILITY

    def summary(self) -> str:
        return (
            f"LGBM: {self.direction} (P={self.probability:.1%}) | "
            f"threshold={Config.LGBM_MIN_PROBABILITY:.0%}"
        )


class _LGBMCache:
    """In-memory cache for the LightGBM model artifacts."""
    model: object = None
    scaler: Optional[StandardScaler] = None
    feature_columns: Optional[list[str]] = None
    loaded: bool = False

    @classmethod
    def load(cls) -> bool:
        model_path = os.path.join(Config.BASE_DIR, Config.LGBM_MODEL_PATH)
        scaler_path = os.path.join(Config.BASE_DIR, Config.LGBM_SCALER_PATH)
        cols_path = os.path.join(Config.BASE_DIR, Config.LGBM_FEATURES_PATH)

        try:
            if not all(os.path.isfile(p) for p in [model_path, scaler_path, cols_path]):
                missing = [p for p in [model_path, scaler_path, cols_path]
                           if not os.path.isfile(p)]
                logger.debug("LGBM model files missing: %s", missing)
                cls.loaded = False
                return False

            cls.model = joblib.load(model_path)
            cls.scaler = joblib.load(scaler_path)
            cls.feature_columns = joblib.load(cols_path)
            cls.loaded = True

            logger.info(
                "LGBM model loaded: %s | %d features",
                type(cls.model).__name__,
                len(cls.feature_columns),
            )
            return True

        except Exception as exc:
            logger.exception("Failed to load LGBM model: %s", exc)
            cls.loaded = False
            return False

    @classmethod
    def reload(cls) -> bool:
        cls.loaded = False
        return cls.load()

    @classmethod
    def ensure_loaded(cls) -> bool:
        if cls.loaded:
            return True
        return cls.load()


def predict_lgbm(df: pd.DataFrame) -> LGBMPrediction:
    """
    Generate an LGBM prediction for the latest candle.

    This predicts P(trade will be profitable), not direction.
    If P >= LGBM_MIN_PROBABILITY, the trade is allowed.

    Args:
        df: Processed OHLCV DataFrame (250+ rows for feature warm-up)

    Returns:
        LGBMPrediction with probability and filter decision.
    """
    if not _LGBMCache.ensure_loaded():
        return LGBMPrediction(
            available=False,
            reason="LGBM model not trained — run: venv/bin/python -m ml.trainer"
        )

    try:
        feat_df = build_lgbm_features(df, include_target=False, dropna=True)
    except Exception as exc:
        logger.exception("LGBM feature engineering failed: %s", exc)
        return LGBMPrediction(available=False, reason=f"Feature error: {exc}")

    if feat_df.empty:
        return LGBMPrediction(
            available=False,
            reason="LGBM feature matrix empty — insufficient data"
        )

    # Align columns
    expected_cols = _LGBMCache.feature_columns
    missing = set(expected_cols) - set(feat_df.columns)
    if missing:
        logger.warning("LGBM prediction: missing features %s — filling with 0", missing)
        for col in missing:
            feat_df[col] = 0.0

    X = feat_df[expected_cols].iloc[-1:]

    try:
        X_scaled = pd.DataFrame(
            _LGBMCache.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )
    except Exception as exc:
        logger.exception("LGBM scaling failed: %s", exc)
        return LGBMPrediction(available=False, reason=f"Scaler error: {exc}")

    try:
        proba = _LGBMCache.model.predict_proba(X_scaled)[0]
        p_profitable = float(proba[1])  # P(class 1 = profitable)

        if p_profitable >= Config.LGBM_MIN_PROBABILITY:
            direction = "TRADE_OK"
            reason = f"P(profitable)={p_profitable:.1%} >= {Config.LGBM_MIN_PROBABILITY:.0%} threshold"
        else:
            direction = "SKIP"
            reason = f"P(profitable)={p_profitable:.1%} < {Config.LGBM_MIN_PROBABILITY:.0%} threshold"

    except Exception as exc:
        logger.exception("LGBM prediction failed: %s", exc)
        return LGBMPrediction(available=False, reason=f"Prediction error: {exc}")

    result = LGBMPrediction(
        direction=direction,
        probability=p_profitable,
        available=True,
        reason=reason,
    )

    logger.info("LGBM prediction: %s", result.summary())
    return result


def predict_lgbm_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate LGBM predictions for ALL rows (for backtesting).

    Returns DataFrame with columns: lgbm_prob, lgbm_pass
    """
    if not _LGBMCache.ensure_loaded():
        logger.error("Cannot run LGBM batch prediction — model not loaded.")
        return pd.DataFrame()

    feat_df = build_lgbm_features(df, include_target=False, dropna=False)

    valid_mask = ~feat_df.isnull().any(axis=1)
    feat_valid = feat_df[valid_mask].copy()

    if feat_valid.empty:
        logger.warning("LGBM batch prediction: no valid rows.")
        return pd.DataFrame()

    expected_cols = _LGBMCache.feature_columns
    missing = set(expected_cols) - set(feat_valid.columns)
    for col in missing:
        feat_valid[col] = 0.0
    X = feat_valid[expected_cols]

    X_scaled = pd.DataFrame(
        _LGBMCache.scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )

    proba = _LGBMCache.model.predict_proba(X_scaled)[:, 1]

    result = pd.DataFrame(index=df.index)
    result["lgbm_prob"] = np.nan
    result["lgbm_pass"] = False

    result.loc[feat_valid.index, "lgbm_prob"] = proba
    result.loc[feat_valid.index, "lgbm_pass"] = proba >= Config.LGBM_MIN_PROBABILITY

    n_pass = result["lgbm_pass"].sum()
    n_total = len(feat_valid)
    logger.info(
        "LGBM batch: %d/%d rows predicted | pass=%d (%.1f%%) skip=%d",
        n_total, len(df), n_pass, n_pass / n_total * 100 if n_total else 0,
        n_total - n_pass,
    )
    return result


def is_lgbm_ready() -> bool:
    """Check if LGBM model is loaded and ready."""
    return _LGBMCache.ensure_loaded()


def invalidate_lgbm_cache() -> bool:
    """Force-reload LGBM model from disk."""
    return _LGBMCache.reload()
