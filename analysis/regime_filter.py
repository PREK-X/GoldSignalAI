"""
GoldSignalAI -- analysis/regime_filter.py
==========================================
Hidden Markov Model (HMM) regime detection for XAU/USD.

Uses a 3-state GaussianHMM trained on H1 data to classify the current
market environment into one of:

  State 0  TRENDING   low volatility, directional moves   → full size (1.0x)
  State 1  RANGING    medium volatility, mean-reverting   → half size (0.5x)
  State 2  CRISIS     high volatility, chaotic            → no trading (0.0x)

States are labeled *post-training* by sorting the learned state means
by realized volatility (ascending).  This guarantees a stable mapping
regardless of which internal label hmmlearn assigns.

Input features (computed from H1 OHLCV):
  1. Log returns:  ln(close_t / close_{t-1})
  2. Realized vol: rolling 14-bar standard deviation of log returns

Why HMM:
  - Market regimes are *latent* — we observe price, not the regime.
  - HMM captures regime persistence (transition matrix) and distinct
    volatility signatures (emission distributions).
  - 3 states map naturally to trading action: go / reduce / stop.

Usage:
    from analysis.regime_filter import RegimeDetector, get_current_regime

    # Train + predict in one shot (backtest)
    detector = RegimeDetector()
    detector.fit(h1_df)
    state, label, mult = detector.predict_current(h1_df)

    # Quick one-shot for live trading
    state, label, mult = get_current_regime(h1_df)
"""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------

N_STATES = 3
VOL_WINDOW = 14  # rolling window for realized volatility
MIN_BARS_FOR_FIT = 200  # minimum H1 bars to train a useful model

# State labels and position size multipliers, indexed by volatility rank
_LABELS = {0: "TRENDING", 1: "RANGING", 2: "CRISIS"}
_MULTIPLIERS = {0: 1.0, 1: 0.5, 2: 0.0}

_MODEL_PATH = os.path.join(Config.BASE_DIR, "models", "hmm_regime.pkl")


# -------------------------------------------------------------------------
# RESULT TYPE
# -------------------------------------------------------------------------

@dataclass
class RegimeState:
    """Current regime classification."""
    state: int           # 0=TRENDING, 1=RANGING, 2=CRISIS (volatility-ranked)
    label: str           # Human-readable label
    multiplier: float    # Position size multiplier (1.0 / 0.5 / 0.0)
    log_return: float    # Latest log return used for prediction
    realized_vol: float  # Latest realized volatility used for prediction


# -------------------------------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------------------------------

def _extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract HMM input features from H1 OHLCV data.

    Returns:
        2D array of shape (n_samples, 2) with columns:
          [log_return, realized_vol]
        Rows with NaN (warm-up period) are dropped.
    """
    close = df["close"].values.astype(float)

    # Log returns: ln(close_t / close_{t-1})
    log_returns = np.log(close[1:] / close[:-1])

    # Realized volatility: rolling std of log returns over VOL_WINDOW bars
    realized_vol = np.full_like(log_returns, np.nan)
    for i in range(VOL_WINDOW - 1, len(log_returns)):
        window = log_returns[i - VOL_WINDOW + 1 : i + 1]
        realized_vol[i] = np.std(window, ddof=1)

    # Stack into feature matrix and drop NaN rows
    features = np.column_stack([log_returns, realized_vol])
    valid_mask = ~np.isnan(features).any(axis=1)
    return features[valid_mask]


def _extract_features_with_index(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Like _extract_features but also returns the integer indices into
    the original DataFrame that correspond to each valid feature row.
    Used for aligning predictions back to bars.
    """
    close = df["close"].values.astype(float)
    n = len(close)

    log_returns = np.log(close[1:] / close[:-1])
    realized_vol = np.full_like(log_returns, np.nan)
    for i in range(VOL_WINDOW - 1, len(log_returns)):
        window = log_returns[i - VOL_WINDOW + 1 : i + 1]
        realized_vol[i] = np.std(window, ddof=1)

    features = np.column_stack([log_returns, realized_vol])

    # Indices into original df: log_returns[i] corresponds to df.iloc[i+1]
    orig_indices = np.arange(1, n)

    valid_mask = ~np.isnan(features).any(axis=1)
    return features[valid_mask], orig_indices[valid_mask]


# -------------------------------------------------------------------------
# REGIME DETECTOR
# -------------------------------------------------------------------------

class RegimeDetector:
    """
    Train a GaussianHMM on H1 data and predict market regime states.

    The detector handles the full lifecycle:
      1. fit(h1_df)              → train the HMM
      2. predict_current(h1_df)  → classify the latest bar
      3. predict_all(h1_df)      → classify every bar (for backtest)
      4. save() / load()         → persist/restore the trained model
    """

    def __init__(self):
        self._model = None
        self._vol_sort_map: Optional[dict[int, int]] = None  # raw_state → ranked_state
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, h1_df: pd.DataFrame, n_iter: int = 100) -> bool:
        """
        Train the HMM on H1 OHLCV data.

        Args:
            h1_df:  H1 DataFrame with at least 'close' column.
            n_iter: Maximum EM iterations.

        Returns:
            True if training succeeded, False otherwise.
        """
        features = _extract_features(h1_df)

        if len(features) < MIN_BARS_FOR_FIT:
            logger.warning(
                "Not enough H1 bars for HMM: %d < %d required",
                len(features), MIN_BARS_FOR_FIT,
            )
            return False

        try:
            from hmmlearn.hmm import GaussianHMM

            # Try "full" covariance first; fall back to "diag" if singular
            model = None
            for cov_type in ("full", "diag"):
                try:
                    m = GaussianHMM(
                        n_components=N_STATES,
                        covariance_type=cov_type,
                        n_iter=n_iter,
                        random_state=42,
                        verbose=False,
                    )
                    m.fit(features)
                    model = m
                    break
                except ValueError as ve:
                    if "positive-definite" in str(ve) and cov_type == "full":
                        logger.info("Full covariance singular, falling back to diag")
                        continue
                    raise

            if model is None:
                logger.error("HMM training failed: could not fit any covariance type")
                return False

            self._model = model

            # Build volatility-rank mapping: sort states by their mean
            # realized volatility (column index 1 in feature matrix)
            vol_means = model.means_[:, 1]  # mean realized_vol per state
            sorted_indices = np.argsort(vol_means)  # ascending volatility

            # Map: raw hmmlearn state → our volatility-ranked state
            self._vol_sort_map = {}
            for rank, raw_state in enumerate(sorted_indices):
                self._vol_sort_map[int(raw_state)] = rank

            self._is_fitted = True

            logger.info(
                "HMM trained on %d H1 bars. State volatility means: %s",
                len(features),
                {_LABELS[self._vol_sort_map[i]]: f"{vol_means[i]:.6f}"
                 for i in range(N_STATES)},
            )
            return True

        except Exception as exc:
            logger.error("HMM training failed: %s", exc)
            self._is_fitted = False
            return False

    def _map_state(self, raw_state: int) -> int:
        """Map a raw hmmlearn state to our volatility-ranked state."""
        if self._vol_sort_map is None:
            return raw_state
        return self._vol_sort_map.get(int(raw_state), raw_state)

    def predict_current(self, h1_df: pd.DataFrame) -> RegimeState:
        """
        Predict the regime for the most recent bar.

        Returns RegimeState with the classification.
        Falls back to RANGING (state=1) if model is not fitted.
        """
        if not self._is_fitted or self._model is None:
            return RegimeState(
                state=1, label="RANGING", multiplier=0.5,
                log_return=0.0, realized_vol=0.0,
            )

        features = _extract_features(h1_df)
        if len(features) == 0:
            return RegimeState(
                state=1, label="RANGING", multiplier=0.5,
                log_return=0.0, realized_vol=0.0,
            )

        # Predict on the last bar
        raw_state = int(self._model.predict(features[-1:].reshape(1, -1))[0])
        ranked = self._map_state(raw_state)

        return RegimeState(
            state=ranked,
            label=_LABELS[ranked],
            multiplier=_MULTIPLIERS[ranked],
            log_return=float(features[-1, 0]),
            realized_vol=float(features[-1, 1]),
        )

    def predict_all(self, h1_df: pd.DataFrame) -> np.ndarray:
        """
        Predict regime states for ALL bars in the DataFrame.

        Returns an array of ranked states (0/1/2) of length len(h1_df).
        Bars in the warm-up period (before enough data for features)
        are assigned state 1 (RANGING) as a safe default.
        """
        if not self._is_fitted or self._model is None:
            return np.full(len(h1_df), 1)  # default RANGING

        features, orig_indices = _extract_features_with_index(h1_df)
        if len(features) == 0:
            return np.full(len(h1_df), 1)

        raw_states = self._model.predict(features)
        ranked_states = np.array([self._map_state(int(s)) for s in raw_states])

        # Map back to full DataFrame length
        all_states = np.full(len(h1_df), 1)  # default RANGING for warm-up
        all_states[orig_indices] = ranked_states

        return all_states

    def save(self, path: str = _MODEL_PATH) -> bool:
        """Save the trained model to disk."""
        if not self._is_fitted:
            logger.warning("Cannot save: model not fitted.")
            return False
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "vol_sort_map": self._vol_sort_map,
                }, f)
            logger.info("HMM model saved to %s", path)
            return True
        except Exception as exc:
            logger.error("Failed to save HMM model: %s", exc)
            return False

    def load(self, path: str = _MODEL_PATH) -> bool:
        """Load a previously trained model from disk."""
        if not os.path.isfile(path):
            logger.warning("HMM model file not found: %s", path)
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._model = data["model"]
            self._vol_sort_map = data["vol_sort_map"]
            self._is_fitted = True
            logger.info("HMM model loaded from %s", path)
            return True
        except Exception as exc:
            logger.error("Failed to load HMM model: %s", exc)
            return False


# -------------------------------------------------------------------------
# MODULE-LEVEL SINGLETON (for live trading)
# -------------------------------------------------------------------------

_detector: Optional[RegimeDetector] = None


def get_current_regime(h1_df: pd.DataFrame) -> tuple[int, str, float]:
    """
    One-shot convenience function for live trading.

    Trains the HMM if not already fitted (uses all available H1 data),
    then predicts the current regime.

    Returns:
        (state_int, label_str, size_multiplier)
        e.g. (0, "TRENDING", 1.0) or (2, "CRISIS", 0.0)
    """
    global _detector

    if _detector is None:
        _detector = RegimeDetector()

    # Try loading a saved model first
    if not _detector.is_fitted:
        _detector.load()

    # If still not fitted, train on available data
    if not _detector.is_fitted:
        logger.info("Training HMM regime detector on %d H1 bars...", len(h1_df))
        _detector.fit(h1_df)
        _detector.save()

    result = _detector.predict_current(h1_df)
    return result.state, result.label, result.multiplier


def is_hmmlearn_available() -> tuple[bool, str]:
    """
    Check if hmmlearn is installed and GaussianHMM can be instantiated.
    Used by the health check system.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
        # Verify we can create an instance
        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1)
        return True, "hmmlearn installed, GaussianHMM available"
    except ImportError:
        return False, "hmmlearn not installed. Run: pip install hmmlearn"
    except Exception as exc:
        return False, f"hmmlearn error: {exc}"
