"""
GoldSignalAI — ml/retrainer.py
================================
Stage 13: ML Auto-Retraining Pipeline.

Handles weekly LightGBM retraining and CNN-BiLSTM retrain once 150+
real trade outcomes are available.

Usage (from scheduler):
    retrainer = ModelRetrainer()
    if retrainer.should_retrain_lgbm():
        result = retrainer.retrain_lgbm()

Design:
  - Always backs up current model files before training
  - Only deploys if new_cv >= RETRAIN_LGBM_MIN_ACCURACY AND >= old_accuracy - 0.01
  - On failure or gate miss, restores backup so live trading is never interrupted
  - State persisted to state/retrain_state.json (survives restarts)
"""

import json
import logging
import os
import shutil
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STATE FILE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_STATE: dict = {
    "lgbm": {
        "last_retrain": None,
        "last_accuracy": None,
        "deployed": False,
        "retrain_count": 0,
    },
    "deep": {
        "last_retrain": None,
        "last_accuracy": None,
        "deployed": False,
        "retrain_count": 0,
        "trade_outcomes_available": 0,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL RETRAINER
# ─────────────────────────────────────────────────────────────────────────────

class ModelRetrainer:
    """
    Orchestrates LightGBM and CNN-BiLSTM retraining with safety gates.

    Safety contract:
      - Always backs up current model before touching it.
      - Restores backup if new model fails quality gate or any exception.
      - Never leaves the models directory in a partial state.
    """

    def __init__(self):
        os.makedirs(Config.RETRAIN_BACKUP_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(Config.RETRAIN_STATE_FILE), exist_ok=True)

    # ── State management ───────────────────────────────────────────────────

    def load_state(self) -> dict:
        """Load retrain state from disk. Returns default state if missing."""
        try:
            if os.path.isfile(Config.RETRAIN_STATE_FILE):
                with open(Config.RETRAIN_STATE_FILE) as f:
                    loaded = json.load(f)
                # Merge with defaults to handle new keys added in future
                state = {
                    "lgbm": {**_DEFAULT_STATE["lgbm"], **loaded.get("lgbm", {})},
                    "deep": {**_DEFAULT_STATE["deep"], **loaded.get("deep", {})},
                }
                return state
        except Exception as exc:
            logger.warning("Could not load retrain state: %s — using defaults", exc)
        return {
            "lgbm": dict(_DEFAULT_STATE["lgbm"]),
            "deep": dict(_DEFAULT_STATE["deep"]),
        }

    def save_state(self, state: dict) -> None:
        """Persist retrain state to disk."""
        try:
            os.makedirs(os.path.dirname(Config.RETRAIN_STATE_FILE), exist_ok=True)
            with open(Config.RETRAIN_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Could not save retrain state: %s", exc)

    def should_retrain_lgbm(self) -> bool:
        """
        Return True if enough days have passed since the last LGBM retrain.
        Always returns True if no retrain has ever occurred.
        """
        if not Config.RETRAIN_LGBM_ENABLED:
            return False
        state = self.load_state()
        last_retrain_str = state["lgbm"].get("last_retrain")
        if not last_retrain_str:
            return True
        try:
            last_retrain = datetime.fromisoformat(last_retrain_str)
            if last_retrain.tzinfo is None:
                last_retrain = last_retrain.replace(tzinfo=timezone.utc)
            elapsed_days = (datetime.now(timezone.utc) - last_retrain).total_seconds() / 86400
            return elapsed_days >= Config.RETRAIN_LGBM_INTERVAL_DAYS
        except Exception as exc:
            logger.warning("Could not parse last_retrain timestamp: %s — retraining", exc)
            return True

    def get_trade_outcome_count(self) -> int:
        """
        Count live trade outcomes in SQLite.

        Returns rows where result IS NOT NULL AND status = 'closed'.
        Only live trades are stored in SQLite (backtest uses CSV/in-memory),
        so this count reflects real completed forward-test trades.
        """
        db_path = os.path.join(Config.BASE_DIR, "database", "goldsignalai.db")
        if not os.path.isfile(db_path):
            return 0
        try:
            conn = sqlite3.connect(db_path, timeout=5)
            row = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE result IS NOT NULL AND status = 'closed'"
            ).fetchone()
            conn.close()
            return int(row[0]) if row else 0
        except Exception as exc:
            logger.warning("Could not count trade outcomes: %s", exc)
            return 0

    # ── Backup helpers ─────────────────────────────────────────────────────

    def _backup_lgbm(self, timestamp_str: str) -> dict[str, str]:
        """
        Copy current LGBM model files to the backups directory.

        Returns a dict mapping original path → backup path (only for files
        that actually existed). Empty dict means nothing to restore.
        """
        files = {
            os.path.join(Config.BASE_DIR, Config.LGBM_MODEL_PATH):
                os.path.join(Config.RETRAIN_BACKUP_DIR, f"lgbm_{timestamp_str}.pkl"),
            os.path.join(Config.BASE_DIR, Config.LGBM_SCALER_PATH):
                os.path.join(Config.RETRAIN_BACKUP_DIR, f"lgbm_scaler_{timestamp_str}.pkl"),
            os.path.join(Config.BASE_DIR, Config.LGBM_FEATURES_PATH):
                os.path.join(Config.RETRAIN_BACKUP_DIR, f"lgbm_features_{timestamp_str}.joblib"),
        }
        backed_up = {}
        for src, dst in files.items():
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                backed_up[src] = dst
                logger.info("Backup: %s → %s", os.path.basename(src), os.path.basename(dst))
        return backed_up

    def _restore_lgbm(self, backups: dict[str, str]) -> None:
        """Restore LGBM model files from backup."""
        for original, backup in backups.items():
            if os.path.isfile(backup):
                shutil.copy2(backup, original)
                logger.info("Restored: %s from backup", os.path.basename(original))

    def _backup_deep(self, timestamp_str: str) -> dict[str, str]:
        """Copy current deep model files to the backups directory."""
        files = {
            os.path.join(Config.BASE_DIR, Config.DEEP_MODEL_PATH):
                os.path.join(Config.RETRAIN_BACKUP_DIR, f"deep_model_{timestamp_str}.keras"),
            os.path.join(Config.BASE_DIR, Config.DEEP_SCALER_PATH):
                os.path.join(Config.RETRAIN_BACKUP_DIR, f"deep_scaler_{timestamp_str}.pkl"),
        }
        backed_up = {}
        for src, dst in files.items():
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                backed_up[src] = dst
                logger.info("Backup: %s → %s", os.path.basename(src), os.path.basename(dst))
        return backed_up

    def _restore_deep(self, backups: dict[str, str]) -> None:
        """Restore deep model files from backup."""
        for original, backup in backups.items():
            if os.path.isfile(backup):
                shutil.copy2(backup, original)
                logger.info("Restored: %s from backup", os.path.basename(original))

    # ── LightGBM retrain ───────────────────────────────────────────────────

    def retrain_lgbm(self) -> dict:
        """
        Retrain LightGBM with latest 2yr data.

        Steps:
          1. Back up current model files
          2. Fetch 2yr M15 data + macro
          3. Walk-forward CV via ml.trainer.train_lgbm(save=False)
          4. Deploy if new_cv >= RETRAIN_LGBM_MIN_ACCURACY AND >= old - 0.01
          5. Update retrain_state.json

        Returns:
            {
                'success': bool,
                'old_accuracy': float,
                'new_accuracy': float,
                'deployed': bool,
                'reason': str,
                'timestamp': str,
                'backup_paths': dict,
            }
        """
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        now_iso = datetime.now(timezone.utc).isoformat()

        state = self.load_state()
        old_accuracy = state["lgbm"].get("last_accuracy") or 0.0

        result = {
            "success": False,
            "old_accuracy": old_accuracy,
            "new_accuracy": 0.0,
            "deployed": False,
            "reason": "",
            "timestamp": now_iso,
            "backup_paths": {},
        }

        # Step 1: Backup
        backups = self._backup_lgbm(timestamp_str)
        result["backup_paths"] = backups

        try:
            # Step 2: Fetch data
            logger.info("[Retrainer] Fetching 2yr M15 data for LGBM retrain...")
            from data.fetcher import get_candles
            from data.processor import process

            raw = get_candles(timeframe="M15", n_candles=47000, symbol=Config.SYMBOL)
            if raw is None or raw.empty:
                result["reason"] = "Data fetch failed — no M15 candles returned"
                logger.error("[Retrainer] %s", result["reason"])
                return result

            df = process(raw, timeframe="M15", label="LGBM_RETRAIN")
            if df is None or df.empty:
                result["reason"] = "Data processing failed"
                logger.error("[Retrainer] %s", result["reason"])
                return result

            logger.info("[Retrainer] Fetched %d M15 bars", len(df))

            # Fetch macro data
            try:
                from data.macro_fetcher import fetch_and_cache_macro
                fetch_and_cache_macro(years=3)
            except Exception as exc:
                logger.warning("[Retrainer] Macro fetch failed (non-fatal): %s", exc)

            # Step 3: Run walk-forward CV — do NOT save to disk yet
            logger.info("[Retrainer] Running LGBM walk-forward CV...")
            from ml.trainer import train_lgbm
            training_result = train_lgbm(df, save=False)

            new_accuracy = training_result.cv_accuracy
            result["new_accuracy"] = new_accuracy

            if training_result.rejected:
                result["reason"] = f"Training rejected: {training_result.reject_reason}"
                logger.warning("[Retrainer] %s", result["reason"])
                self._restore_lgbm(backups)
                return result

            # Step 4: Deploy gate
            min_acc = Config.RETRAIN_LGBM_MIN_ACCURACY
            regression_guard = old_accuracy - 0.01  # allow 1% slack

            if new_accuracy >= min_acc and new_accuracy >= regression_guard:
                # Re-run with save=True to write model files
                logger.info(
                    "[Retrainer] Gate passed (%.1f%% >= %.0f%% and >= %.1f%%) — deploying...",
                    new_accuracy * 100, min_acc * 100, regression_guard * 100,
                )
                train_lgbm(df, save=True)

                # Reload predictor cache
                from ml.predictor import reload_lgbm_model
                reload_lgbm_model()

                result["deployed"] = True
                result["success"] = True
                result["reason"] = f"Deployed: CV {new_accuracy:.1%} >= {min_acc:.0%} gate"
                logger.info("[Retrainer] LGBM deployed: %.1f%% accuracy", new_accuracy * 100)

            else:
                if new_accuracy < min_acc:
                    result["reason"] = (
                        f"Below minimum threshold: {new_accuracy:.1%} < {min_acc:.0%}"
                    )
                else:
                    result["reason"] = (
                        f"Accuracy regression: {new_accuracy:.1%} < old {old_accuracy:.1%} - 1%"
                    )
                logger.warning("[Retrainer] Not deploying — %s", result["reason"])
                self._restore_lgbm(backups)
                result["success"] = True  # run succeeded, just didn't deploy

        except Exception as exc:
            result["reason"] = f"Unexpected error: {exc}"
            logger.exception("[Retrainer] LGBM retrain crashed: %s", exc)
            self._restore_lgbm(backups)

        # Step 5: Update state
        state["lgbm"]["last_retrain"] = now_iso
        state["lgbm"]["last_accuracy"] = result["new_accuracy"] if result["deployed"] else old_accuracy
        state["lgbm"]["deployed"] = result["deployed"]
        state["lgbm"]["retrain_count"] = state["lgbm"].get("retrain_count", 0) + 1
        self.save_state(state)

        return result

    # ── CNN-BiLSTM retrain ─────────────────────────────────────────────────

    def retrain_deep_if_ready(self) -> Optional[dict]:
        """
        Retrain the CNN-BiLSTM if 150+ real trade outcomes are available.

        Returns None if insufficient trade outcomes.
        Returns same dict structure as retrain_lgbm() otherwise.
        """
        if not Config.RETRAIN_DEEP_ENABLED:
            logger.info("[Retrainer] Deep retrain disabled by config.")
            return None

        trade_count = self.get_trade_outcome_count()
        logger.info(
            "[Retrainer] Live trade outcomes: %d / %d required",
            trade_count, Config.RETRAIN_DEEP_MIN_TRADES,
        )

        # Update state with current count regardless
        state = self.load_state()
        state["deep"]["trade_outcomes_available"] = trade_count
        self.save_state(state)

        if trade_count < Config.RETRAIN_DEEP_MIN_TRADES:
            return None

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        now_iso = datetime.now(timezone.utc).isoformat()

        old_accuracy = state["deep"].get("last_accuracy") or 0.0

        result = {
            "success": False,
            "old_accuracy": old_accuracy,
            "new_accuracy": 0.0,
            "deployed": False,
            "reason": "",
            "timestamp": now_iso,
            "backup_paths": {},
        }

        backups = self._backup_deep(timestamp_str)
        result["backup_paths"] = backups

        try:
            import numpy as np
            import joblib
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score

            # Fetch data (same pipeline as deep_trainer.main)
            logger.info("[Retrainer] Fetching data for deep model retrain...")
            from data.fetcher import get_candles
            from data.processor import process
            from data.macro_fetcher import fetch_and_cache_macro, get_macro_series
            from ml.deep_features import build_deep_features, build_sequences
            from ml.deep_model import build_model, train_model, save_model

            raw = get_candles(timeframe="M15", n_candles=47000, symbol=Config.SYMBOL)
            if raw is None or raw.empty:
                result["reason"] = "Data fetch failed"
                self._restore_deep(backups)
                return result

            m15 = process(raw, timeframe="M15", label="DEEP_RETRAIN")
            if m15 is None or m15.empty:
                result["reason"] = "Data processing failed"
                self._restore_deep(backups)
                return result

            try:
                fetch_and_cache_macro(years=3)
                macro_df = get_macro_series()
            except Exception:
                macro_df = None

            # Build sequences
            feat_df = build_deep_features(m15, macro_df=macro_df)
            X, y, _ = build_sequences(
                feat_df,
                lookback=Config.DEEP_LOOKBACK,
                future_bars=3,
                close_series=m15["close"],
            )

            if len(X) < 1000:
                result["reason"] = f"Only {len(X)} sequences (need 1000+)"
                self._restore_deep(backups)
                return result

            # Chronological split (70/15/15)
            n = len(X)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)

            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            X_test, y_test = X[val_end:], y[val_end:]

            # Normalize
            scaler = StandardScaler()
            train_flat = X_train.reshape(-1, X_train.shape[2])
            scaler.fit(train_flat)

            def _scale(arr):
                shape = arr.shape
                return scaler.transform(arr.reshape(-1, shape[2])).reshape(shape).astype(np.float32)

            X_train_s = _scale(X_train)
            X_val_s = _scale(X_val)
            X_test_s = _scale(X_test)

            # Build + train
            model = build_model(Config.DEEP_LOOKBACK, X_train.shape[2])
            model, _ = train_model(
                X_train_s, y_train, X_val_s, y_val,
                epochs=100, batch_size=32, verbose=0,
            )

            # Evaluate
            test_probs = model.predict(X_test_s, batch_size=256, verbose=0).flatten()
            test_preds = (test_probs >= 0.5).astype(int)
            val_accuracy = float(accuracy_score(y_test, test_preds))
            result["new_accuracy"] = val_accuracy

            logger.info("[Retrainer] Deep model test accuracy: %.1f%%", val_accuracy * 100)

            min_acc = Config.RETRAIN_DEEP_MIN_ACCURACY
            if val_accuracy >= min_acc:
                # Save model and scaler
                save_model(model)
                scaler_path = os.path.join(Config.BASE_DIR, Config.DEEP_SCALER_PATH)
                os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
                joblib.dump(scaler, scaler_path)

                # Reload cache
                from ml.deep_predictor import reload_deep_model
                reload_deep_model()

                result["deployed"] = True
                result["success"] = True
                result["reason"] = f"Deployed: test accuracy {val_accuracy:.1%} >= {min_acc:.0%}"
                logger.info("[Retrainer] CNN-BiLSTM deployed: %.1f%%", val_accuracy * 100)
            else:
                result["reason"] = (
                    f"Below minimum threshold: {val_accuracy:.1%} < {min_acc:.0%}"
                )
                logger.warning("[Retrainer] Deep model not deployed — %s", result["reason"])
                self._restore_deep(backups)
                result["success"] = True

        except Exception as exc:
            result["reason"] = f"Unexpected error: {exc}"
            logger.exception("[Retrainer] Deep retrain crashed: %s", exc)
            self._restore_deep(backups)

        # Update state
        state["deep"]["last_retrain"] = now_iso
        state["deep"]["last_accuracy"] = result["new_accuracy"] if result["deployed"] else old_accuracy
        state["deep"]["deployed"] = result["deployed"]
        state["deep"]["retrain_count"] = state["deep"].get("retrain_count", 0) + 1
        state["deep"]["trade_outcomes_available"] = trade_count
        self.save_state(state)

        return result
