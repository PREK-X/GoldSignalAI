"""
GoldSignalAI — ml/deep_trainer.py
====================================
Training script for the CNN-BiLSTM deep model (Stage 7).

Usage:
    venv/bin/python -m ml.deep_trainer

Pipeline:
  1. Fetch M15 data from Polygon (full 2yr history)
  2. Fetch macro data (DXY, VIX, US10Y)
  3. Build feature matrix → 60-bar sliding windows
  4. Split: 70% train, 15% val, 15% test (chronological)
  5. Normalize: StandardScaler fit on train only
  6. Train CNN-BiLSTM with early stopping
  7. Evaluate on test set
  8. Gate check: test accuracy >= 54% → enable filter
  9. Print classification report and feature stats
"""

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress TF noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    from config import Config
    from data.fetcher import get_candles
    from data.processor import process
    from data.macro_fetcher import fetch_and_cache_macro, get_macro_series
    from ml.deep_features import build_deep_features, build_sequences, N_FEATURES, DEEP_FEATURE_NAMES
    from ml.deep_model import build_model, train_model, save_model

    print("=" * 70)
    print("  Stage 7: CNN-BiLSTM Deep Learning Model — Training")
    print("=" * 70)

    # ── 1. Fetch data ─────────────────────────────────────────────────────
    print("\n[1/9] Fetching M15 data from Polygon...")
    m15_raw = get_candles(
        timeframe="M15",
        n_candles=47000,
    )
    if m15_raw is None or m15_raw.empty:
        print("FATAL: Could not fetch M15 data.")
        sys.exit(1)

    m15 = process(m15_raw, timeframe="M15", label="DEEP_M15")
    if m15 is None or m15.empty:
        print("FATAL: M15 data processing failed.")
        sys.exit(1)
    print(f"  M15 bars: {len(m15):,}")

    # ── 2. Fetch macro data ───────────────────────────────────────────────
    print("\n[2/9] Fetching macro data (DXY, VIX, US10Y)...")
    try:
        macro_result = fetch_and_cache_macro()
        for name, count in macro_result.items():
            print(f"  {name}: {count} daily bars")
        macro_df = get_macro_series()
        print(f"  Macro series: {len(macro_df)} days")
    except Exception as exc:
        print(f"  WARNING: Macro fetch failed ({exc}) — using zeros")
        macro_df = None

    # ── 3. Build features ─────────────────────────────────────────────────
    print("\n[3/9] Building deep features...")
    feat_df = build_deep_features(m15, macro_df=macro_df)
    print(f"  Feature columns ({len(feat_df.columns)}): {list(feat_df.columns)}")

    # ── 4. Build sequences ────────────────────────────────────────────────
    lookback = Config.DEEP_LOOKBACK
    future_bars = 3
    print(f"\n[4/9] Building {lookback}-bar windows with {future_bars}-bar labels...")
    X, y, indices = build_sequences(
        feat_df,
        lookback=lookback,
        future_bars=future_bars,
        close_series=m15["close"],
    )
    print(f"  Sequences: {X.shape[0]:,} samples × {X.shape[1]} bars × {X.shape[2]} features")
    print(f"  Label distribution: UP={y.sum():.0f} ({y.mean():.1%}) DOWN={len(y) - y.sum():.0f} ({1 - y.mean():.1%})")

    if len(X) < 1000:
        print("FATAL: Not enough valid sequences for training.")
        sys.exit(1)

    # ── 5. Chronological split ────────────────────────────────────────────
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"\n[5/9] Split: train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

    # ── 6. Normalize: fit scaler on train, apply to all ───────────────────
    print("\n[6/9] Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    # Reshape to 2D for scaler: (n_samples * lookback, n_features)
    train_flat = X_train.reshape(-1, X_train.shape[2])
    scaler.fit(train_flat)

    def scale_3d(X_3d):
        shape = X_3d.shape
        flat = X_3d.reshape(-1, shape[2])
        scaled = scaler.transform(flat)
        return scaled.reshape(shape).astype(np.float32)

    X_train_s = scale_3d(X_train)
    X_val_s = scale_3d(X_val)
    X_test_s = scale_3d(X_test)

    # Save scaler
    scaler_path = os.path.join(Config.BASE_DIR, Config.DEEP_SCALER_PATH)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved to {scaler_path}")

    # ── 7. Train ──────────────────────────────────────────────────────────
    print(f"\n[7/9] Training CNN-BiLSTM (max 100 epochs, early stopping patience=10)...")
    model, history = train_model(
        X_train_s, y_train,
        X_val_s, y_val,
        epochs=100,
        batch_size=32,
        verbose=1,
    )

    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_val_loss = min(history.history["val_loss"])
    best_val_acc = history.history["val_accuracy"][best_epoch - 1]
    print(f"\n  Best epoch: {best_epoch} | val_loss={best_val_loss:.4f} val_acc={best_val_acc:.4f}")

    # ── 8. Evaluate on test set ───────────────────────────────────────────
    print("\n[8/9] Evaluating on test set...")
    test_probs = model.predict(X_test_s, batch_size=256, verbose=0).flatten()
    test_preds = (test_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    try:
        auc = roc_auc_score(y_test, test_probs)
    except ValueError:
        auc = 0.5

    print(f"\n  Test Accuracy:  {acc:.4f} ({acc * 100:.1f}%)")
    print(f"  Test Precision: {prec:.4f}")
    print(f"  Test Recall:    {rec:.4f}")
    print(f"  Test AUC:       {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["DOWN", "UP"]))

    # ── 9. Gate check and save ────────────────────────────────────────────
    gate = Config.DEEP_ACCURACY_GATE
    gate_passed = acc >= gate

    print("=" * 70)
    if gate_passed:
        print(f"  GATE PASSED: {acc:.1%} >= {gate:.0%}")
        print(f"  USE_DEEP_FILTER will be set to True")
    else:
        print(f"  GATE FAILED: {acc:.1%} < {gate:.0%}")
        print(f"  USE_DEEP_FILTER remains False")
    print("=" * 70)

    # Save model regardless
    model_path = save_model(model)
    print(f"\n  Model saved to {model_path}")

    # Update config if gate passed
    if gate_passed:
        _update_config_flag(True)

    # Print training history summary
    print("\n  Training History (last 5 epochs):")
    print(f"  {'Epoch':>5} {'train_loss':>10} {'train_acc':>10} {'val_loss':>10} {'val_acc':>10}")
    n_epochs = len(history.history["loss"])
    for e in range(max(0, n_epochs - 5), n_epochs):
        print(f"  {e + 1:5d} {history.history['loss'][e]:10.4f} "
              f"{history.history['accuracy'][e]:10.4f} "
              f"{history.history['val_loss'][e]:10.4f} "
              f"{history.history['val_accuracy'][e]:10.4f}")

    return acc, gate_passed


def _update_config_flag(enable: bool):
    """Update USE_DEEP_FILTER in config.py."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.py")
    config_path = os.path.normpath(config_path)

    try:
        with open(config_path, "r") as f:
            content = f.read()

        old = "USE_DEEP_FILTER       = False"
        new = "USE_DEEP_FILTER       = True "
        if enable and old in content:
            content = content.replace(old, new)
            with open(config_path, "w") as f:
                f.write(content)
            print(f"  Updated config.py: USE_DEEP_FILTER = True")
        elif not enable:
            old = "USE_DEEP_FILTER       = True "
            new = "USE_DEEP_FILTER       = False"
            if old in content:
                content = content.replace(old, new)
                with open(config_path, "w") as f:
                    f.write(content)
    except Exception as exc:
        print(f"  WARNING: Could not update config.py: {exc}")
        print(f"  Manually set USE_DEEP_FILTER = {enable}")


if __name__ == "__main__":
    main()
