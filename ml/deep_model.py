"""
GoldSignalAI — ml/deep_model.py
=================================
CNN-BiLSTM model definition, training, saving, and loading (Stage 7).

Architecture:
  Conv1D(64) → BN → Conv1D(128) → BN → Dropout(0.2) →
  BiLSTM(64, return_seq) → BiLSTM(32, return_seq) →
  MultiHeadAttention(1 head) → GlobalAvgPool1D →
  Dense(64) → Dropout(0.3) → Dense(1, sigmoid)

Uses TensorFlow/Keras. Model saved in .keras format.
"""

import logging
import os
from typing import Optional

import numpy as np

from config import Config

logger = logging.getLogger(__name__)


def build_model(n_timesteps: int, n_features: int):
    """
    Build the CNN-BiLSTM-Attention model.

    Args:
        n_timesteps: Lookback window size (60).
        n_features: Number of features per bar (15).

    Returns:
        Compiled Keras model.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(n_timesteps, n_features), name="input_window")

    # Conv block 1
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    # Conv block 2
    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # BiLSTM layers
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

    # Attention (single head)
    attn_out = layers.MultiHeadAttention(num_heads=1, key_dim=32)(x, x)
    x = layers.Add()([x, attn_out])  # residual connection

    # Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Dense head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation="sigmoid", name="direction")(x)

    model = keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 1,
):
    """
    Train the CNN-BiLSTM model with early stopping.

    Returns:
        (model, history) tuple.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    n_timesteps, n_features = X_train.shape[1], X_train.shape[2]
    model = build_model(n_timesteps, n_features)

    if verbose:
        model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose,
    )

    return model, history


def save_model(model, path: Optional[str] = None) -> str:
    """Save model to .keras format."""
    if path is None:
        path = os.path.join(Config.BASE_DIR, Config.DEEP_MODEL_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    logger.info("Deep model saved to %s", path)
    return path


def load_model(path: Optional[str] = None):
    """Load model from .keras format. Returns None on failure."""
    import tensorflow as tf

    if path is None:
        path = os.path.join(Config.BASE_DIR, Config.DEEP_MODEL_PATH)

    if not os.path.isfile(path):
        logger.warning("Deep model not found at %s", path)
        return None

    try:
        model = tf.keras.models.load_model(path)
        logger.info("Deep model loaded from %s", path)
        return model
    except Exception as exc:
        logger.error("Failed to load deep model: %s", exc)
        return None
