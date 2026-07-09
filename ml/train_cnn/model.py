"""Small tf.keras Conv1D model for ring gesture windows."""

from __future__ import annotations


def build_model(window_samples: int = 128, channels: int = 6, classes: int = 5):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(window_samples, channels), name="imu_raw")
    x = tf.keras.layers.Conv1D(16, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(32, 5, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ring_microgesture_cnn")
    assert model.count_params() < 25_000, f"CNN too large: {model.count_params()} params"
    return model
