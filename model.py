import typing
import tensorflow as tf
from keras import layers, Model

def residual_block(
        x: tf.Tensor,
        filter_num: int,
        strides: typing.Union[int, list] = 2,
        kernel_size: typing.Union[int, list] = 3,
        skip_conv: bool = True,
        padding: str = "same",
        kernel_initializer: str = "he_uniform",
        activation: str = "relu",
        dropout: float = 0.2):
    x_skip = x
    x = layers.Conv2D(filter_num, kernel_size, padding=padding, strides=strides, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation=activation)
    x = layers.Conv2D(filter_num, kernel_size, padding=padding, kernel_initializer=kernel_initializer)(x)
    x = layers.BatchNormalization()(x)
    if skip_conv:
        x_skip = layers.Conv2D(filter_num, 1, padding=padding, strides=strides, kernel_initializer=kernel_initializer)(x_skip)
    x = layers.Add()([x, x_skip])
    x = activation_layer(x, activation=activation)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x

def activation_layer(layer, activation="relu", alpha=0.1):
    if activation == "relu":
        return layers.ReLU()(layer)
    elif activation == "leaky_relu":
        return layers.LeakyReLU(negative_slope=alpha)(layer)

def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2):
    inputs = layers.Input(shape=input_dim, name="input")
    x = layers.Lambda(lambda x: x / 255)(inputs)

    x = residual_block(x, 32, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x = residual_block(x, 32, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x = residual_block(x, 32, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x = residual_block(x, 64, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x = residual_block(x, 64, activation=activation, skip_conv=False, strides=1, dropout=dropout)
    x = residual_block(x, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x = residual_block(x, 128, activation=activation, skip_conv=True, strides=1, dropout=dropout)
    x = residual_block(x, 128, activation=activation, skip_conv=True, strides=2, dropout=dropout)
    x = residual_block(x, 128, activation=activation, skip_conv=False, strides=1, dropout=dropout)

    x = layers.Reshape((x.shape[-3] * x.shape[2], x.shape[-1]))(x)
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(output_dim + 1, activation="softmax", name="output")(x)
    return Model(inputs=inputs, outputs=outputs)

