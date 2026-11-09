import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers


# Mixed precision
keras.mixed_precision.set_global_policy("mixed_float16")


# CNN convolution block
def conv_block(x, filters):

    for _ in range(2):

        x = layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False
        )(x)

        x = layers.BatchNormalization(momentum=0.9)(x)

        x = layers.ReLU()(x)

    return layers.MaxPooling2D(2)(x)


# Model builder
def build_model(steps_per_epoch, epochs=30):

    inputs = keras.Input(shape=(224, 224, 3))

    x = inputs

    for f in (32, 64, 128, 256):
        x = conv_block(x, f)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(128, activation="relu")(x)

    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(
        4,
        activation="softmax",
        dtype="float32"
    )(x)

    model = keras.Model(
        inputs,
        outputs,
        name="NeuroScan_CNN_v2_BN"
    )

    # Cosine learning rate decay
    lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4,
        decay_steps=epochs * steps_per_epoch
    )

    # Compile model
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=1e-4
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model