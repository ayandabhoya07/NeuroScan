import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers
import math


base_model = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

print("Model name:", base_model.name)
print("Total layers:", len(base_model.layers))
print("Input shape:", base_model.input_shape)
print("Output shape:", base_model.output_shape)
print("Total parameters:", base_model.count_params())

# Freeze the pretrained EfficientNetB0 layers
base_model.trainable = False

trainable_params = sum(
    math.prod(v.shape) for v in base_model.trainable_weights
)

non_trainable_params = sum(
    math.prod(v.shape) for v in base_model.non_trainable_weights
)

print("Trainable parameters:", trainable_params)
print("Non-trainable parameters:", non_trainable_params)

inputs = keras.Input(shape=(224, 224, 3))

x = layers.Rescaling(255.0)(inputs)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(4, activation="softmax")(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="NeuroScan_EfficientNetB0"
)

#model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)