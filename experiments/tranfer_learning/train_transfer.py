import os 
os.environ["KERAS_BACKEND"] = "torch"
from keras.callbacks import ReduceLROnPlateau , EarlyStopping
from src.preprocessing import train_loader , val_loader
from .transfer_model import model

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.000001,
    verbose=1
)

history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=30,
    callbacks=[early_stopping, reduce_lr]
)

model.save("neuroscan_efficientnetb0.keras")
print("EfficientNetB0 model saved successfully!")


# to run this use python -m experiments.transfer_learning.train_transfer in the root directory as this file was moved from src to here 
