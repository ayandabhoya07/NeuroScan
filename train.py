import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import keras
import matplotlib.pyplot as plt

from src.preprocessing import train_loader, val_loader
from src.cnn_model import build_model


def main():

    EPOCHS = 15

    # Build model
    model = build_model(
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS
    )

    model.summary()

    # Callbacks
    callbacks = [

        keras.callbacks.ModelCheckpoint(
            "best_cnn_v2_bn.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save final model
    model.save("neuroscan_cnn_v2_bn_final.keras")

    print("Model saved successfully!")

    # Save training history
    with open("training_history_v2.json", "w") as f:
        json.dump(history.history, f)

    # Print metrics
    print("\nTraining Results")

    print(
        "Best Training Accuracy:",
        max(history.history["accuracy"])
    )

    print(
        "Best Validation Accuracy:",
        max(history.history["val_accuracy"])
    )

    print(
        "Best Training Loss:",
        min(history.history["loss"])
    )

    print(
        "Best Validation Loss:",
        min(history.history["val_loss"])
    )

    # Accuracy graph
    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title("NeuroScan CNN v2 - Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid()

    plt.savefig("accuracy_v2_bn.png")
    plt.show()

    # Loss graph
    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title("NeuroScan CNN v2 - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid()

    plt.savefig("loss_v2_bn.png")
    plt.show()


if __name__ == "__main__":
    main()