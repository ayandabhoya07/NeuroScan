import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

from src.preprocessing import test_loader


def main():

    # 1. Load the trained model
    model = keras.models.load_model(
        "best_cnn_v2_bn.keras"
    )

    print("Model loaded successfully!")

    # 2. Evaluate on test dataset
    test_loss, test_accuracy = model.evaluate(
        test_loader,
        verbose=1
    )

    print("\nTest Results")
    print("-" * 40)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # 3. Generate predictions
    predictions = model.predict(
        test_loader,
        verbose=1
    )

    y_pred = np.argmax(predictions, axis=1)

    # 4. Collect actual labels
    y_true = np.concatenate([
        labels.numpy()
        for _, labels in test_loader
    ])

    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    # 5. Classification report
    class_names = [
        "glioma",
        "meningioma",
        "notumor",
        "pituitary"
    ]

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    
    print("\nClassification Report")
    print("-" * 60)
    print(report)

    # Save classification report
    with open("classification_report_v2.txt", "w") as f:
        f.write(report)

    # 6. Confusion Matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display.plot(
        cmap="Blues",
        ax=ax,
        values_format="d"
    )

    plt.title("NeuroScan CNN v2 - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "confusion_matrix_v2.png",
        dpi=300
    )

    plt.show()

    print("\nEvaluation completed!")
    print("Saved:")
    print("- classification_report_v2.txt")
    print("- confusion_matrix_v2.png")


if __name__ == "__main__":
    main()