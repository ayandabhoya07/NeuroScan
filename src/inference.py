import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
from PIL import Image
import keras
from torchvision import transforms
from functools import lru_cache
from pathlib import Path

def convert_rgb(image):
    return image.convert("RGB")


def to_channels_last(image):
    return image.permute(1, 2, 0)


inference_transform = transforms.Compose([
    transforms.Lambda(convert_rgb),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(to_channels_last)
])


def preprocess_image(image):

    image = inference_transform(image)

    image = np.expand_dims(image.numpy(), axis=0)

    return image

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# Load the model only once
@lru_cache(maxsize=1)
def load_trained_model():

    model_path = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "neuroscan_cnn_v2_bn_final.keras"
)

    model = keras.models.load_model(
        model_path,
        compile=False
    )

    return model


# Generate prediction
def predict_image(image):

    # Preprocess uploaded image
    processed_image = preprocess_image(image)

    # Load trained model
    model = load_trained_model()

    # Get probabilities
    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    # Find class with highest probability
    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(predictions[predicted_index])

    # Create probability dictionary
    probabilities = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities
    }