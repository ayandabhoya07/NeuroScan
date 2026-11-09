# NeuroScan

### Brain MRI Image Classification with a Custom CNN, FastAPI & JavaScript

```{=html}
<p align="center">
```
`<strong>`{=html}An end-to-end deep learning project for classifying
brain MRI images into four dataset categories.`</strong>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<img src="https://img.shields.io/badge/Python-3.11.9-blue?logo=python&logoColor=white" alt="Python 3.11.9">`{=html}
`<img src="https://img.shields.io/badge/Keras-3.x-red?logo=keras&logoColor=white" alt="Keras 3">`{=html}
`<img src="https://img.shields.io/badge/PyTorch-Backend-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">`{=html}
`<img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white" alt="FastAPI">`{=html}
`<img src="https://img.shields.io/badge/Status-Local%20application%20tested-success" alt="Status">`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## Overview

**NeuroScan** is an educational deep learning application that uses a
custom Convolutional Neural Network (CNN) to classify brain MRI images
into one of four categories:

-   **Glioma**
-   **Meningioma**
-   **No Tumor** (`notumor`)
-   **Pituitary**

The project covers the full machine-learning workflow: dataset
exploration, image preprocessing, CNN development, model training,
evaluation, inference, API development, and a browser-based interface.

The selected CNN V2 model achieved **93.56% accuracy** on the project's
held-out test set of 2,080 images.

> **Important medical disclaimer:** NeuroScan is an educational and
> experimental project. It is not a medical device, has not been
> clinically validated, and must not be used to diagnose, rule out, or
> make treatment decisions about any medical condition. MRI scans must
> be interpreted by qualified healthcare professionals.

------------------------------------------------------------------------

## Key Features

-   Four-class brain MRI image classification.
-   Custom CNN architecture developed and trained for this project.
-   Experiments with Batch Normalization, regularization, learning-rate
    scheduling, and transfer learning.
-   Image upload and preview through a browser interface.
-   FastAPI endpoint for model inference.
-   Displays the predicted dataset class and scores for all four
    classes.
-   Visual confidence indicator with high, moderate, and low model-score
    labels.
-   Evaluation reports, accuracy/loss plots, and confusion matrices.
-   Separate folders for the final model and experimental checkpoints.

------------------------------------------------------------------------

## Model Performance

The final selected model was evaluated on **2,080 test images**.

### Overall result

  Metric                     Score
  ------------------- ------------
  Test accuracy         **93.56%**
  Macro precision           93.31%
  Macro recall              93.15%
  Macro F1-score            93.20%
  Weighted F1-score         93.51%

### Per-class results

  Class          Precision   Recall   F1-score   Test images
  ------------ ----------- -------- ---------- -------------
  Glioma            93.98%   93.33%     93.65%           585
  Meningioma        90.12%   85.19%     87.59%           439
  No Tumor          94.92%   97.54%     96.21%           651
  Pituitary         94.22%   96.54%     95.37%           405

### Evaluation notes

Meningioma has the lowest recall at **85.19%**. This means that some
meningioma examples in the test set were classified as another category.
This limitation is important and should be considered alongside the
overall accuracy.

The reported metrics describe performance on this dataset split only.
They do not establish performance on scans from other hospitals,
scanners, populations, or clinical settings.

Evaluation artifacts are available in the [`reports/`](reports/)
directory.

------------------------------------------------------------------------

## Model Development

NeuroScan includes multiple experiments rather than relying on a single
model attempt.

  -----------------------------------------------------------------------
  Experiment              Description             Role
  ----------------------- ----------------------- -----------------------
  CNN V1                  Initial custom CNN with Baseline
                          three convolutional     
                          blocks                  

  CNN V2                  Deeper custom CNN with  Improved custom model
                          four convolutional      
                          blocks and Batch        
                          Normalization           

  EfficientNetB0          Transfer-learning       Alternative approach
                          experiment using        explored
                          ImageNet weights        
  -----------------------------------------------------------------------

### Selected architecture: CNN V2

The selected custom model uses:

-   Four convolutional blocks with filter sizes of 32, 64, 128, and 256.
-   Two convolutional layers in each block.
-   Batch Normalization and ReLU activations.
-   Max Pooling for spatial downsampling.
-   Global Average Pooling.
-   A 128-unit dense layer.
-   Dropout regularization.
-   A four-unit softmax output layer.

The training experiments also explored AdamW, weight decay, cosine-decay
learning-rate scheduling, and mixed precision.

The final application loads:

``` text
models/neuroscan_cnn_v2_bn_final.keras
```

Earlier model files are retained under `experiments/` to document the
development process. They are not the model used by the application.

------------------------------------------------------------------------

## Dataset

The project used the following Kaggle dataset:

**De-duplicated Brain MRI Dataset**\
Dataset identifier: `faysal202114023/de-duplicated-brain-mri-dataset`

The recorded dataset split contains:

  Split          Number of images
  ------------ ------------------
  Training                 15,595
  Validation                3,119
  Testing                   2,080
  **Total**            **20,794**

### Class distribution recorded during exploration

  Class          Images
  ------------ --------
  Glioma          4,388
  Meningioma      3,292
  No Tumor        4,880
  Pituitary       3,035

### Preprocessing

Images are converted to RGB, resized to **224 × 224**, converted to
tensors, and arranged in channels-last format for the model.

Training-time augmentation includes small rotations and affine
transformations. Validation, testing, and inference use deterministic
preprocessing without random augmentation.

The dataset is not included in this repository. Users must obtain it
separately from its original source and follow the dataset's terms.

------------------------------------------------------------------------

## Application Architecture

The application separates the browser interface, API, and
model-inference logic.

``` text
                 User
                  |
                  v
        Browser-based Frontend
          HTML / CSS / JavaScript
                  |
          POST /predict
                  |
                  v
             FastAPI
          backend/main.py
                  |
                  v
          Inference Pipeline
           src/inference.py
                  |
        Image preprocessing
       RGB -> 224x224 -> Tensor
                  |
                  v
          Trained CNN Model
 models/neuroscan_cnn_v2_bn_final.keras
                  |
                  v
        Four class scores returned
                  |
                  v
        FastAPI JSON response
                  |
                  v
       Frontend displays results
```

### Prediction flow

1.  The user opens the NeuroScan interface.
2.  The user selects or drops an MRI image.
3.  JavaScript validates the selected file and sends it to the API.
4.  FastAPI receives the uploaded image.
5.  The inference pipeline applies the expected preprocessing.
6.  The cached Keras model generates four class scores.
7.  The API returns the predicted class and all four scores.
8.  The frontend displays the result and confidence visualization.

------------------------------------------------------------------------

## Project Structure

``` text
NeuroScan/
│
├── backend/
│   ├── __init__.py
│   └── main.py
│
├── experiments/
│   ├── cnn_v1/
│   │   └── neuroscan_cnn.keras
│   ├── cnn_v2/
│   │   └── best_cnn_v2_bn.keras
│   ├── training_history_v2.json
│   └── tranfer_learning/
│       ├── train_transfer.py
│       └── transfer_model.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── models/
│   └── neuroscan_cnn_v2_bn_final.keras
│
├── notebook/
│   └── data_exploration.py
│
├── reports/
│   ├── accuracy_v2_bn.png
│   ├── classification_report_v2.txt
│   ├── confusion_matrix.png
│   ├── confusion_matrix_v2.png
│   ├── loss_v2_bn.png
│   └── prediction_samples.png
│
├── src/
│   ├── cnn_model.py
│   ├── inference.py
│   └── preprocessing.py
│
├── .gitignore
├── .python-version
├── requirements.txt
├── train.py
└── evaluate.py
```

> Note: The transfer-learning folder is currently named
> `tranfer_learning` in the project tree. Consider renaming it to
> `transfer_learning` for correct spelling.

------------------------------------------------------------------------

## Tech Stack

  Area                   Technology
  ---------------------- -----------------------
  Programming language   Python 3.11
  Deep learning API      Keras 3
  Keras backend          PyTorch
  Model architecture     Custom CNN
  Image processing       Pillow, torchvision
  Numerical operations   NumPy
  Backend/API            FastAPI
  ASGI server            Uvicorn
  Frontend               HTML, CSS, JavaScript
  Version control        Git and GitHub

------------------------------------------------------------------------

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/ayandabhoya07/NeuroScan.git
cd NeuroScan
```

Replace `YOUR` with the GitHub account that hosts the
repository.

### 2. Create and activate a virtual environment

On Windows:

``` bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

``` bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project was developed with Keras 3 using the PyTorch backend. The
dependency versions and installation method should be checked for
compatibility with your operating system and hardware. The original
development environment used an NVIDIA GPU, while a deployment host may
use CPU-only inference.

### 4. Start the application

Run the command from the project root:

``` bash
python -m uvicorn backend.main:app --reload
```

### 5. Open NeuroScan

Visit:

``` text
http://127.0.0.1:8000
```

FastAPI also exposes interactive API documentation at:

``` text
http://127.0.0.1:8000/docs
```

The application loads the selected model from:

``` text
models/neuroscan_cnn_v2_bn_final.keras
```

Make sure this file is present before starting the application.

------------------------------------------------------------------------

## API Endpoints

  -----------------------------------------------------------------------
  Method                  Endpoint                Description
  ----------------------- ----------------------- -----------------------
  `GET`                   `/`                     Serves the NeuroScan
                                                  frontend

  `GET`                   `/health`               Checks
                                                  application/model
                                                  health

  `POST`                  `/predict`              Accepts an image and
                                                  returns classification
                                                  results
  -----------------------------------------------------------------------

### Example prediction response

The `/predict` endpoint returns JSON in the following structure:

``` json
{
  "predicted_class": "glioma",
  "confidence": 0.94,
  "probabilities": {
    "glioma": 0.94,
    "meningioma": 0.03,
    "notumor": 0.02,
    "pituitary": 0.01
  }
}
```

This is an illustrative response showing the response format, not a real
prediction or a guaranteed result.

------------------------------------------------------------------------

## Important Notes and Limitations

-   NeuroScan is an educational project, not a clinically validated
    diagnostic tool.
-   Model performance is measured on the selected dataset and its
    held-out test split.
-   Performance may change when images come from different sources or
    imaging systems.
-   Meningioma recall is lower than the other class recalls in the
    reported evaluation.
-   Softmax scores are not necessarily calibrated probabilities and must
    not be interpreted as medical certainty.
-   The frontend's high, moderate, and low confidence labels are display
    thresholds, not clinically validated categories.
-   Do not use the application to make medical decisions. Consult
    qualified healthcare professionals for medical interpretation.

------------------------------------------------------------------------

## Future Improvements

Potential next steps include:

-   Add automated tests for preprocessing, inference, and API endpoints.
-   Add model calibration and more comprehensive uncertainty evaluation.
-   Evaluate on an independent external dataset.
-   Improve analysis of class-specific errors, particularly meningioma.
-   Add structured experiment tracking and reproducible training
    configurations.
-   Review CPU inference speed, memory use, and dependency compatibility
    before cloud deployment.
-   Add deployment monitoring, request logging, and input validation.
-   Improve documentation with application screenshots and a
    demonstration video.

------------------------------------------------------------------------

## Acknowledgements

-   The creators and maintainers of the brain MRI dataset used for this
    educational project.
-   The open-source communities behind Keras, PyTorch, FastAPI, NumPy,
    Pillow, and the frontend technologies.

------------------------------------------------------------------------

## Author

**Your Name**\
GitHub: [YOUR_GITHUB_PROFILE](https://github.com/ayandabhoya07)

------------------------------------------------------------------------

```{=html}
<p align="center">
```
`<strong>`{=html}NeuroScan`</strong>`{=html}`<br>`{=html} An educational
end-to-end deep learning project.
```{=html}
</p>
```
