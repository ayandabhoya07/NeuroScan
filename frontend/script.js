// =========================================
// 1. GET HTML ELEMENTS
// =========================================

const fileInput = document.getElementById("fileInput");
const browseButton = document.getElementById("browseButton");

const uploadArea = document.getElementById("uploadArea");
const uploadContent = document.getElementById("uploadContent");

const imagePreview = document.getElementById("imagePreview");
const previewImage = document.getElementById("previewImage");

const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");

const removeImage = document.getElementById("removeImage");
const analyzeButton = document.getElementById("analyzeButton");

let selectedFile = null;
let previewURL = null;


// =========================================
// 2. OPEN FILE SELECTOR
// =========================================

browseButton.addEventListener("click", () => {
    fileInput.click();
});


// =========================================
// 3. HANDLE FILE SELECTION
// =========================================

fileInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (file) {
        handleFile(file);
    }

});


// =========================================
// 4. PROCESS SELECTED IMAGE
// =========================================

function handleFile(file) {

    const allowedTypes = [
        "image/jpeg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {
        showError("Please upload a JPG or PNG image.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        showError("Image size must be less than 10 MB.");
        return;
    }

    selectedFile = file;

    clearError();
    resetResults();

    // Create image preview
    if (previewURL) {
        URL.revokeObjectURL(previewURL);
    }

    previewURL = URL.createObjectURL(file);

    previewImage.src = previewURL;

    fileName.textContent = file.name;

    fileSize.textContent =
        (file.size / (1024 * 1024)).toFixed(2) + " MB";

    // Show preview
    uploadContent.style.display = "none";
    imagePreview.style.display = "flex";

    // Enable analyze button
    analyzeButton.disabled = false;
}


// =========================================
// 5. REMOVE SELECTED IMAGE
// =========================================

removeImage.addEventListener("click", () => {

    selectedFile = null;

    fileInput.value = "";

    if (previewURL) {
        URL.revokeObjectURL(previewURL);
        previewURL = null;
    }

    previewImage.src = "";

    uploadContent.style.display = "block";
    imagePreview.style.display = "none";

    analyzeButton.disabled = true;

    resetResults();
    clearError();

});

// =========================================
// 6. DRAG AND DROP
// =========================================

uploadArea.addEventListener("dragover", (event) => {

    event.preventDefault();

    uploadArea.classList.add("drag-over");

});


uploadArea.addEventListener("dragleave", () => {

    uploadArea.classList.remove("drag-over");

});


uploadArea.addEventListener("drop", (event) => {

    event.preventDefault();

    uploadArea.classList.remove("drag-over");

    const file = event.dataTransfer.files[0];

    if (file) {
        handleFile(file);
    }

});

// =========================================
// 7. SEND IMAGE TO FASTAPI
// =========================================

analyzeButton.addEventListener("click", async () => {

    if (!selectedFile) {
        return;
    }

    clearError();

    showLoading();

    const formData = new FormData();

    formData.append("file", selectedFile);

    try {

        const response = await fetch(
                "/predict",
            {
                method: "POST",
                body: formData
            }
        );

        const result = await response.json();

        if (!response.ok) {
            throw new Error(
                result.detail || "Prediction failed."
            );
        }

        displayResults(result);

    }

    catch (error) {

        hideLoading();

        showError(error.message);

        console.error("Prediction error:", error);

    }

});

// =========================================
// 8. DISPLAY PREDICTION RESULTS
// =========================================

function displayResults(result) {

    hideLoading();

    document.getElementById("emptyResults").style.display = "none";

    const predictionResults = document.getElementById(
        "predictionResults"
    );

    predictionResults.style.display = "block";

    // Class name formatting
    const classLabels = {
        glioma: "Glioma",
        meningioma: "Meningioma",
        notumor: "No Tumor",
        pituitary: "Pituitary"
    };

    const predictedClass = document.getElementById(
        "predictedClass"
    );

    predictedClass.textContent =
        classLabels[result.predicted_class] || result.predicted_class;

    // Confidence
    const confidence = result.confidence * 100;

    document.getElementById("confidenceValue").textContent =
        confidence.toFixed(2) + "%";

    document.getElementById("confidenceBar").style.width =
        confidence + "%";

// Confidence Assessment
const confidenceStatus = document.getElementById("confidenceStatus");

if (confidence >= 90) {

    confidenceStatus.textContent = "High model confidence";
    confidenceStatus.className = "confidence-status high";

}
else if (confidence >= 70) {

    confidenceStatus.textContent = "Moderate model confidence";
    confidenceStatus.className = "confidence-status moderate";

}
else {

    confidenceStatus.textContent =
        "Low model confidence — prediction is uncertain.";

    confidenceStatus.className = "confidence-status low";

}
    // Probability bars
    const probabilityList = document.getElementById(
        "probabilityList"
    );

    probabilityList.innerHTML = "";

    for (const [className, probability] of Object.entries(
        result.probabilities
    )) {

        const percentage = probability * 100;

        const item = document.createElement("div");

        item.className = "probability-item";

        const label = document.createElement("span");
        label.textContent = classLabels[className] || className;

        const value = document.createElement("span");
        value.textContent = percentage.toFixed(2) + "%";

        const info = document.createElement("div");
        info.className = "probability-info";

        info.append(label, value);

        const track = document.createElement("div");
        track.className = "probability-track";

        const fill = document.createElement("div");
        fill.className = "probability-fill";

        track.appendChild(fill);

        item.append(info, track);

        probabilityList.appendChild(item);

        requestAnimationFrame(() => {
            fill.style.width = percentage + "%";
        });

    }

}

// =========================================
// 9. LOADING STATE
// =========================================

function showLoading() {

    document.getElementById("emptyResults").style.display = "none";

    document.getElementById("predictionResults").style.display = "none";

    document.getElementById("loadingState").style.display = "flex";

    analyzeButton.disabled = true;

}


function hideLoading() {

    document.getElementById("loadingState").style.display = "none";

    analyzeButton.disabled = !selectedFile;

}


// =========================================
// 10. ERROR HANDLING
// =========================================

function showError(message) {

    const errorMessage = document.getElementById("errorMessage");

    errorMessage.textContent = message;

    errorMessage.style.display = "block";

}


function clearError() {

    const errorMessage = document.getElementById("errorMessage");

    errorMessage.textContent = "";

    errorMessage.style.display = "none";

}


// =========================================
// 11. RESET RESULTS
// =========================================

function resetResults() {

    document.getElementById("predictionResults").style.display = "none";

    document.getElementById("loadingState").style.display = "none";

    document.getElementById("emptyResults").style.display = "flex";

    document.getElementById("confidenceBar").style.width = "0%";

}