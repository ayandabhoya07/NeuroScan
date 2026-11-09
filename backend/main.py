from fastapi import FastAPI, HTTPException, UploadFile, File
from src.inference import load_trained_model, predict_image
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="NeuroScan API",
    description="Brain MRI classification using a CNN model.",
    version="1.0.0"
)
BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )

@app.get("/health")
def health_check():

    try:
        model = load_trained_model()

        return {
            "status": "healthy",
            "model": "loaded",
            "message": "NeuroScan CNN is ready for predictions."
        }

    except Exception as e:

        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {str(e)}"
        )

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        image = Image.open(BytesIO(contents))

        result = predict_image(image)

        return result

    except (UnidentifiedImageError, OSError):

        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid MRI image."
        )