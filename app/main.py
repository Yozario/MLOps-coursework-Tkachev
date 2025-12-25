from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.schemas import PredictRequest, PredictResponse
from src.inference.predictor import SpamPredictor, PredictorConfig

app = FastAPI(
    title="Spam Classifier API",
    version="1.0.0",
    description="MLOps coursework: spam/ham classification by email content",
)

# Load model on startup (once)
predictor: SpamPredictor | None = None


@app.on_event("startup")
def _startup():
    global predictor
    cfg = PredictorConfig(
        model_path="models/best",
        max_length=128,
        device="auto",
        threshold=0.5,
    )
    predictor = SpamPredictor(cfg)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if predictor is None:
        return JSONResponse(status_code=503, content={"error": "Model is not loaded yet"})

    label, score = predictor.predict(req.text)
    return PredictResponse(label=label, score=score)
