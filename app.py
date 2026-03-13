import os
import re
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Render persistent disk mount path
MODEL_DIR = "sentiment_model"

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

app = FastAPI(title="Twitter Sentiment Analysis API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None


def clean_tweet(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SentimentRequest(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global tokenizer, model

    if not os.path.exists(MODEL_DIR):
        raise RuntimeError(
            f"Model directory not found: {MODEL_DIR}. "
            "Make sure download_model.py runs before starting the API."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print("Using device:", device)
    print("Model path:", MODEL_DIR)


@app.get("/")
def root():
    return {
        "message": "Sentiment API is running",
        "device": str(device),
        "model_dir": MODEL_DIR
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None and tokenizer is not None
    }


@app.post("/predict")
def predict(request: SentimentRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    cleaned_text = clean_tweet(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze().cpu().numpy()
        pred_id = int(torch.argmax(outputs.logits, dim=1).item())

    return {
        "text": text,
        "cleaned_text": cleaned_text,
        "prediction": ID2LABEL[pred_id],
        "scores": {
            "negative": float(np.round(probs[0], 6)),
            "neutral": float(np.round(probs[1], 6)),
            "positive": float(np.round(probs[2], 6))
        }
    }