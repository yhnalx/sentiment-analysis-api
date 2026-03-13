import re
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./sentiment_model"

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

app = FastAPI(title="Sentiment Analysis API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


def clean_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SentimentRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Sentiment API is running"}


@app.post("/predict")
def predict(request: SentimentRequest):
    cleaned_text = clean_tweet(request.text)

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
        "text": request.text,
        "cleaned_text": cleaned_text,
        "prediction": ID2LABEL[pred_id],
        "scores": {
            "negative": float(probs[0]),
            "neutral": float(probs[1]),
            "positive": float(probs[2])
        }
    }