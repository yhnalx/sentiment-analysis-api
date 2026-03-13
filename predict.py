import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./sentiment_model"

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def clean_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model loaded.")
print("Using device:", device)

while True:
    text = input("\nEnter a tweet (or type 'exit'): ")

    if text.lower() == "exit":
        break

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
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    print("\nInput:", text)
    print("Cleaned:", cleaned_text)
    print("Prediction:", ID2LABEL[predicted_class])
    print("Confidence scores:")
    print(f"  negative: {probabilities[0]:.4f}")
    print(f"  neutral : {probabilities[1]:.4f}")
    print(f"  positive: {probabilities[2]:.4f}")