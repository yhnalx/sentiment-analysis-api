import re
import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

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


def load_and_prepare(path):
    df = pd.read_csv(path)

    df = df[["sentence", "label"]].copy()
    df["sentence"] = df["sentence"].astype(str).apply(clean_tweet)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df[df["label"].isin(LABEL_MAP.keys())].copy()
    df["labels"] = df["label"].map(LABEL_MAP)

    df = df.dropna(subset=["sentence", "labels"])
    df = df[df["sentence"].str.len() > 0]

    return df


def tokenize_function(examples):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=64
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    print("\nLoading datasets...")
    train_df = load_and_prepare("train.csv")
    val_df = load_and_prepare("val.csv")
    test_df = load_and_prepare("test.csv")

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain label distribution:")
    print(train_df["label"].value_counts())

    train_dataset = Dataset.from_pandas(
        train_df[["sentence", "labels"]],
        preserve_index=False
    )
    val_dataset = Dataset.from_pandas(
        val_df[["sentence", "labels"]],
        preserve_index=False
    )
    test_dataset = Dataset.from_pandas(
        test_df[["sentence", "labels"]],
        preserve_index=False
    )

    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Force safetensors so transformers does not fall back to torch.load on .bin files
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
        use_safetensors=True
    )

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.remove_columns(["sentence"])
    val_dataset = val_dataset.remove_columns(["sentence"])
    test_dataset = test_dataset.remove_columns(["sentence"])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    print("\nStarting training...")
    trainer.train()

    print("\nEvaluating on validation set...")
    val_results = trainer.predict(val_dataset)
    val_preds = np.argmax(val_results.predictions, axis=1)
    val_labels = val_results.label_ids

    print("Validation Accuracy:", accuracy_score(val_labels, val_preds))
    print("Validation Macro F1:", f1_score(val_labels, val_preds, average="macro"))

    print("\nEvaluating on test set...")
    test_results = trainer.predict(test_dataset)
    test_preds = np.argmax(test_results.predictions, axis=1)
    test_labels = test_results.label_ids

    print("Test Accuracy:", accuracy_score(test_labels, test_preds))
    print("Test Macro F1:", f1_score(test_labels, test_preds, average="macro"))

    print("\nClassification Report:")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=["negative", "neutral", "positive"]
    ))

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

    print("\nSaving model...")
    trainer.save_model("./sentiment_model")
    tokenizer.save_pretrained("./sentiment_model")

    print("\nDone. Model saved to ./sentiment_model")