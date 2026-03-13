import pandas as pd
from sklearn.model_selection import train_test_split

combined = pd.read_csv("merged_sentiment_dataset.csv")

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

combined["label_id"] = combined["label"].map(label_map)

train_df, test_df = train_test_split(
    combined,
    test_size=0.2,
    stratify=combined["label_id"],
    random_state=42
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label_id"],
    random_state=42
)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Splits saved!")