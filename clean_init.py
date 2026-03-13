import pandas as pd

df1 = pd.read_csv("LabeledText_cleaned.csv")
df2 = pd.read_csv("unicode_emojis_with_sentiment.csv")

df1 = df1[["Caption", "LABEL"]].copy()
df2 = df2[["caption", "sentiment"]].copy()

df1.columns = ["sentence", "label"]
df2.columns = ["sentence", "label"]

df1["sentence"] = df1["sentence"].astype(str).str.strip()
df1["label"] = df1["label"].astype(str).str.strip().str.lower()

df2["sentence"] = df2["sentence"].astype(str).str.strip()
df2["label"] = df2["label"].astype(str).str.strip().str.lower()

valid_labels = ["positive", "neutral", "negative"]
df1 = df1[df1["label"].isin(valid_labels)]
df2 = df2[df2["label"].isin(valid_labels)]

combined = pd.concat([df1, df2], ignore_index=True)

combined = combined.dropna(subset=["sentence", "label"])
combined = combined.drop_duplicates()

combined = combined[combined["sentence"].str.len() > 0]

print(combined.head())
print(combined["label"].value_counts())

combined.to_csv("merged_sentiment_dataset.csv", index=False)
print("Saved as merged_sentiment_dataset.csv")