import pandas as pd

# tweets = pd.read_csv("LabeledText_cleaned.csv")
# emoji = pd.read_csv("unicode_emojis_with_sentiment.csv")

# print(tweets.head())
# print(tweets.columns)

# print(emoji.head())
# print(emoji.columns)

merge = pd.read_csv("train.csv")

print(merge.tail(100))
print(merge.columns)
print(merge.shape)
print(merge["label"].value_counts(normalize=True))