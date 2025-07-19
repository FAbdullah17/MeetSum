# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk

def load_datasets():
    path = "C:/Users/Fahad's WorkStation/Programs/PF/MeetSum/data"
    train = load_from_disk(f"{path}/train")
    val = load_from_disk(f"{path}/validation")
    test = load_from_disk(f"{path}/test")
    return train.to_pandas(), val, test

def basic_eda(df):
    print("Columns:", df.columns.tolist())
    print("\nSample row:\n", df.iloc[:2])

    df["transcript_length"] = df["transcript"].apply(lambda x: len(x.split()))
    df["summary_length"] = df["summary"].apply(lambda x: len(x.split()))

    print("\nTranscript vs Summary Length Stats:")
    print(df[["transcript_length", "summary_length"]].describe())

    print("\nMissing Transcripts:", df["transcript"].isnull().sum())
    print("Missing Summaries:", df["summary"].isnull().sum())

    df["transcript_words"] = df["transcript"].apply(lambda x: len(x.split()))
    df["summary_words"] = df["summary"].apply(lambda x: len(x.split()))
    df["length_ratio"] = df["transcript_words"] / df["summary_words"]

    print("\n📊 Summary Statistics:")
    print(df[["transcript_words", "summary_words", "length_ratio"]].describe())
    return df

def plot_distributions(df):
    plt.figure(figsize=(10, 5))
    sns.histplot(df["transcript_words"], bins=50, kde=True, color="steelblue")
    plt.title("Transcript Length Distribution (words)")
    plt.xlabel("Transcript Length (words)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(df["summary_words"], bins=50, kde=True, color="orange")
    plt.title("Summary Length Distribution (words)")
    plt.xlabel("Summary Length (words)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    sns.histplot(df["length_ratio"], bins=50, kde=True, color="green")
    plt.title("Transcript-to-Summary Length Ratio")
    plt.xlabel("Ratio")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df["length_ratio"])
    plt.title("Length Ratio Boxplot")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.heatmap(df[["transcript_words", "summary_words", "length_ratio"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation")
    plt.show()

def show_sample_pairs(df, n=2):
    print("\n📝 Sample Summary-Transcript Pair:")
    for i in range(n):
        print(f"\n--- Example {i+1} ---")
        print("Transcript (short preview):", df.iloc[i]["transcript"][:300], "...")
        print("Summary:", df.iloc[i]["summary"])

# Uncomment to test as script
# if __name__ == "__main__":
#     df, _, _ = load_datasets()
#     df = basic_eda(df)
#     plot_distributions(df)
#     show_sample_pairs(df)
