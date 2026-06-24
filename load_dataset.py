import pandas as pd
from datasets import load_dataset

DATASET_NAME = "allegro/klej-polemo2-in"
AMBIGUOUS_LABEL = "__label__meta_amb"
FILTERED_OUTPUT_PATH = "polemo2_test_filtered.csv"

LABEL_MAP = {
    "__label__meta_plus_m": "plus",
    "__label__meta_minus_m": "minus",
    "__label__meta_zero": "neutral",
}


def load_polemo_dataset():
    return load_dataset(DATASET_NAME)


def filter_test_split(df, exclude_ambiguous=True):
    if exclude_ambiguous:
        return df[df["target"] != AMBIGUOUS_LABEL].copy()
    return df.copy()


def print_split_statistics(dataset_dict):
    print("=" * 60)
    print("Liczność próbek w podzbiorach")
    print("=" * 60)
    for split_name, split_data in dataset_dict.items():
        print(f"  {split_name}: {len(split_data)}")

    test_df = dataset_dict["test"].to_pandas()
    ambiguous_count = (test_df["target"] == AMBIGUOUS_LABEL).sum()
    filtered_count = len(test_df) - ambiguous_count

    print("\nSplit test (używany do klasyfikacji):")
    print(f"  Przed filtrem:           {len(test_df)}")
    print(f"  Usunięte (ambiguous):    {ambiguous_count}")
    print(f"  Po filtrze:              {filtered_count}")


def print_class_balance(df):
    print("\n" + "=" * 60)
    print("Zrównoważenie klas (po filtrze)")
    print("=" * 60)

    counts = df["target"].value_counts()
    percentages = df["target"].value_counts(normalize=True) * 100

    print("\nLiczebność:")
    for label, count in counts.items():
        readable = LABEL_MAP.get(label, label)
        print(f"  {label} ({readable}): {count}")

    print("\nProcenty:")
    for label, pct in percentages.items():
        readable = LABEL_MAP.get(label, label)
        print(f"  {label} ({readable}): {pct:.1f}%")


def print_text_length_stats(df):
    print("\n" + "=" * 60)
    print("Długość tekstów (kolumna sentence)")
    print("=" * 60)

    char_lengths = df["sentence"].str.len()
    word_lengths = df["sentence"].str.split().str.len()

    print("\nDługość w znakach:")
    print(char_lengths.describe().to_string())

    print("\nDługość w słowach:")
    print(word_lengths.describe().to_string())

    empty_count = (df["sentence"].str.strip() == "").sum()
    if empty_count > 0:
        print(f"\nUwaga: znaleziono {empty_count} pustych tekstów.")


def print_sample_examples(df):
    print("\n" + "=" * 60)
    print("Przykłady z każdej klasy")
    print("=" * 60)

    for label in df["target"].unique():
        readable = LABEL_MAP.get(label, label)
        row = df[df["target"] == label].iloc[0]
        sentence_preview = row["sentence"][:120]
        if len(row["sentence"]) > 120:
            sentence_preview += "..."
        print(f"\n[{readable}] {label}")
        print(f"  {sentence_preview}")


def save_filtered_dataset(df, path=FILTERED_OUTPUT_PATH):
    df.to_csv(path, index=False)
    print(f"\nZapisano przefiltrowany zbiór do: {path}")


def main():
    dataset_dict = load_polemo_dataset()
    print_split_statistics(dataset_dict)

    test_df = dataset_dict["test"].to_pandas()
    df_filtered = filter_test_split(test_df)

    assert AMBIGUOUS_LABEL not in df_filtered["target"].values
    assert df_filtered["target"].value_counts().sum() == len(df_filtered)

    print_class_balance(df_filtered)
    print_text_length_stats(df_filtered)
    print_sample_examples(df_filtered)
    save_filtered_dataset(df_filtered)

    return df_filtered


if __name__ == "__main__":
    main()
