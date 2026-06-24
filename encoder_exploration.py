import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from encoder_utils import (
    evaluate_predictions,
    load_filtered_test_data,
    load_model_and_tokenizer,
    map_hf_labels_to_dataset,
    map_model_labels_to_dataset,
    predict_with_max_length,
    print_confusion_matrix,
    uses_sigmoid,
)

MODELS = [
    "Voicelab/herbert-base-cased-sentiment",
    "bardsai/twitter-sentiment-pl-base",
    "nie3e/sentiment-polish-gpt2-small",
]

MAX_LENGTHS = [128, 256, 512]

RESULTS_JSON_PATH = "encoder_exploration_results.json"
RESULTS_CSV_PATH = "encoder_exploration_results.csv"
COMPARISON_PLOT_PATH = "encoder_exploration_comparison.png"



def run_experiment(model_name, max_length, texts, y_true, batch_size=16):
    print(f"\n--- Eksperyment: {model_name}, max_length={max_length} ---")
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    model_labels, scores = predict_with_max_length(
        model,
        tokenizer,
        texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        use_sigmoid=uses_sigmoid(model_name),
    )
    y_pred = map_model_labels_to_dataset(model_labels)

    results = evaluate_predictions(y_true, y_pred)
    results.update(
        {
            "model": model_name,
            "max_length": max_length,
            "num_samples": len(texts),
            "use_sigmoid": uses_sigmoid(model_name),
            "ambiguous_predictions": sum(1 for label in y_pred if label == "ambiguous"),
        }
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"  Accuracy={results['accuracy']:.4f}, "
        f"F1 macro={results['f1_macro']:.4f}, "
        f"F1 weighted={results['f1_weighted']:.4f}, "
        f"ambiguous={results['ambiguous_predictions']}"
    )
    return results, model_labels, scores, y_pred


def compare_results(all_results, baseline_results=None):
    rows = []
    for result in all_results:
        rows.append(
            {
                "model": result["model"],
                "max_length": result["max_length"],
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "f1_weighted": result["f1_weighted"],
                "ambiguous_predictions": result.get("ambiguous_predictions", 0),
            }
        )

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("Porównanie eksperymentów (posortowane po F1 macro)")
    print("=" * 60)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_row = df.iloc[0]
    best = next(
        (
            r
            for r in all_results
            if r["model"] == best_row["model"] and r["max_length"] == best_row["max_length"]
        ),
        None,
    )

    if baseline_results is not None:
        print(
            f"\nBaseline (zad. 2): accuracy={baseline_results['accuracy']:.4f}, "
            f"f1_macro={baseline_results['f1_macro']:.4f}"
        )
    if best is not None:
        print(
            f"Najlepsza konfiguracja: {best['model']}, max_length={best['max_length']} — "
            f"accuracy={best['accuracy']:.4f}, f1_macro={best['f1_macro']:.4f}"
        )

    return df, best


def plot_comparison(df, path=COMPARISON_PLOT_PATH):
    df = df.copy()
    df["label"] = df.apply(
        lambda row: f"{row['model'].split('/')[-1][:20]}\nL={int(row['max_length'])}", axis=1
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(df["label"], df["f1_macro"], color="steelblue")
    ax.set_ylabel("F1 macro")
    ax.set_title("Porównanie eksperymentów encoder-only")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")

    for bar, value in zip(bars, df["f1_macro"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Zapisano wykres porównawczy do: {path}")


def main():
    df = load_filtered_test_data()
    texts = df["sentence"].tolist()
    y_true = map_hf_labels_to_dataset(df["target"].tolist())

    print(f"Liczba próbek: {len(texts)}")
    print(f"Urządzenie: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Modele: {MODELS}")
    print(f"max_length: {MAX_LENGTHS}")

    all_results = []
    experiment_outputs = {}

    for model_name in MODELS:
        for max_length in MAX_LENGTHS:
            key = (model_name, max_length)
            results, model_labels, scores, y_pred = run_experiment(
                model_name, max_length, texts, y_true
            )
            all_results.append(results)
            experiment_outputs[key] = (model_labels, scores, y_pred)

    baseline_path = Path("encoder_results.json")
    baseline_results = None
    if baseline_path.exists():
        baseline_results = json.loads(baseline_path.read_text(encoding="utf-8"))

    comparison_df, best = compare_results(all_results, baseline_results)

    Path(RESULTS_JSON_PATH).write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    comparison_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nZapisano wyniki do: {RESULTS_JSON_PATH}")
    print(f"Zapisano tabelę porównawczą do: {RESULTS_CSV_PATH}")

    plot_comparison(comparison_df)

    if best is not None:
        print("\nNajlepsza konfiguracja — macierz pomyłek:")
        print_confusion_matrix(best["confusion_matrix"])

    return all_results, comparison_df


if __name__ == "__main__":
    main()
