import json
import time
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
    plot_confusion_matrix,
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
TIMING_PLOT_PATH = "encoder_exploration_timing.png"
BEST_CONFUSION_MATRIX_PATH = "encoder_exploration_best_confusion_matrix.png"



def run_experiment(model_name, max_length, texts, y_true, batch_size=16):
    print(f"\n--- Eksperyment: {model_name}, max_length={max_length} ---")
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    model_labels, scores = predict_with_max_length(
        model,
        tokenizer,
        texts,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        use_sigmoid=uses_sigmoid(model_name),
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prediction_time_sec = time.perf_counter() - start
    y_pred = map_model_labels_to_dataset(model_labels)

    results = evaluate_predictions(y_true, y_pred)
    results.update(
        {
            "model": model_name,
            "max_length": max_length,
            "num_samples": len(texts),
            "prediction_time_sec": prediction_time_sec,
            "samples_per_sec": len(texts) / prediction_time_sec if prediction_time_sec > 0 else 0,
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
        f"ambiguous={results['ambiguous_predictions']}, "
        f"time={prediction_time_sec:.2f}s ({results['samples_per_sec']:.1f} samples/s)"
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
                "prediction_time_sec": result["prediction_time_sec"],
                "samples_per_sec": result["samples_per_sec"],
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


def compare_timing(all_results):
    rows = [
        {
            "model": r["model"].split("/")[-1],
            "max_length": r["max_length"],
            "prediction_time_sec": r["prediction_time_sec"],
            "samples_per_sec": r["samples_per_sec"],
        }
        for r in all_results
    ]
    df = pd.DataFrame(rows).sort_values(["model", "max_length"]).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("Porównanie czasu predykcji na zbiorze (max_length)")
    print("=" * 60)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    pivot = df.pivot(index="max_length", columns="model", values="prediction_time_sec")
    print("\nCzas [s] — tabela przestawna (wiersze = max_length):")
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    return df


def plot_timing_comparison(all_results, path=TIMING_PLOT_PATH):
    df = pd.DataFrame(
        [
            {
                "model": r["model"],
                "max_length": r["max_length"],
                "prediction_time_sec": r["prediction_time_sec"],
            }
            for r in all_results
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name in MODELS:
        subset = df[df["model"] == model_name].sort_values("max_length")
        if subset.empty:
            continue
        label = model_name.split("/")[-1][:25]
        ax.plot(
            subset["max_length"],
            subset["prediction_time_sec"],
            marker="o",
            linewidth=2,
            label=label,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                f"{row['prediction_time_sec']:.1f}s",
                (row["max_length"], row["prediction_time_sec"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    ax.set_xlabel("max_length (tokeny wejścia)")
    ax.set_ylabel("Czas predykcji [s]")
    ax.set_title("Czas testu na całym zbiorze vs długość wejścia")
    ax.set_xticks(MAX_LENGTHS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Zapisano wykres czasu predykcji do: {path}")


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
    compare_timing(all_results)

    Path(RESULTS_JSON_PATH).write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    comparison_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nZapisano wyniki do: {RESULTS_JSON_PATH}")
    print(f"Zapisano tabelę porównawczą do: {RESULTS_CSV_PATH}")

    plot_comparison(comparison_df)
    plot_timing_comparison(all_results)

    if best is not None:
        print("\nNajlepsza konfiguracja — macierz pomyłek:")
        print_confusion_matrix(best["confusion_matrix"])
        model_short = best["model"].split("/")[-1]
        plot_confusion_matrix(
            best["confusion_matrix"],
            path=BEST_CONFUSION_MATRIX_PATH,
            title=f"Macierz pomyłek — {model_short}, max_length={best['max_length']}",
        )

    return all_results, comparison_df


if __name__ == "__main__":
    main()
