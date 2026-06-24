import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from encoder_utils import load_filtered_test_data, map_hf_labels_to_dataset, print_confusion_matrix
from llm_config import (
    ENCODER_BASELINE_PATH,
    EXPLORATION_ARTIFACTS_DIR,
    EXPLORATION_COMPARISON_PLOT,
    EXPLORATION_RESULTS_CSV,
    EXPLORATION_RESULTS_JSON,
    LLM_BASELINE_PATH,
    LLM_MODEL_NAME,
)
from llm_utils import (
    ModelCache,
    build_experiment_grid,
    build_predictions_df,
    compare_with_baselines,
    plot_confusion_matrix,
    run_experiment,
)


def compare_results(all_results, encoder_baseline=None, llm_baseline=None):
    rows = []
    for result in all_results:
        rows.append(
            {
                "name": result["name"],
                "aspect": result.get("aspect", ""),
                "prompt_name": result["prompt_name"],
                "temperature": result["temperature"],
                "parser": result["parser"],
                "quantization": result.get("quantization") or "fp16",
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "f1_weighted": result["f1_weighted"],
                "unparsed_count": result["unparsed_count"],
            }
        )

    df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("Porównanie eksperymentów LLM (posortowane po F1 macro)")
    print("=" * 60)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best_row = df.iloc[0]
    best = next((r for r in all_results if r["name"] == best_row["name"]), None)

    if encoder_baseline is not None:
        print(
            f"\nBaseline encoder (zad. 2): accuracy={encoder_baseline['accuracy']:.4f}, "
            f"f1_macro={encoder_baseline['f1_macro']:.4f}"
        )
    if llm_baseline is not None:
        print(
            f"Baseline LLM (zad. 4):     accuracy={llm_baseline['accuracy']:.4f}, "
            f"f1_macro={llm_baseline['f1_macro']:.4f}"
        )
    if best is not None:
        print(
            f"Najlepsza konfiguracja: {best['name']} — "
            f"accuracy={best['accuracy']:.4f}, f1_macro={best['f1_macro']:.4f}"
        )

    return df, best


def plot_comparison(df, path=EXPLORATION_COMPARISON_PLOT):
    df = df.copy()
    df["label"] = df.apply(
        lambda row: (
            f"{row['name']}\n{row['prompt_name']}\nT={row['temperature']}"
        ),
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 1.2), 5))
    bars = ax.bar(df["label"], df["f1_macro"], color="steelblue")
    ax.set_ylabel("F1 macro")
    ax.set_title("Porównanie eksperymentów LLM (zad. 5)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")

    for bar, value in zip(bars, df["f1_macro"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Zapisano wykres porównawczy do: {path}")


def save_experiment_artifacts(experiment_outputs, all_results, output_dir=EXPLORATION_ARTIFACTS_DIR):
    output_dir = Path(output_dir)
    predictions_dir = output_dir / "predictions"
    matrices_dir = output_dir / "confusion_matrices"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    results_by_name = {result["name"]: result for result in all_results}
    for name, predictions_df in experiment_outputs.items():
        out_df = pd.DataFrame(
            {
                "eksperyment": name,
                "zdanie": predictions_df["sentence"],
                "odpowiedz": predictions_df["raw_output"],
                "wynik_prawidlowy": predictions_df["true_label"],
                "wynik_llm": predictions_df["predicted_label"],
            }
        )
        out_df.to_csv(predictions_dir / f"{name}.csv", index=False)

        result = results_by_name[name]
        plot_confusion_matrix(
            result["confusion_matrix"],
            path=matrices_dir / f"{name}.png",
            title=f"Macierz pomyłek — {name}",
        )

    print(f"Zapisano predykcje ({len(experiment_outputs)}) do: {predictions_dir}/")
    print(f"Zapisano macierze pomyłek ({len(experiment_outputs)}) do: {matrices_dir}/")


def main(limit=None, only_aspect=None):
    save_artifacts = limit is None

    df = load_filtered_test_data()
    if limit is not None:
        df = df.head(limit)

    texts = df["sentence"].tolist()
    y_true = map_hf_labels_to_dataset(df["target"].tolist())
    grid = build_experiment_grid(only_aspect=only_aspect)

    print(f"Liczba próbek: {len(texts)}")
    print(f"Urządzenie: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Model: {LLM_MODEL_NAME}")
    print(f"Liczba eksperymentów: {len(grid)}")
    for config in grid:
        print(f"  - {config.name}: {config.label()}")

    model_cache = ModelCache()
    all_results = []
    experiment_outputs = {}

    for config in grid:
        results, raw_outputs, y_pred = run_experiment(config, texts, y_true, model_cache)
        all_results.append(results)
        experiment_outputs[config.name] = build_predictions_df(df, y_true, raw_outputs, y_pred)

    encoder_baseline = None
    encoder_path = Path(ENCODER_BASELINE_PATH)
    if encoder_path.exists():
        encoder_baseline = json.loads(encoder_path.read_text(encoding="utf-8"))

    llm_baseline = None
    llm_path = Path(LLM_BASELINE_PATH)
    if llm_path.exists():
        llm_baseline = json.loads(llm_path.read_text(encoding="utf-8"))

    comparison_df, best = compare_results(all_results, encoder_baseline, llm_baseline)

    if best is not None:
        print("\nNajlepsza konfiguracja — macierz pomyłek:")
        print_confusion_matrix(best["confusion_matrix"])
        compare_with_baselines(best)

    if save_artifacts:
        Path(EXPLORATION_RESULTS_JSON).write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        comparison_df.to_csv(EXPLORATION_RESULTS_CSV, index=False)
        plot_comparison(comparison_df)
        save_experiment_artifacts(experiment_outputs, all_results)

        print(f"\nZapisano wyniki do: {EXPLORATION_RESULTS_JSON}")
        print(f"Zapisano tabelę porównawczą do: {EXPLORATION_RESULTS_CSV}")
    else:
        print(f"\nPominięto zapis plików (--limit {limit}); uruchom bez --limit dla pełnych artefaktów.")

    model_cache.clear()
    return all_results, comparison_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM parameter exploration (task 5)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for quick testing (default: all)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["prompt", "temperature", "quantization", "parser"],
        help="Run only experiments from one exploration axis",
    )
    args = parser.parse_args()
    main(limit=args.limit, only_aspect=args.only)
