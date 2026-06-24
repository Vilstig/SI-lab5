import argparse
import json
from pathlib import Path

import torch

from encoder_utils import load_filtered_test_data, map_hf_labels_to_dataset
from llm_config import (
    BASELINE_CONFUSION_MATRIX_PATH,
    BASELINE_EXPERIMENT,
    BASELINE_PREDICTIONS_PATH,
    BASELINE_RESULTS_PATH,
    LLM_MODEL_NAME,
)
from llm_utils import (
    ExperimentConfig,
    ModelCache,
    build_predictions_df,
    compare_with_baselines,
    print_misclassified_examples,
    print_results,
    run_experiment,
)


def main(limit=None):
    save_artifacts = limit is None

    df = load_filtered_test_data()
    if limit is not None:
        df = df.head(limit)

    texts = df["sentence"].tolist()
    y_true = map_hf_labels_to_dataset(df["target"].tolist())

    config = ExperimentConfig.from_dict(BASELINE_EXPERIMENT)
    print(f"Ładowanie modelu: {LLM_MODEL_NAME}")
    print(f"Urządzenie: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Liczba próbek: {len(texts)}")

    model_cache = ModelCache()
    results, raw_outputs, y_pred = run_experiment(config, texts, y_true, model_cache)

    print_results(
        results,
        save_artifacts=save_artifacts,
        confusion_matrix_path=BASELINE_CONFUSION_MATRIX_PATH,
        title="Macierz pomyłek — LLM",
    )
    compare_with_baselines(results)

    predictions_df = build_predictions_df(df, y_true, raw_outputs, y_pred)
    print_misclassified_examples(predictions_df)

    if save_artifacts:
        Path(BASELINE_RESULTS_PATH).write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        predictions_df.to_csv(BASELINE_PREDICTIONS_PATH, index=False)
        print(f"\nZapisano wyniki do: {BASELINE_RESULTS_PATH}")
        print(f"Zapisano predykcje do: {BASELINE_PREDICTIONS_PATH}")
    else:
        print(f"\nPominięto zapis plików (--limit {limit}); uruchom bez --limit dla pełnych artefaktów.")

    model_cache.clear()
    return results, predictions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM sentiment classification (task 4)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples for quick testing (default: all)",
    )
    args = parser.parse_args()
    main(limit=args.limit)
