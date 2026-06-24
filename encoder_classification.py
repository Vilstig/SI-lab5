import json
from pathlib import Path

import torch
from transformers import pipeline

from encoder_utils import (
    CLASS_ORDER,
    evaluate_predictions,
    map_dataset_labels_to_hf,
    map_hf_labels_to_dataset,
    map_model_labels_to_dataset,
    plot_confusion_matrix,
    print_confusion_matrix,
    load_filtered_test_data,
)

ENCODER_MODEL_NAME = "Voicelab/herbert-base-cased-sentiment"
RESULTS_PATH = "encoder_results.json"
PREDICTIONS_PATH = "encoder_predictions.csv"
CONFUSION_MATRIX_PATH = "encoder_confusion_matrix.png"


def create_sentiment_pipeline(model_name=ENCODER_MODEL_NAME, device=None):
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=model_name,
        device=device,
        truncation=True,
    )


def predict_labels(sentiment_pipeline, texts, batch_size=16):
    model_labels = []
    scores = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        outputs = sentiment_pipeline(batch)
        if isinstance(outputs, dict):
            outputs = [outputs]
        for output in outputs:
            model_labels.append(output["label"])
            scores.append(output["score"])

    return model_labels, scores


def print_results(results):
    print("=" * 60)
    print("Wyniki klasyfikacji encoder-only")
    print("=" * 60)
    print(f"Model: {ENCODER_MODEL_NAME}")
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"F1 macro:    {results['f1_macro']:.4f}")
    print(f"F1 weighted: {results['f1_weighted']:.4f}")

    print("\nRaport per-class:")
    report = results["classification_report"]
    for label in CLASS_ORDER:
        stats = report[label]
        print(
            f"  {label:8s}  P={stats['precision']:.3f}  "
            f"R={stats['recall']:.3f}  F1={stats['f1-score']:.3f}  "
            f"support={int(stats['support'])}"
        )

    print_confusion_matrix(results["confusion_matrix"])
    plot_confusion_matrix(results["confusion_matrix"], path=CONFUSION_MATRIX_PATH)


def print_misclassified_examples(df, limit=5):
    wrong = df[df["true_label"] != df["predicted_label"]]
    print(f"\nPrzykłady błędnych klasyfikacji ({min(limit, len(wrong))} z {len(wrong)}):")
    for _, row in wrong.head(limit).iterrows():
        preview = row["sentence"][:100]
        if len(row["sentence"]) > 100:
            preview += "..."
        print(
            f"  true={row['true_label']}, pred={row['predicted_label']} "
            f"(score={row['score']:.3f}): {preview}"
        )


def save_results(results, path=RESULTS_PATH):
    Path(path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nZapisano wyniki do: {path}")


def save_predictions(df, path=PREDICTIONS_PATH):
    df.to_csv(path, index=False)
    print(f"Zapisano predykcje do: {path}")


def main():
    df = load_filtered_test_data()
    texts = df["sentence"].tolist()
    y_true = map_hf_labels_to_dataset(df["target"].tolist())

    print(f"Ładowanie modelu: {ENCODER_MODEL_NAME}")
    sentiment_pipeline = create_sentiment_pipeline()
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Urządzenie: {device_name}")
    print(f"Liczba próbek: {len(texts)}")

    model_labels, scores = predict_labels(sentiment_pipeline, texts)
    y_pred = map_model_labels_to_dataset(model_labels)

    assert len(y_pred) == len(y_true) == len(texts)

    results = evaluate_predictions(y_true, y_pred)
    results["model"] = ENCODER_MODEL_NAME
    results["num_samples"] = len(texts)

    print_results(results)

    predictions_df = df.copy()
    predictions_df["true_label"] = y_true
    predictions_df["model_label"] = model_labels
    predictions_df["predicted_label"] = y_pred
    predictions_df["predicted_hf_label"] = map_dataset_labels_to_hf(y_pred)
    predictions_df["score"] = scores

    print_misclassified_examples(predictions_df)
    save_results(results)
    save_predictions(predictions_df)

    return results, predictions_df


if __name__ == "__main__":
    main()
