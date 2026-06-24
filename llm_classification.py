import argparse
import json
import re
from pathlib import Path

import torch
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from encoder_utils import (
    CLASS_ORDER,
    evaluate_predictions,
    load_filtered_test_data,
    map_dataset_labels_to_hf,
    map_hf_labels_to_dataset,
    normalize_model_label,
    plot_confusion_matrix,
    print_confusion_matrix,
)

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 15
MAX_TEXT_CHARS = 2000
MAX_INPUT_TOKENS = 512

RESULTS_PATH = "llm_results.json"
PREDICTIONS_PATH = "llm_predictions.csv"
CONFUSION_MATRIX_PATH = "llm_confusion_matrix.png"
ENCODER_BASELINE_PATH = "encoder_results.json"

PROMPT_TEMPLATE = """Classify the text sentiment into one of three classes: positive, negative, neutral.
Text: {text}
Class:"""

LABEL_PATTERN = re.compile(r"\b(positive|negative|neutral)\b", re.IGNORECASE)


def truncate_text(text, tokenizer, max_chars=MAX_TEXT_CHARS, max_tokens=MAX_INPUT_TOKENS):
    if len(text) > max_chars:
        text = text[:max_chars]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text


def create_llm_chain(model_name=LLM_MODEL_NAME, temperature=TEMPERATURE):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        do_sample=True,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    return chain, tokenizer


def parse_llm_output(raw_text):
    if not raw_text:
        return None

    text = str(raw_text).strip()
    class_marker = "Class:"
    if class_marker in text:
        text = text.split(class_marker)[-1]

    match = LABEL_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    return None


def map_parsed_label(parsed_label):
    if parsed_label is None:
        return "unknown"
    try:
        return normalize_model_label(parsed_label)
    except ValueError:
        return "unknown"


def predict_all(chain, texts, tokenizer):
    raw_outputs = []
    parsed_labels = []

    for text in tqdm(texts, desc="LLM inference"):
        truncated = truncate_text(text, tokenizer)
        raw = chain.invoke({"text": truncated})
        parsed = parse_llm_output(raw)
        raw_outputs.append(raw)
        parsed_labels.append(map_parsed_label(parsed))

    return raw_outputs, parsed_labels


def print_results(results):
    print("=" * 60)
    print("Wyniki klasyfikacji LLM (decoder-only)")
    print("=" * 60)
    print(f"Model:       {results['model']}")
    print(f"Temperature: {results['temperature']}")
    print(f"Accuracy:    {results['accuracy']:.4f}")
    print(f"F1 macro:    {results['f1_macro']:.4f}")
    print(f"F1 weighted: {results['f1_weighted']:.4f}")
    print(f"Unparsed:    {results['unparsed_count']}")

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
    plot_confusion_matrix(
        results["confusion_matrix"],
        path=CONFUSION_MATRIX_PATH,
        title="Macierz pomyłek — LLM",
    )


def print_misclassified_examples(df, limit=5):
    wrong = df[df["true_label"] != df["predicted_label"]]
    print(f"\nPrzykłady błędnych klasyfikacji ({min(limit, len(wrong))} z {len(wrong)}):")
    for _, row in wrong.head(limit).iterrows():
        preview = row["sentence"][:80]
        if len(row["sentence"]) > 80:
            preview += "..."
        print(
            f"  true={row['true_label']}, pred={row['predicted_label']}: "
            f"{row['raw_output'][:60]!r} | {preview}"
        )


def compare_with_encoder(results):
    baseline_path = Path(ENCODER_BASELINE_PATH)
    if not baseline_path.exists():
        return
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    print("\nPorównanie z encoderem (Herbert, zad. 2):")
    print(f"  Encoder: accuracy={baseline['accuracy']:.4f}, f1_macro={baseline['f1_macro']:.4f}")
    print(f"  LLM:     accuracy={results['accuracy']:.4f}, f1_macro={results['f1_macro']:.4f}")


def evaluate_with_unknown(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    unparsed_count = sum(1 for label in y_pred if label == "unknown")

    y_pred_for_report = []
    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label == "unknown":
            wrong_label = next(label for label in CLASS_ORDER if label != true_label)
            y_pred_for_report.append(wrong_label)
        else:
            y_pred_for_report.append(pred_label)

    results = evaluate_predictions(y_true, y_pred_for_report)
    results["accuracy"] = accuracy
    results["unparsed_count"] = unparsed_count
    return results


def main(limit=None):
    df = load_filtered_test_data()
    if limit is not None:
        df = df.head(limit)

    texts = df["sentence"].tolist()
    y_true = map_hf_labels_to_dataset(df["target"].tolist())

    print(f"Ładowanie modelu: {LLM_MODEL_NAME}")
    print(f"Urządzenie: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Liczba próbek: {len(texts)}")

    chain, tokenizer = create_llm_chain()
    raw_outputs, y_pred = predict_all(chain, texts, tokenizer)

    results = evaluate_with_unknown(y_true, y_pred)
    results.update(
        {
            "model": LLM_MODEL_NAME,
            "temperature": TEMPERATURE,
            "num_samples": len(texts),
        }
    )

    print_results(results)
    compare_with_encoder(results)

    predictions_df = df.copy()
    predictions_df["true_label"] = y_true
    predictions_df["raw_output"] = raw_outputs
    predictions_df["predicted_label"] = y_pred
    predictions_df["predicted_hf_label"] = [
        map_dataset_labels_to_hf([label])[0] if label != "unknown" else "unknown"
        for label in y_pred
    ]

    print_misclassified_examples(predictions_df)
    Path(RESULTS_PATH).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"\nZapisano wyniki do: {RESULTS_PATH}")
    print(f"Zapisano predykcje do: {PREDICTIONS_PATH}")

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
