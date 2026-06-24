import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from load_dataset import FILTERED_OUTPUT_PATH, LABEL_MAP

CLASS_ORDER = ["minus", "neutral", "plus"]

MODEL_TO_DATASET_LABEL = {
    "negative": "minus",
    "neutral": "neutral",
    "positive": "plus",
}

DATASET_TO_HF_LABEL = {v: k for k, v in LABEL_MAP.items()}

SIGMOID_MODELS = {
    "nie3e/sentiment-polish-gpt2-small",
}


def uses_sigmoid(model_name):
    return model_name in SIGMOID_MODELS


def load_filtered_test_data(path=FILTERED_OUTPUT_PATH):
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(f"Plik {path} jest pusty. Uruchom najpierw load_dataset.py.")
    return df


def normalize_model_label(label):
    key = label.lower().strip()
    mapping = {
        "negative": "minus",
        "neg": "minus",
        "positive": "plus",
        "pos": "plus",
        "neutral": "neutral",
        "ambiguous": "ambiguous",
        "negatywny": "minus",
        "pozytywny": "plus",
        "neutralny": "neutral",
    }
    if key in mapping:
        return mapping[key]
    if label in MODEL_TO_DATASET_LABEL:
        return MODEL_TO_DATASET_LABEL[label]
    raise ValueError(f"Nieznana etykieta modelu: {label}")


def map_model_labels_to_dataset(model_labels):
    return [normalize_model_label(label) for label in model_labels]


def map_hf_labels_to_dataset(hf_labels):
    return [LABEL_MAP[label] for label in hf_labels]


def map_dataset_labels_to_hf(readable_labels):
    return [DATASET_TO_HF_LABEL[label] for label in readable_labels]


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=CLASS_ORDER, zero_division=0),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", labels=CLASS_ORDER, zero_division=0
        ),
        "classification_report": classification_report(
            y_true, y_pred, labels=CLASS_ORDER, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
    }


def print_confusion_matrix(cm, labels=CLASS_ORDER):
    cm = np.asarray(cm)
    row_totals = cm.sum(axis=1)
    col_totals = cm.sum(axis=0)
    grand_total = cm.sum()

    col_width = max(len(label) for label in labels) + 2
    header = " " * 10 + "".join(f"{label:>{col_width}}" for label in labels) + f"{'Total':>{col_width}}"

    print("\nMacierz pomyłek (wiersze = prawda, kolumny = predykcja):")
    print(header)
    print("-" * len(header))

    for i, label in enumerate(labels):
        row = "".join(f"{cm[i, j]:>{col_width}}" for j in range(len(labels)))
        print(f"{label:10s}{row}{row_totals[i]:>{col_width}}")

    print("-" * len(header))
    totals_row = "".join(f"{col_totals[j]:>{col_width}}" for j in range(len(labels)))
    print(f"{'Total':10s}{totals_row}{grand_total:>{col_width}}")

    print("\nMacierz pomyłek (procenty wiersza):")
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        row_total = row_totals[i] or 1
        row = "".join(f"{cm[i, j] / row_total * 100:>{col_width}.1f}" for j in range(len(labels)))
        print(f"{label:10s}{row}{'100.0':>{col_width}}")


def plot_confusion_matrix(cm, labels=CLASS_ORDER, path="encoder_confusion_matrix.png", title="Macierz pomyłek"):
    cm = np.asarray(cm)
    row_totals = cm.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    cm_pct = cm / row_totals * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawda")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            pct = cm_pct[i, j]
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, label="Liczba próbek")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Zapisano wykres macierzy pomyłek do: {path}")


def load_model_and_tokenizer(model_name, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, int):
        device = torch.device("cuda" if device >= 0 and torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_with_max_length(
    model,
    tokenizer,
    texts,
    max_length=512,
    batch_size=16,
    device=None,
    use_sigmoid=False,
):
    if device is None:
        device = next(model.parameters()).device

    model_labels = []
    scores = []
    id2label = model.config.id2label

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoding = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors="pt",
        )
        encoding = {key: value.to(device) for key, value in encoding.items()}

        with torch.no_grad():
            logits = model(**encoding).logits
            if use_sigmoid:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)
            confidences, predictions = torch.max(probs, dim=-1)

        for pred_id, confidence in zip(predictions.cpu().tolist(), confidences.cpu().tolist()):
            label = id2label.get(pred_id, id2label.get(str(pred_id)))
            if label is None:
                raise ValueError(f"Brak etykiety dla id={pred_id} w model.config.id2label")
            model_labels.append(label)
            scores.append(confidence)

    return model_labels, scores
