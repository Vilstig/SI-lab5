import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import torch
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from encoder_utils import (
    CLASS_ORDER,
    evaluate_predictions,
    map_dataset_labels_to_hf,
    normalize_model_label,
    plot_confusion_matrix,
    print_confusion_matrix,
)
from llm_config import (
    CUSTOM_EXPERIMENTS,
    GRID_MODE,
    LLM_MODEL_NAME,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    MAX_TEXT_CHARS,
    PARSERS_TO_RUN,
    PROMPT_REGISTRY,
    PROMPTS_TO_RUN,
    QUANTIZATION_TO_RUN,
    TEMPERATURES_TO_RUN,
)

LABEL_PATTERN = re.compile(
    r"\b(positive|negative|neutral|pozytywny|negatywny|neutralny)\b",
    re.IGNORECASE,
)
OUTPUT_MARKERS = ("Class:", "Klasa:", "JSON:")


class SentimentOutput(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")


@dataclass
class ExperimentConfig:
    name: str
    prompt_name: str
    temperature: float
    parser: str = "regex"
    quantization: str | None = None
    aspect: str = "custom"
    model_name: str = LLM_MODEL_NAME
    max_new_tokens: int = MAX_NEW_TOKENS

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        return cls(
            name=data["name"],
            prompt_name=data["prompt_name"],
            temperature=data["temperature"],
            parser=data.get("parser", "regex"),
            quantization=data.get("quantization"),
            aspect=data.get("aspect", "custom"),
            model_name=data.get("model_name", LLM_MODEL_NAME),
            max_new_tokens=data.get("max_new_tokens", MAX_NEW_TOKENS),
        )

    def label(self) -> str:
        quant = self.quantization or "fp16"
        return f"{self.prompt_name} | T={self.temperature} | {self.parser} | {quant}"


class ModelCache:
    def __init__(self):
        self._models: dict[tuple, tuple] = {}
        self._chains: dict[tuple, tuple] = {}

    def get_chain(self, config: ExperimentConfig):
        model_key = (config.model_name, config.quantization)
        chain_key = (
            config.model_name,
            config.quantization,
            config.prompt_name,
            config.temperature,
            config.max_new_tokens,
        )

        if chain_key not in self._chains:
            if model_key not in self._models:
                self._models[model_key] = load_model_and_tokenizer(
                    config.model_name, quantization=config.quantization
                )
            model, tokenizer = self._models[model_key]
            prompt_template = PROMPT_REGISTRY[config.prompt_name]
            chain = create_llm_chain(
                model,
                tokenizer,
                prompt_template,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
            )
            self._chains[chain_key] = (chain, tokenizer)

        return self._chains[chain_key]

    def clear(self):
        self._models.clear()
        self._chains.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def is_quantization_available():
    try:
        import bitsandbytes  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False


def load_model_and_tokenizer(model_name, quantization=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantization == "4bit":
        if not is_quantization_available():
            raise RuntimeError(
                "Kwantyzacja 4-bit wymaga bitsandbytes i GPU. "
                "Zainstaluj: pip install bitsandbytes"
            )
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif quantization == "8bit":
        if not is_quantization_available():
            raise RuntimeError("Kwantyzacja 8-bit wymaga bitsandbytes i GPU.")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    return model, tokenizer


def create_llm_chain(model, tokenizer, prompt_template, temperature, max_new_tokens=MAX_NEW_TOKENS):
    do_sample = temperature > 0
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=temperature if do_sample else 1.0,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    prompt = PromptTemplate.from_template(prompt_template)
    return prompt | llm


def truncate_text(text, tokenizer, max_chars=MAX_TEXT_CHARS, max_tokens=MAX_INPUT_TOKENS):
    if len(text) > max_chars:
        text = text[:max_chars]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text


def extract_generated_suffix(raw_text: str) -> str:
    text = str(raw_text).strip()
    for marker in OUTPUT_MARKERS:
        if marker in text:
            return text.split(marker)[-1].strip()
    return text


def parse_llm_output_regex(raw_text):
    if not raw_text:
        return None
    text = extract_generated_suffix(raw_text)
    match = LABEL_PATTERN.search(text)
    if match:
        return match.group(1).lower()
    return None


def parse_llm_output_json(raw_text):
    if not raw_text:
        return None
    text = extract_generated_suffix(raw_text)
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    parser = JsonOutputParser(pydantic_object=SentimentOutput)
    try:
        parsed = parser.parse(text)
        if isinstance(parsed, dict):
            return parsed.get("sentiment", "").lower() or None
        return str(parsed.sentiment).lower()
    except Exception:
        match = re.search(r'"sentiment"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return parse_llm_output_regex(raw_text)



def get_parser(parser_name: str):
    if parser_name == "regex":
        return parse_llm_output_regex
    if parser_name == "json":
        return parse_llm_output_json
    raise ValueError(f"Nieznany parser: {parser_name}")


def map_parsed_label(parsed_label):
    if parsed_label is None:
        return "unknown"
    try:
        return normalize_model_label(parsed_label)
    except ValueError:
        return "unknown"


def predict_all(chain, texts, tokenizer, parser_fn, desc="LLM inference"):
    raw_outputs = []
    parsed_labels = []

    for text in tqdm(texts, desc=desc):
        truncated = truncate_text(text, tokenizer)
        raw = chain.invoke({"text": truncated})
        parsed = parser_fn(raw)
        raw_outputs.append(raw)
        parsed_labels.append(map_parsed_label(parsed))

    return raw_outputs, parsed_labels


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


def build_experiment_grid(only_aspect: str | None = None) -> list[ExperimentConfig]:
    if GRID_MODE == "explicit":
        configs = [ExperimentConfig.from_dict(spec) for spec in CUSTOM_EXPERIMENTS]
    else:
        configs = []
        for prompt_name, temperature, parser, quantization in product(
            PROMPTS_TO_RUN, TEMPERATURES_TO_RUN, PARSERS_TO_RUN, QUANTIZATION_TO_RUN
        ):
            if prompt_name == "json_en" and parser != "json":
                continue
            if prompt_name != "json_en" and parser == "json":
                continue
            quant_suffix = quantization or "fp16"
            configs.append(
                ExperimentConfig(
                    name=f"{prompt_name}_t{temperature}_{parser}_{quant_suffix}",
                    prompt_name=prompt_name,
                    temperature=temperature,
                    parser=parser,
                    quantization=quantization,
                )
            )

    if only_aspect is not None:
        configs = [c for c in configs if c.aspect == only_aspect]

    available = []
    for config in configs:
        if config.quantization and not is_quantization_available():
            print(f"Pominięto {config.name}: brak bitsandbytes/GPU")
            continue
        if config.prompt_name not in PROMPT_REGISTRY:
            raise ValueError(f"Nieznany prompt: {config.prompt_name}")
        available.append(config)
    return available


def run_experiment(config: ExperimentConfig, texts, y_true, model_cache: ModelCache):
    print(f"\n--- Eksperyment: {config.name} ({config.label()}) ---")
    chain, tokenizer = model_cache.get_chain(config)
    parser_fn = get_parser(config.parser)
    raw_outputs, y_pred = predict_all(
        chain, texts, tokenizer, parser_fn, desc=f"LLM [{config.name}]"
    )

    results = evaluate_with_unknown(y_true, y_pred)
    results.update(
        {
            "name": config.name,
            "model": config.model_name,
            "prompt_name": config.prompt_name,
            "temperature": config.temperature,
            "parser": config.parser,
            "quantization": config.quantization,
            "aspect": config.aspect,
            "num_samples": len(texts),
        }
    )

    print(
        f"  Accuracy={results['accuracy']:.4f}, "
        f"F1 macro={results['f1_macro']:.4f}, "
        f"unparsed={results['unparsed_count']}"
    )
    return results, raw_outputs, y_pred


def print_results(results, save_artifacts=True, confusion_matrix_path=None, title=None):
    print("=" * 60)
    print("Wyniki klasyfikacji LLM (decoder-only)")
    print("=" * 60)
    print(f"Model:       {results.get('model', 'n/a')}")
    if "prompt_name" in results:
        print(f"Prompt:      {results['prompt_name']}")
    print(f"Temperature: {results.get('temperature', 'n/a')}")
    if "parser" in results:
        print(f"Parser:      {results['parser']}")
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
    if save_artifacts and confusion_matrix_path:
        plot_confusion_matrix(
            results["confusion_matrix"],
            path=confusion_matrix_path,
            title=title or "Macierz pomyłek — LLM",
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


def build_predictions_df(df, y_true, raw_outputs, y_pred):
    predictions_df = df.copy()
    predictions_df["true_label"] = y_true
    predictions_df["raw_output"] = raw_outputs
    predictions_df["predicted_label"] = y_pred
    predictions_df["predicted_hf_label"] = [
        map_dataset_labels_to_hf([label])[0] if label != "unknown" else "unknown"
        for label in y_pred
    ]
    return predictions_df


def compare_with_baselines(results, encoder_path="encoder_results.json", llm_path="llm_results.json"):
    print("\nPorównanie z baseline:")
    encoder_path = Path(encoder_path)
    if encoder_path.exists():
        baseline = json.loads(encoder_path.read_text(encoding="utf-8"))
        print(
            f"  Encoder (zad. 2): accuracy={baseline['accuracy']:.4f}, "
            f"f1_macro={baseline['f1_macro']:.4f}"
        )
    llm_path = Path(llm_path)
    if llm_path.exists():
        baseline = json.loads(llm_path.read_text(encoding="utf-8"))
        print(
            f"  LLM (zad. 4):     accuracy={baseline['accuracy']:.4f}, "
            f"f1_macro={baseline['f1_macro']:.4f}"
        )
    print(
        f"  Bieżący:          accuracy={results['accuracy']:.4f}, "
        f"f1_macro={results['f1_macro']:.4f}"
    )
