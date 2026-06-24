"""Edytowalna konfiguracja LLM — zmieniaj tutaj prompty i siatkę eksperymentów."""

LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 15
MAX_TEXT_CHARS = 2000
MAX_INPUT_TOKENS = 512

# --- Prompty (klucz → szablon LangChain z {text}) ---
PROMPT_REGISTRY = {
    "basic_en": """Classify the text sentiment into one of three classes: positive, negative, neutral.
Text: {text}
Class:""",
    "polish": """Sklasyfikuj wydźwięk tekstu do jednej z trzech klas: pozytywny, negatywny, neutralny.
Tekst: {text}
Klasa:""",
    "json_en": """Classify the sentiment of the text. Reply with JSON only, no other text.
Use exactly one of: positive, negative, neutral.
Format: {{"sentiment": "<label>"}}
Text: {text}
JSON:""",
}

# --- Tryb siatki: "explicit" (lista poniżej) lub "cartesian" (iloczyn list) ---
GRID_MODE = "explicit"

# Używane gdy GRID_MODE == "cartesian"
PROMPTS_TO_RUN = ["basic_en", "polish", "json_en"]
TEMPERATURES_TO_RUN = [0.0, 0.1, 0.5]
PARSERS_TO_RUN = ["regex", "json"]
QUANTIZATION_TO_RUN = [None, "4bit"]

# Jawna lista eksperymentów (GRID_MODE == "explicit")
# Pola: name, prompt_name, temperature, parser, quantization (opcjonalnie), aspect
CUSTOM_EXPERIMENTS = [
    # Oś 1: prompt (T=0.1)
    {"name": "prompt_basic_en", "prompt_name": "basic_en", "temperature": 0.1, "parser": "regex", "aspect": "prompt"},
    {"name": "prompt_polish", "prompt_name": "polish", "temperature": 0.1, "parser": "regex", "aspect": "prompt"},
    {"name": "prompt_json_en", "prompt_name": "json_en", "temperature": 0.1, "parser": "json", "aspect": "prompt"},
    # Oś 2: temperatura (basic_en)
    {"name": "temp_0.0", "prompt_name": "basic_en", "temperature": 0.0, "parser": "regex", "aspect": "temperature"},
    {"name": "temp_0.5", "prompt_name": "basic_en", "temperature": 0.5, "parser": "regex", "aspect": "temperature"},
    {"name": "temp_1.0", "prompt_name": "basic_en", "temperature": 1.0, "parser": "regex", "aspect": "temperature"},
    # Oś 3: kwantyzacja
    {
        "name": "quant_4bit",
        "prompt_name": "basic_en",
        "temperature": 0.1,
        "parser": "regex",
        "quantization": "4bit",
        "aspect": "quantization",
    },
]

# Baseline zadania 4
BASELINE_EXPERIMENT = {
    "name": "baseline",
    "prompt_name": "basic_en",
    "temperature": 0.1,
    "parser": "regex",
    "aspect": "baseline",
}

# Ścieżki artefaktów — zadanie 4
BASELINE_RESULTS_PATH = "llm_results.json"
BASELINE_PREDICTIONS_PATH = "llm_predictions.csv"
BASELINE_CONFUSION_MATRIX_PATH = "llm_confusion_matrix.png"

# Ścieżki artefaktów — zadanie 5
EXPLORATION_RESULTS_JSON = "llm_exploration_results.json"
EXPLORATION_RESULTS_CSV = "llm_exploration_results.csv"
EXPLORATION_COMPARISON_PLOT = "llm_exploration_comparison.png"
EXPLORATION_BEST_PREDICTIONS = "llm_exploration_best_predictions.csv"
EXPLORATION_BEST_CONFUSION_MATRIX = "llm_exploration_best_confusion_matrix.png"

ENCODER_BASELINE_PATH = "encoder_results.json"
LLM_BASELINE_PATH = "llm_results.json"
