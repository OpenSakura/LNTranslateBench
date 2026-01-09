# Project Structure

## Overview

This project implements a 3-stage benchmark for evaluating LLM translation quality:

```
llm_translate_benchmark/
├── samples/                    # Input: Japanese light novel samples
│   ├── syosetu.*.txt
│   ├── kakuyomu.*.txt
│   ├── alphapolis.*.txt
│   └── novelup.*.txt
│
├── translated_results/         # Stage 1 Output: Translations from all models
│   ├── sample1.model1.translated.txt
│   ├── sample1.model2.translated.txt
│   └── benchmark_summary.json
│
├── comparison_results/         # Stage 2 Output: Pairwise comparisons
│   ├── sample1.model1_vs_model2.json
│   └── ...
│
├── benchmark_translate.py      # Core translation script
├── translate_all.sh           # Stage 1: Run all models
├── translate_custom.sh        # Stage 1: Custom model selection
├── compare_translations.py    # Stage 2: LLM-as-judge (core)
├── compare_translations.sh    # Stage 2: LLM-as-judge (wrapper)
├── serve_dashboard.py         # Stage 3: Local web dashboard (ELO + drilldowns)
├── dashboard/                 # Dashboard static assets (HTML/CSS/JS)
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variable template
└── README.md                 # Main documentation
```

## Scripts

### Stage 1: Translation

**translate_all.sh**
- Translates all samples with all 16 models
- Default: Japanese → Chinese
- Output: `translated_results/`

**translate_custom.sh**
- Interactive model selection
- Supports filtering by family (--deepseek, --glm, --qwen, --thinking)
- Custom API endpoint and parameters

**benchmark_translate.py**
- Core Python script for translation
- API key authentication support
- Handles requests/responses with translation API

### Stage 2: Comparison

**compare_translations.sh**
- Load all translations from Stage 1
- Generate scheduled pairwise comparisons (round-robin by default)
- Use LLM-as-judge to evaluate anonymous A/B
- Output: per-pair JSON (winner/tie + breakdown)

### Stage 3: Dashboard (ELO + Charts)

**serve_dashboard.py**
- Load comparison results from Stage 2
- Compute ELO ratings (and pairwise stats)
- Serve an interactive web UI (heatmap + histogram + chapter drilldowns)

## Models

The benchmark supports 16 models across multiple families:

- **DeepSeek** (5): R1-0528, V3-250324, V3.1, V3.1-Terminus, V3.2
- **GLM** (3): 4.5, 4.6, 4.7
- **Qwen** (6): 30B/235B Instruct/Thinking variants, Next-80B variants
- **Other** (2): Kimi-K2, MiniMax-M2

## Output Format

### Stage 1: Translation Files

Each translation result file contains:

```
================================================================================
TRANSLATION METADATA
================================================================================
Sample File: kakuyomu.1177354054893080355.ch1
Model: DeepSeek-V3.2
Timestamp: 2026-01-04T20:00:00
Source Language: ja
Target Language: zh
Original Text Length: 3245 characters
Translated Text Length: 2987 characters
Success: True
Status Code: 200
================================================================================

================================================================================
TRANSLATED TEXT
================================================================================

[Translated content here...]
```

### Stage 2: Comparison Files

```json
{
  "schema_version": 1,
  "sample": "kakuyomu.1177354054893080355.ch1",
  "models": {
    "model_1": "DeepSeek-V3.2",
    "model_2": "Qwen3-235B-A22B-Instruct-2507"
  },
  "presentation": { "A": "DeepSeek-V3.2", "B": "Qwen3-235B-A22B-Instruct-2507" },
  "decision": { "winner": "A", "confidence": 0.72, "scores": { "accuracy": { "A": 8, "B": 7, "notes": "..." } } },
  "winner_model": "DeepSeek-V3.2"
}
```

### Stage 3: Dashboard API (Example)

The dashboard computes Elo on load and serves JSON for the UI.

Example: `GET /api/summary?judge=combined`

```json
{
  "schema_version": 1,
  "judge": "combined",
  "num_matches": 128,
  "num_models": 16,
  "ratings": [{ "model": "GLM-4.7", "elo": 1580.12, "wins": 10, "losses": 4, "ties": 2, "games": 16 }],
  "pairwise": { "models": ["GLM-4.7", "DeepSeek-V3.2"], "expected": [[null, 0.61], [0.39, null]] }
}
```

## Environment Variables

Set these in `.env` or export them:

```bash
API_URL=http://localhost:8000/api/v1/translate
API_KEY=your-api-key-here
JUDGE_API_URL=http://localhost:8000/v1/chat/completions
JUDGE_MODEL=your-judge-model-here
JUDGE_API_KEY=your-judge-api-key-here
SOURCE_LANG=ja
TARGET_LANG=zh
SAMPLES_DIR=./samples
OUTPUT_DIR=./translated_results
TRANSLATIONS_DIR=./translated_results
COMPARISON_DIR=./comparison_results
RANKINGS_DIR=./rankings
```
