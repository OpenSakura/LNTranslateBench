# Light Novel Translation Benchmark

A comprehensive benchmark system for evaluating LLM translation quality using a 3-stage workflow:

1. **Translation** - Generate translations from multiple models
2. **Comparison** - Use LLM-as-judge to compare translation pairs
3. **Dashboard** - Compute ELO ratings and visualize results

This benchmark translates Japanese light novel samples using different LLM models and ranks them based on translation quality.

## Workflow

```
│ Stage 1: Translation                                        │
│ Tool:   LinguaGacha (External)                              │
│ Input:  samples/*.txt                                       │
│ Output: raw_translations/*.txt                              │
│ Action: Use utils/wrap_lingua_gacha.py to format output     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Comparison                                        │
│ Script: compare_translations.sh                             │
│ Input:  translated_results/*.translated.txt                 │
│ Output: comparison_results/*.json                           │
│ Purpose: LLM-as-judge pairwise comparison (win/loss/tie)   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: Dashboard                                         │
│ Script: serve_dashboard.py                                  │
│ Input:  comparison_results/*.json                           │
│ Output: Local web dashboard (ELO + charts + drilldowns)     │
│ Purpose: Calculate ELO ratings and explore results          │
└─────────────────────────────────────────────────────────────┘
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your API key and settings
```

3. Set your API key(s) (choose one method):

```bash
# Option 1: Export in current shell
export API_KEY=your-api-key-here
export JUDGE_API_KEY=your-judge-api-key-here

# Option 2: Add to .env file (not tracked by git)
echo "API_KEY=your-api-key-here" > .env
echo "JUDGE_API_KEY=your-judge-api-key-here" >> .env
source .env

# Option 3: Pass directly via command line (see examples below)
```

## Usage

### Quick Start with Bash Scripts

The benchmark consists of 3 stages:
1. **Translation** - Generate translations from all models
2. **Comparison** - Use LLM-as-judge to compare translation pairs
3. **Ranking** - Calculate ELO ratings and model rankings

### Stage 1: Translation with LinguaGacha

This benchmark relies on **LinguaGacha** for the initial translation stage.

1.  **Generate Translations**: Use [LinguaGacha](https://github.com/neavo/LinguaGacha) to translate the files in `samples/` from Japanese to Chinese (or your target language).
2.  **Save Output**: Save the raw translated `.txt` files to a temporary directory (e.g., `raw_translations/`).
3.  **Format for Benchmark**: Run the provided wrapper script to add the required metadata headers for Stage 2.

```bash
python utils/wrap_lingua_gacha.py \
  --input-dir raw_translations \
  --output-dir translated_results \
  --model "Your-Model-Name"
```

This will populate `translated_results/` with files named `sample.Your-Model-Name.translated.txt` which are ready for comparison.

#### Stage 2: Comparison

Uses an LLM as a judge to perform scheduled pairwise comparisons of translations (round-robin by default).

```bash
# Configure judge API (OpenAI-compatible recommended)
export JUDGE_API_URL="http://localhost:8000/v1/chat/completions"
export JUDGE_MODEL="your-judge-model-here"
export JUDGE_API_KEY="your-judge-api-key-here"

# Compare translations using LLM-as-judge (round-robin schedule by default)
./compare_translations.sh

# Example output: comparison_results/
#   - sample1.model1_vs_model2.json
#   - sample1.model4_vs_model9.json
#   - ... (one round per sample)
#   - comparison_summary.json
```

Each comparison JSON includes an anonymous A/B mapping, a detailed category breakdown, and a final winner (`A`/`B`/`tie`).

Round-robin note: with `N` models, each sample produces `N/2` comparisons. To cover every pair once you need `N-1` samples (e.g. 16 models → 15 samples → 120 total comparisons).

#### Stage 3: Dashboard (ELO + Charts)

Serves a local, interactive dashboard that computes Elo from the current `comparison_results/` and lets you:
- Switch between judge models (or combined results)
- View Elo leaderboard + histogram
- View a pairwise heatmap
- Expand chapters to see all pairs and judge outputs

**Live Demo**: [https://bench.opensakura.com/](https://bench.opensakura.com/)

To run locally:

```bash
python serve_dashboard.py --host 127.0.0.1 --port 8001
# Open http://127.0.0.1:8001 in your browser
```

### Direct Python Usage

**Basic Usage** - Translate samples with a single model:

```bash
python benchmark_translate.py http://localhost:8000/translate --models gpt-4
```

**Multiple Models** - Translate with multiple models (comma-separated):

```bash
python benchmark_translate.py http://localhost:8000/translate --models gpt-4,claude-3-opus,gemini-pro
```

**Custom Directories** - Specify custom sample and output directories:

```bash
python benchmark_translate.py http://localhost:8000/translate \
  --models gpt-4 \
  --samples ./my_samples \
  --output ./results
```

**Different Languages** - Translate from English to Japanese:

```bash
python benchmark_translate.py http://localhost:8000/translate \
  --models gpt-4 \
  --source en \
  --target ja
```

### Included Results

We ran this benchmark once and the results are available in the `translated_results/` and `comparison_results/` folders. The dashboard is live at [https://bench.opensakura.com/](https://bench.opensakura.com/).

The run included the following models:

**DeepSeek Family:**
- DeepSeek-R1-0528
- DeepSeek-V3-250324
- DeepSeek-V3.1
- DeepSeek-V3.1-Terminus
- DeepSeek-V3.2

**GLM Family:**
- GLM-4.5
- GLM-4.6
- GLM-4.7

**Qwen Family:**
- Qwen3-30B-A3B-Instruct-2507
- Qwen3-30B-A3B-Thinking-2507
- Qwen3-235B-A22B-Instruct-2507
- Qwen3-235B-A22B-Thinking-2507
- Qwen3-Next-80B-A3B-Instruct
- Qwen3-Next-80B-A3B-Thinking

**Other Models:**
- Kimi-K2
- MiniMax-M2

## Command Line Arguments

- `api_url` (required): Translation API endpoint URL (e.g., `http://localhost:8000/api/v1/translate`)
- `--models, -m` (required): Comma-separated list of model names
- `--samples, -s`: Directory containing sample text files (default: `samples`)
- `--output, -o`: Output directory for results (default: `translated_results`)
- `--source`: Source language code (default: `ja`)
- `--target`: Target language code (default: `en`)
- `--api-key, -k`: API key for authentication (can also use `API_KEY` environment variable)
- `--api-key-header`: Custom API key header name (default: `X-API-Key`)

## API Authentication

The benchmark supports API key authentication in three ways:

1. **Command line argument**:
   ```bash
   ./translate_all.sh
   # Or with custom script:
   ./translate_custom.sh --api-key your-api-key-here
   ```

2. **Environment variable** (recommended):
   ```bash
   export API_KEY=your-api-key-here
   ./translate_all.sh
   ```

3. **Direct Python usage**:
   ```bash
   python benchmark_translate.py http://localhost:8000/api/v1/translate \
     --models DeepSeek-V3.2 \
     --api-key your-api-key-here
   ```

## Output Format

Each translation result is saved as a separate file with the format:

```
<sample_name>.<model_name>.translated.txt
```

Each file contains:

1. **Metadata section** with:
   - Sample file name
   - Model used
   - Timestamp
   - Source/target languages
   - Text lengths
   - Success status
   - Error details (if failed)

2. **Separator**

3. **Translated text**

Example output file: `kakuyomu.1177354054893080355.ch1.gpt-4.translated.txt`

## Benchmark Summary

After completion, a JSON summary is created at `translated_results/benchmark_summary.json` containing:

- Total number of translations
- Success/failure counts
- Details for each translation
