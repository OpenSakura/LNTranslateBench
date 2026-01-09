#!/bin/bash
# Stage 2: Comparison - Use LLM-as-judge to compare translation pairs
# Part of the 3-stage benchmark: Translation → Comparison → Ranking

set -euo pipefail

show_usage() {
  cat << EOF
Usage: $0 [OPTIONS] [-- <extra args passed to python>]

Stage 2 (Comparison): Pairwise LLM-as-judge on Stage 1 translations.
Default schedule: round-robin (one round per sample, N/2 comparisons per sample for N models).

Required (or set env vars):
  --judge-base-url URL       Judge API base URL (env: JUDGE_BASE_URL)
  --judge-model MODEL        Judge model name (env: JUDGE_MODEL)

Optional:
  --judge-key KEY            Judge API key (env: JUDGE_API_KEY)
  --translations DIR         Stage 1 output dir (env: TRANSLATIONS_DIR, default: ./translated_results)
  --samples DIR              Source samples dir (env: SAMPLES_DIR, default: ./samples)
  --output DIR               Comparison output dir (env: COMPARISON_DIR, default: ./comparison_results)
  --concurrent-requests N    Number of concurrent judge requests (env: CONCURRENT_REQUESTS, default: 1)
  --seed N                   Seed for round-robin offset (env: SEED, default: 0)
  --tracing                  Enable OpenTelemetry tracing (env: ENABLE_TRACING)
  --otlp-endpoint URL        OTLP traces endpoint (env: OTLP_ENDPOINT)
  --otlp-auth-header HDR     OTLP auth header (env: OTLP_AUTH_HEADER)
  --otlp-project NAME        OTLP project name (env: OTLP_PROJECT_NAME)

Examples:
  export JUDGE_BASE_URL="http://localhost:8000/v1"
  export JUDGE_MODEL="gpt-4o-mini"
  export JUDGE_API_KEY="..."
  ./compare_translations.sh

  ./compare_translations.sh --judge-base-url http://localhost:8000/v1 --judge-model gpt-4.1 --output ./comparison_results

  # With tracing enabled
  export OTLP_ENDPOINT="http://localhost:4318"
  export ENABLE_TRACING="true"
  ./compare_translations.sh --tracing --otlp-endpoint http://localhost:4318

Advanced:
  ./compare_translations.sh -- --max-pairs 10 --delay 0.5 --response-format json_object
EOF
}

JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-pro}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"

TRANSLATIONS_DIR="${TRANSLATIONS_DIR:-./translated_results}"
SAMPLES_DIR="${SAMPLES_DIR:-./samples}"
COMPARISON_DIR="${COMPARISON_DIR:-./comparison_results}"
CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS:-20}"
SEED="${SEED:-1}"

# Tracing options
ENABLE_TRACING="${ENABLE_TRACING:-true}"
OTLP_ENDPOINT="${OTLP_ENDPOINT:-[REDACTED]}"
OTLP_AUTH_HEADER="${OTLP_AUTH_HEADER:-}"
OTLP_PROJECT_NAME="${OTLP_PROJECT_NAME:-}"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
    --judge-base-url)
      JUDGE_BASE_URL="$2"
      shift 2
      ;;
    --judge-model)
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --judge-key)
      JUDGE_API_KEY="$2"
      shift 2
      ;;
    --translations)
      TRANSLATIONS_DIR="$2"
      shift 2
      ;;
    --samples)
      SAMPLES_DIR="$2"
      shift 2
      ;;
    --output)
      COMPARISON_DIR="$2"
      shift 2
      ;;
    --concurrent-requests)
      CONCURRENT_REQUESTS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --tracing)
      ENABLE_TRACING="true"
      shift
      ;;
    --otlp-endpoint)
      OTLP_ENDPOINT="$2"
      shift 2
      ;;
    --otlp-auth-header)
      OTLP_AUTH_HEADER="$2"
      shift 2
      ;;
    --otlp-project)
      OTLP_PROJECT_NAME="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$JUDGE_BASE_URL" || -z "$JUDGE_MODEL" ]]; then
  echo "Error: Missing judge config. Set JUDGE_BASE_URL and JUDGE_MODEL (or pass --judge-base-url/--judge-model)." >&2
  echo "" >&2
  show_usage >&2
  exit 2
fi

echo "=========================================="
echo "Stage 2: Translation Comparison (LLM-as-judge)"
echo "=========================================="
echo "Translations: $TRANSLATIONS_DIR"
echo "Samples:      $SAMPLES_DIR"
echo "Output:       $COMPARISON_DIR"
echo "Judge URL:    $JUDGE_BASE_URL"
echo "Judge model:  $JUDGE_MODEL"
echo "=========================================="

PY_ARGS=(
  --translations-dir "$TRANSLATIONS_DIR"
  --samples-dir "$SAMPLES_DIR"
  --output-dir "$COMPARISON_DIR"
  --judge-base-url "$JUDGE_BASE_URL"
  --judge-model "$JUDGE_MODEL"
  --concurrent-requests "$CONCURRENT_REQUESTS"
  --seed "$SEED"
)
if [[ -n "$JUDGE_API_KEY" ]]; then
  PY_ARGS+=(--judge-api-key "$JUDGE_API_KEY")
fi

# Add tracing parameters if enabled
if [[ "$ENABLE_TRACING" == "true" ]]; then
  PY_ARGS+=(--tracing)
fi
if [[ -n "$OTLP_ENDPOINT" ]]; then
  PY_ARGS+=(--otlp-endpoint "$OTLP_ENDPOINT")
fi
if [[ -n "$OTLP_AUTH_HEADER" ]]; then
  PY_ARGS+=(--otlp-auth-header "$OTLP_AUTH_HEADER")
fi
if [[ -n "$OTLP_PROJECT_NAME" ]]; then
  PY_ARGS+=(--otlp-project-name "$OTLP_PROJECT_NAME")
fi

python compare_translations.py "${PY_ARGS[@]}" "${EXTRA_ARGS[@]}"
