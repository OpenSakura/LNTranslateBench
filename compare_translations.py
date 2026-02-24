#!/usr/bin/env python3
"""
Stage 2: Comparison - Use an LLM as a judge to compare translation pairs.

Reads Stage 1 outputs from translated_results/ and produces pairwise comparison
JSON files in comparison_results/ using an implicit round-robin schedule (one round per sample).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

add_span_event: Any
create_span: Any
init_tracing: Any
record_exception: Any
set_span_attributes: Any
shutdown_tracing: Any

try:
    from utils.tracing import (
        add_span_event as _add_span_event,
        create_span as _create_span,
        init_tracing as _init_tracing,
        record_exception as _record_exception,
        set_span_attributes as _set_span_attributes,
        shutdown_tracing as _shutdown_tracing,
    )

    add_span_event = _add_span_event
    create_span = _create_span
    init_tracing = _init_tracing
    record_exception = _record_exception
    set_span_attributes = _set_span_attributes
    shutdown_tracing = _shutdown_tracing

    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

    def _noop(*args: Any, **kwargs: Any) -> None:
        return None

    def _noop_span(*args: Any, **kwargs: Any):
        from contextlib import nullcontext

        return nullcontext()

    add_span_event = _noop
    create_span = _noop_span
    init_tracing = _noop
    record_exception = _noop
    set_span_attributes = _noop
    shutdown_tracing = _noop


def _noop_context():
    """No-op context manager for when tracing is disabled."""
    from contextlib import nullcontext

    return nullcontext()


TRANSLATION_FILE_GLOB = "*.translated.txt"
TRANSLATED_TEXT_HEADER = "TRANSLATED TEXT"


@dataclass(frozen=True)
class TranslationRecord:
    sample: str
    model: str
    source_lang: Optional[str]
    target_lang: Optional[str]
    success: bool
    translated_text: str
    file_path: Path


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_translation_file(path: Path) -> TranslationRecord:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    sample: Optional[str] = None
    model: Optional[str] = None
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    success: Optional[bool] = None

    for line in lines:
        if line.startswith("Sample File:"):
            sample = line.split(":", 1)[1].strip()
        elif line.startswith("Model:"):
            model = line.split(":", 1)[1].strip()
        elif line.startswith("Source Language:"):
            source_lang = line.split(":", 1)[1].strip()
        elif line.startswith("Target Language:"):
            target_lang = line.split(":", 1)[1].strip()
        elif line.startswith("Success:"):
            success = _parse_bool(line.split(":", 1)[1])

    if sample is None or model is None or success is None:
        raise ValueError(f"Missing required metadata in {path}")

    translated_text = ""
    for i, line in enumerate(lines):
        if line.strip() == TRANSLATED_TEXT_HEADER:
            # File format from benchmark_translate.py:
            #  line i: "TRANSLATED TEXT"
            #  line i+1: "====...===="
            #  line i+2: "" (blank)
            translated_text = "\n".join(lines[i + 3 :]).strip("\n")
            break

    if not translated_text:
        # Fallback: treat the whole file as translated text, if the expected header is missing.
        translated_text = text.strip("\n")

    return TranslationRecord(
        sample=sample,
        model=model,
        source_lang=source_lang,
        target_lang=target_lang,
        success=success,
        translated_text=translated_text,
        file_path=path,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sanitize_for_filename(value: str) -> str:
    # Keep common safe characters; replace everything else with underscore.
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "unknown"


def stable_coinflip(seed_text: str) -> bool:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return (digest[0] % 2) == 0


def stable_mod(seed_text: str, modulus: int) -> int:
    if modulus <= 0:
        return 0
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % modulus


def round_robin_rounds(players: List[str]) -> List[List[Tuple[str, str]]]:
    """
    Deterministic round-robin schedule using the circle method.

    For N (even) players, returns N-1 rounds, each with N/2 pairings.
    For N (odd) players, adds a BYE and omits BYE pairings.
    """
    if len(players) < 2:
        return []

    ordered = list(players)
    bye: Optional[str] = None
    if len(ordered) % 2 == 1:
        bye = "__BYE__"
        while bye in ordered:
            bye = "_" + bye
        ordered.append(bye)

    fixed = ordered[0]
    rotating = ordered[1:]
    n = len(ordered)
    rounds: List[List[Tuple[str, str]]] = []

    for _round_idx in range(n - 1):
        lineup = [fixed] + rotating
        half = n // 2
        pairings: List[Tuple[str, str]] = []
        for i in range(half):
            a = lineup[i]
            b = lineup[n - 1 - i]
            if bye and (a == bye or b == bye):
                continue
            pairings.append((a, b))
        rounds.append(pairings)

        # Rotate all players except the fixed anchor.
        rotating = [rotating[-1]] + rotating[:-1]

    return rounds


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove leading ```lang and trailing ```
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = strip_code_fences(text)

    # Fast path: exact JSON object.
    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass

    # Salvage: grab the first {...} block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        loaded = json.loads(candidate)
        if isinstance(loaded, dict):
            return loaded

    raise ValueError("Judge output did not contain a valid JSON object")


def build_judge_messages(
    *,
    source_lang: str,
    target_lang: str,
    source_text: Optional[str],
    translation_a: str,
    translation_b: str,
) -> List[Dict[str, str]]:
    system = (
        "你是一位公正的轻小说翻译质量评判员。\n"
        "你将比较翻译A和翻译B。它们是匿名的；请勿猜测其来源。\n"
        "评估哪个翻译整体更好，考虑以下方面：\n"
        "1) 准确性（意思保留）\n"
        "2) 流畅性（目标语言自然表达）\n"
        "3) 风格/语气（轻小说叙述/对话）\n"
        "4) 术语/名称一致性\n"
        "5) 连贯性/可读性\n"
        "6) 格式/标点符号\n\n"
        "请先进行详细分析，然后在最后输出一个有效的JSON对象作为你的最终判决。\n"
        "JSON格式必须为：\n"
        "{{\n"
        '  "winner": "A" | "B" | "tie",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "scores": {{\n'
        '    "accuracy": {{"A": 1-10, "B": 1-10, "notes": "..."}} ,\n'
        '    "fluency": {{"A": 1-10, "B": 1-10, "notes": "..."}} ,\n'
        '    "style": {{"A": 1-10, "B": 1-10, "notes": "..."}} ,\n'
        '    "terminology": {{"A": 1-10, "B": 1-10, "notes": "..."}} ,\n'
        '    "coherence": {{"A": 1-10, "B": 1-10, "notes": "..."}} ,\n'
        '    "formatting": {{"A": 1-10, "B": 1-10, "notes": "..."}}\n'
        "  }},\n"
        '  "key_differences": ["..."],\n'
        '  "final_summary": "..." \n'
        "}}\n\n"
        "平局指导：只有在整体质量极其接近或权衡相互抵消时才选择'tie'。\n"
        "使用提供的语言：源语言={source_lang}，目标语言={target_lang}。\n"
    ).format(source_lang=source_lang, target_lang=target_lang)

    # Always use interleaved structure as requested, trusting that line counts align
    # (or handling mismatches gracefully by using empty strings).
    lines_a = translation_a.strip().splitlines()
    lines_b = translation_b.strip().splitlines()
    lines_source = source_text.strip().splitlines() if source_text else []

    user_content_parts = ["请比较以下分段对照的文本：\n"]

    max_len = max(len(lines_a), len(lines_b))
    if source_text:
        max_len = max(max_len, len(lines_source))

    for i in range(max_len):
        segment_str = f"段落 {i + 1}:\n"

        if source_text:
            s_line = lines_source[i] if i < len(lines_source) else ""
            segment_str += f"原文: {s_line}\n"

        a_line = lines_a[i] if i < len(lines_a) else ""
        b_line = lines_b[i] if i < len(lines_b) else ""

        segment_str += f"A: {a_line}\n"
        segment_str += f"B: {b_line}\n"
        user_content_parts.append(segment_str)

    user = "\n".join(user_content_parts)

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp_path.replace(path)


def judge_pair(
    *,
    judge_base_url: str,
    judge_model: str,
    judge_api_key: Optional[str],
    temperature: float,
    timeout_s: int,
    messages: List[Dict[str, str]],
    response_api: str,
    response_format: Optional[str],
    reasoning_effort: Optional[str],
) -> Tuple[str, Dict[str, Any], str]:
    """Call the judge LLM using the OpenAI python client.

    Args:
        judge_base_url: Base URL for the API (e.g., "http://localhost:8000/v1")
        judge_model: Model name to use
        judge_api_key: API key (uses Authorization: Bearer header)
        temperature: Sampling temperature
        timeout_s: Request timeout in seconds
        messages: List of chat messages
        response_api: Which OpenAI API to call ("chat_completions" or "responses")
        response_format: Optional response format (e.g., "json_object")
        reasoning_effort: Optional reasoning effort ("low", "medium", "high")

    Returns:
        Tuple of (content, response_json, request_id)
    """
    client = OpenAI(
        api_key=judge_api_key or "dummy-key",
        base_url=judge_base_url,
        timeout=timeout_s,
    )

    if response_api == "responses":
        request_params: Dict[str, Any] = {
            "model": judge_model,
            "input": messages,
            "temperature": temperature,
        }
        if reasoning_effort:
            request_params["reasoning"] = {"effort": reasoning_effort}

        # Responses API uses `text.format` for structured outputs.
        if response_format == "json_object":
            request_params["text"] = {"format": {"type": "json_object"}}

        output_parts: List[str] = []
        final_response: Any = None

        # Stream if supported by the installed SDK.
        if hasattr(client.responses, "stream"):
            with client.responses.stream(**request_params) as stream:
                for event in stream:
                    if getattr(event, "type", None) == "response.output_text.delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            output_parts.append(str(delta))
                final_response = stream.get_final_response()
        else:
            for event in client.responses.create(**request_params, stream=True):
                if getattr(event, "type", None) == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        output_parts.append(str(delta))
                elif getattr(event, "type", None) == "response.completed":
                    final_response = getattr(event, "response", None)

        if final_response is None:
            # As a last resort, fall back to non-streaming.
            final_response = client.responses.create(**request_params)

        content = "".join(output_parts).strip()
        if not content:
            content = (getattr(final_response, "output_text", None) or "").strip()

        response_json: Dict[str, Any] = {
            "id": getattr(final_response, "id", None),
            "object": getattr(final_response, "object", None),
            "created": getattr(final_response, "created", None),
            "model": getattr(final_response, "model", judge_model),
            "output_text": content,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        }

        usage = getattr(final_response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "input_tokens", None)
            completion_tokens = getattr(usage, "output_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            if (
                prompt_tokens is not None
                or completion_tokens is not None
                or total_tokens is not None
            ):
                response_json["usage"] = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }

        request_id = str(getattr(final_response, "id", "") or "")
        return content, response_json, request_id

    if response_api != "chat_completions":
        raise ValueError(
            f"Unsupported response_api: {response_api!r} (expected 'chat_completions' or 'responses')"
        )

    request_params = {
        "model": judge_model,
        "messages": messages,
        "temperature": temperature,
    }
    if reasoning_effort:
        request_params["reasoning_effort"] = reasoning_effort
    if response_format:
        request_params["response_format"] = {"type": response_format}

    response = client.chat.completions.create(**request_params)

    content = response.choices[0].message.content or ""

    response_json = {
        "id": response.id,
        "object": response.object,
        "created": response.created,
        "model": response.model,
        "choices": [
            {
                "index": choice.index,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content,
                },
                "finish_reason": choice.finish_reason,
            }
            for choice in response.choices
        ],
    }

    if response.usage:
        response_json["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    request_id = response.id or ""
    return content, response_json, request_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Compare translations using LLM-as-judge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--translations-dir",
        default="translated_results",
        help="Stage 1 output directory",
    )
    parser.add_argument(
        "--samples-dir",
        default="samples",
        help="Directory containing source sample .txt files",
    )
    parser.add_argument(
        "--output-dir",
        default="comparison_results",
        help="Directory to write comparison JSON files",
    )

    parser.add_argument(
        "--judge-base-url",
        default=os.environ.get("JUDGE_BASE_URL"),
        help="Judge API base URL (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--judge-model", default=os.environ.get("JUDGE_MODEL"), help="Judge model name"
    )
    parser.add_argument(
        "--judge-api-key", default=os.environ.get("JUDGE_API_KEY"), help="Judge API key"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Judge sampling temperature"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries on JSON validation failure",
    )
    parser.add_argument(
        "--retry-temp-increase",
        type=float,
        default=0.3,
        help="Temperature increase per retry",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="HTTP timeout in seconds for judge calls (default: 900)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, help="Sleep between judge calls (seconds)"
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=1,
        help="Number of concurrent judge requests (1 = sequential, >1 = parallel)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Stop after this many comparisons (0 = no limit)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run comparisons even if output exists",
    )
    parser.add_argument(
        "--seed",
        default="0",
        help="Deterministic seed affecting schedule offset and A/B presentation",
    )
    parser.add_argument(
        "--response-format",
        choices=["json_object"],
        default=None,
        help="Send OpenAI-compatible response_format (optional; may not be supported by all APIs)",
    )
    parser.add_argument(
        "--response-api",
        choices=["chat_completions", "responses"],
        default=os.environ.get("JUDGE_RESPONSE_API", "chat_completions"),
        help="Which OpenAI API to use for judge calls",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default=os.environ.get("JUDGE_REASONING_EFFORT"),
        help="Reasoning effort for supported models/APIs",
    )

    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated subset of models to include (default: all discovered from translations)",
    )
    parser.add_argument(
        "--focus-model",
        default="",
        help="Only run comparisons involving this model (default: off)",
    )
    parser.add_argument(
        "--focus-mode",
        choices=["scheduled", "all"],
        default="scheduled",
        help=(
            "When --focus-model is set: 'scheduled' filters the round-robin schedule to only that model; "
            "'all' compares the focus model vs every other active model per sample."
        ),
    )

    # Tracing options
    parser.add_argument(
        "--tracing", action="store_true", help="Enable OpenTelemetry tracing"
    )
    parser.add_argument(
        "--otlp-endpoint",
        default=os.environ.get("OTLP_ENDPOINT"),
        help="OTLP endpoint for traces",
    )
    parser.add_argument(
        "--otlp-auth-header",
        default=os.environ.get("OTLP_AUTH_HEADER"),
        help="OTLP auth header value",
    )
    parser.add_argument(
        "--otlp-project-name",
        default=os.environ.get("OTLP_PROJECT_NAME"),
        help="Project name for tracing",
    )

    args = parser.parse_args()

    if not args.judge_base_url or not args.judge_model:
        print(
            "Error: --judge-base-url and --judge-model are required (or set JUDGE_BASE_URL/JUDGE_MODEL).",
            file=sys.stderr,
        )
        sys.exit(2)

    # Initialize tracing if requested
    if args.tracing and TRACING_AVAILABLE:
        init_tracing(
            otlp_endpoint=args.otlp_endpoint,
            otlp_auth_header=args.otlp_auth_header,
            project_name=args.otlp_project_name,
            service_name="translation-judge",
        )
        print("OpenTelemetry tracing enabled")
    elif args.tracing:
        print(
            "Warning: Tracing requested but utils.tracing module not available",
            file=sys.stderr,
        )

    translations_dir = Path(args.translations_dir)
    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)

    translation_files = sorted(translations_dir.glob(TRANSLATION_FILE_GLOB))
    if not translation_files:
        print(
            f"No translation files found in {translations_dir} (glob: {TRANSLATION_FILE_GLOB})",
            file=sys.stderr,
        )
        sys.exit(1)

    records: List[TranslationRecord] = []
    for path in translation_files:
        try:
            record = parse_translation_file(path)
        except Exception as e:
            print(f"Skipping unreadable translation file {path}: {e}", file=sys.stderr)
            continue
        if not record.success:
            continue
        records.append(record)

    if not records:
        print(
            f"No successful translations found in {translations_dir}", file=sys.stderr
        )
        sys.exit(1)

    by_sample: Dict[str, Dict[str, TranslationRecord]] = {}
    for r in records:
        by_sample.setdefault(r.sample, {})
        by_sample[r.sample][r.model] = r

    samples_sorted = sorted(by_sample.keys())
    model_counts: Dict[str, int] = {}
    for sample in samples_sorted:
        for model in by_sample[sample].keys():
            model_counts[model] = model_counts.get(model, 0) + 1

    discovered_models = sorted(model_counts.keys())

    requested_models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if requested_models:
        missing = [m for m in requested_models if m not in discovered_models]
        if missing:
            print(
                "Error: --models includes unknown model(s): " + ", ".join(missing),
                file=sys.stderr,
            )
            print("Discovered models: " + ", ".join(discovered_models), file=sys.stderr)
            sys.exit(2)
        active_models = sorted(requested_models)
    else:
        active_models = discovered_models

    focus_model = (args.focus_model or "").strip() or None
    if focus_model and focus_model not in active_models:
        print(
            f"Error: --focus-model {focus_model!r} is not in the active model set.",
            file=sys.stderr,
        )
        print("Active models: " + ", ".join(active_models), file=sys.stderr)
        sys.exit(2)

    rounds = round_robin_rounds(active_models)
    if not rounds:
        print("Not enough models to schedule comparisons.", file=sys.stderr)
        sys.exit(1)

    round_offset = (
        0 if str(args.seed).strip() in {"", "0"} else stable_mod(args.seed, len(rounds))
    )

    schedule_type = "round_robin"
    if focus_model:
        schedule_type = "focus_all" if args.focus_mode == "all" else "focus_round_robin"

    scheduled_total = 0
    scheduled_available = 0
    scheduled_missing_translation = 0
    for idx, sample in enumerate(samples_sorted):
        round_idx = (idx + round_offset) % len(rounds)

        models_map = {
            m: r for (m, r) in by_sample[sample].items() if m in active_models
        }
        if focus_model:
            if args.focus_mode == "all":
                pairings = [(focus_model, m) for m in active_models if m != focus_model]
            else:
                pairings = [p for p in rounds[round_idx] if focus_model in p]
        else:
            pairings = rounds[round_idx]

        scheduled_total += len(pairings)
        for m1, m2 in pairings:
            if m1 in models_map and m2 in models_map:
                scheduled_available += 1
            else:
                scheduled_missing_translation += 1

    print(
        f"Loaded {len(records)} successful translations across {len(by_sample)} sample(s)"
    )
    print(f"Models discovered: {len(discovered_models)} (active: {len(active_models)})")
    if focus_model:
        print(f"Focus model: {focus_model} (mode: {args.focus_mode})")

    if schedule_type == "round_robin":
        print(
            f"Schedule: round-robin ({len(rounds)} rounds, {len(rounds[0])} matches/round, offset={round_offset})"
        )
    elif schedule_type == "focus_round_robin":
        print(
            f"Schedule: focus round-robin ({len(rounds)} rounds, <=1 match/round, offset={round_offset})"
        )
    else:
        print(
            f"Schedule: focus all-opponents ({len(active_models) - 1} matches/sample)"
        )

    print(f"Planned comparisons (scheduled): {scheduled_total}")
    print(f"Planned comparisons (available):  {scheduled_available}")
    if scheduled_missing_translation:
        print(
            f"Note: {scheduled_missing_translation} scheduled matches skipped due to missing translations."
        )
    print(f"Judge base URL: {args.judge_base_url}")
    print(f"Judge model: {args.judge_model}")
    print(f"Output dir: {output_dir}")

    completed = 0
    skipped_existing = 0
    skipped_missing = 0
    failed = 0
    started_at = datetime.now().isoformat()

    total_work = scheduled_available
    processed = 0

    # Thread-safe counters for parallel processing
    counter_lock = Lock()

    # Collect all comparison tasks
    comparison_tasks: List[Dict[str, Any]] = []

    # One round per sample; round index increments with sample index.
    for sample_idx, sample in enumerate(samples_sorted):
        models_map = {
            m: r for (m, r) in by_sample[sample].items() if m in active_models
        }
        if len(models_map) < 2:
            continue

        source_text: Optional[str] = None
        source_file = samples_dir / f"{sample}.txt"
        if source_file.exists():
            source_text = source_file.read_text(encoding="utf-8", errors="replace")

        # Pick languages from any record in this sample group.
        any_record = next(iter(models_map.values()))
        source_lang = any_record.source_lang or "ja"
        target_lang = any_record.target_lang or "zh"

        round_idx = (sample_idx + round_offset) % len(rounds)
        if focus_model:
            if args.focus_mode == "all":
                pairings = [(focus_model, m) for m in active_models if m != focus_model]
            else:
                pairings = [p for p in rounds[round_idx] if focus_model in p]
        else:
            pairings = rounds[round_idx]

        for model_1, model_2 in pairings:
            if model_1 not in models_map or model_2 not in models_map:
                with counter_lock:
                    skipped_missing += 1
                continue

            canonical_1, canonical_2 = sorted([model_1, model_2])
            out_name = (
                f"{sanitize_for_filename(sample)}."
                f"{sanitize_for_filename(canonical_1)}_vs_{sanitize_for_filename(canonical_2)}"
                f"_by_{sanitize_for_filename(args.judge_model)}.json"
            )
            out_path = output_dir / out_name

            if out_path.exists() and not args.overwrite:
                with counter_lock:
                    skipped_existing += 1
                    processed += 1
                continue

            record_1 = models_map[canonical_1]
            record_2 = models_map[canonical_2]

            # Deterministically assign which real model appears as anonymous A/B.
            flip_seed = f"{args.seed}|{sample}|{canonical_1}|{canonical_2}"
            if stable_coinflip(flip_seed):
                a_model, b_model = canonical_1, canonical_2
                a_text, b_text = record_1.translated_text, record_2.translated_text
                a_file, b_file = record_1.file_path, record_2.file_path
            else:
                a_model, b_model = canonical_2, canonical_1
                a_text, b_text = record_2.translated_text, record_1.translated_text
                a_file, b_file = record_2.file_path, record_1.file_path

            messages = build_judge_messages(
                source_lang=source_lang,
                target_lang=target_lang,
                source_text=source_text,
                translation_a=a_text,
                translation_b=b_text,
            )

            # Collect task for processing
            comparison_tasks.append(
                {
                    "sample": sample,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "source_file": source_file,
                    "source_text": source_text,
                    "canonical_1": canonical_1,
                    "canonical_2": canonical_2,
                    "a_model": a_model,
                    "b_model": b_model,
                    "a_text": a_text,
                    "b_text": b_text,
                    "a_file": a_file,
                    "b_file": b_file,
                    "out_path": out_path,
                    "messages": messages,
                }
            )

    # Limit tasks if max_pairs is set
    if args.max_pairs and len(comparison_tasks) > args.max_pairs:
        comparison_tasks = comparison_tasks[: args.max_pairs]

    # Function to process a single comparison task
    def process_comparison(task: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Process a single comparison. Returns (status, error_message)."""
        nonlocal completed, failed, processed

        sample = task["sample"]
        source_lang = task["source_lang"]
        target_lang = task["target_lang"]
        source_file = task["source_file"]
        source_text = task["source_text"]
        canonical_1 = task["canonical_1"]
        canonical_2 = task["canonical_2"]
        a_model = task["a_model"]
        b_model = task["b_model"]
        a_text = task["a_text"]
        b_text = task["b_text"]
        a_file = task["a_file"]
        b_file = task["b_file"]
        out_path = task["out_path"]
        messages = task["messages"]

        started_ts = datetime.now().isoformat()
        result_payload: Dict[str, Any] = {
            "schema_version": 1,
            "timestamp": started_ts,
            "sample": sample,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_file": str(source_file) if source_file.exists() else None,
            "models": {"model_1": canonical_1, "model_2": canonical_2},
            "presentation": {"A": a_model, "B": b_model},
            "translation_files": {"A": str(a_file), "B": str(b_file)},
            "input_sha256": {
                "source": sha256_text(source_text) if source_text else None,
                "A": sha256_text(a_text),
                "B": sha256_text(b_text),
            },
            "judge": {
                "base_url": args.judge_base_url,
                "model": args.judge_model,
                "response_api": args.response_api,
                "reasoning_effort": args.reasoning_effort,
            },
            "status": "started",
        }
        atomic_write_json(out_path, result_payload)

        # Retry loop for JSON validation failures
        attempts: List[Dict[str, Any]] = []
        decision = None
        last_error = None

        # Create a span for this comparison if tracing is enabled
        span_context = (
            create_span(
                "judge_comparison",
                attributes={
                    "sample": sample,
                    "model_1": canonical_1,
                    "model_2": canonical_2,
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                    "judge.model": args.judge_model,
                    "judge.response_api": args.response_api,
                    "judge.reasoning_effort": args.reasoning_effort or "",
                    "judge.temperature": args.temperature,
                    "max_retries": args.max_retries,
                },
            )
            if TRACING_AVAILABLE
            else None
        )

        with span_context if span_context is not None else _noop_context():
            for attempt_num in range(args.max_retries + 1):
                current_temp = args.temperature + (
                    attempt_num * args.retry_temp_increase
                )
                attempt_info: Dict[str, Any] = {
                    "attempt": attempt_num + 1,
                    "temperature": current_temp,
                }

                content: str = ""
                response_json: Optional[Dict[str, Any]] = None
                request_id: Optional[str] = None

                # Add event for retry attempt
                if TRACING_AVAILABLE and attempt_num > 0:
                    add_span_event(
                        f"retry_attempt_{attempt_num}",
                        attributes={"temperature": current_temp},
                    )

                try:
                    content, response_json, request_id = judge_pair(
                        judge_base_url=args.judge_base_url,
                        judge_model=args.judge_model,
                        judge_api_key=args.judge_api_key,
                        temperature=current_temp,
                        timeout_s=args.timeout,
                        messages=messages,
                        response_api=args.response_api,
                        response_format=args.response_format,
                        reasoning_effort=args.reasoning_effort,
                    )

                    attempt_info.update(
                        {
                            "judge_request_id": request_id,
                            "judge_response": response_json,
                            "status": "success",
                        }
                    )

                    # Try to extract and validate JSON
                    decision = extract_json_object(content)
                    winner = decision.get("winner")
                    if winner not in {"A", "B", "tie"}:
                        raise ValueError(f'Invalid "winner" value: {winner!r}')

                    attempts.append(attempt_info)

                    # Add success attributes to span
                    if TRACING_AVAILABLE:
                        set_span_attributes(
                            {
                                "judge.attempts": attempt_num + 1,
                                "judge.winner": winner,
                                "judge.request_id": request_id or "",
                                "judge.success": True,
                            }
                        )

                    # Success! Break out of retry loop
                    break

                except Exception as e:
                    last_error = e
                    attempt_info.update(
                        {
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    # Save partial info if available
                    if response_json is not None:
                        attempt_info["judge_response"] = response_json
                    if request_id is not None:
                        attempt_info["judge_request_id"] = request_id

                    attempts.append(attempt_info)

                    # Record exception in span
                    if TRACING_AVAILABLE:
                        record_exception(e)

                    # If this was the last attempt, don't sleep
                    if attempt_num < args.max_retries:
                        print(
                            f"  Attempt {attempt_num + 1} failed: {e}. Retrying with temp={current_temp + args.retry_temp_increase:.2f}...",
                            file=sys.stderr,
                        )
                        if args.delay > 0:
                            time.sleep(args.delay)
                    else:
                        # Mark final failure in span
                        if TRACING_AVAILABLE:
                            set_span_attributes(
                                {
                                    "judge.attempts": attempt_num + 1,
                                    "judge.success": False,
                                    "judge.error": str(last_error),
                                }
                            )

        # Process the final result
        if decision is not None:
            # Convert A/B winner to canonical model outcome for downstream ELO.
            winner = decision.get("winner")
            if winner == "tie":
                outcome = "tie"
                winner_model = None
            else:
                winner_model = a_model if winner == "A" else b_model
                outcome = f"{winner_model}_win"

            result_payload.update(
                {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "attempts": attempts,
                    "decision": decision,
                    "outcome": outcome,
                    "winner_model": winner_model,
                }
            )
            atomic_write_json(out_path, result_payload)

            # Add final result to tracing span
            if TRACING_AVAILABLE:
                set_span_attributes(
                    {
                        "result.outcome": outcome,
                        "result.winner_model": winner_model or "tie",
                        "result.confidence": decision.get("confidence", 0.0),
                        "result.total_attempts": len(attempts),
                        "result.status": "completed",
                    }
                )
                # Add scores if available
                if "scores" in decision:
                    scores = decision["scores"]
                    for category, details in scores.items():
                        if isinstance(details, dict):
                            set_span_attributes(
                                {
                                    f"result.scores.{category}.A": details.get("A", 0),
                                    f"result.scores.{category}.B": details.get("B", 0),
                                }
                            )

            with counter_lock:
                completed += 1
                processed += 1
            retry_info = (
                f" (after {len(attempts)} attempts)" if len(attempts) > 1 else ""
            )
            return (
                "completed",
                f"[{processed}/{total_work}] {sample}: {canonical_1} vs {canonical_2} -> {outcome}{retry_info}",
            )

        else:
            # All retries failed
            result_payload.update(
                {
                    "status": "error",
                    "completed_at": datetime.now().isoformat(),
                    "attempts": attempts,
                    "error": f"All {len(attempts)} attempts failed. Last error: {last_error}",
                }
            )
            atomic_write_json(out_path, result_payload)

            # Add error result to tracing span
            if TRACING_AVAILABLE:
                set_span_attributes(
                    {
                        "result.outcome": "error",
                        "result.total_attempts": len(attempts),
                        "result.status": "error",
                        "result.error": str(last_error),
                    }
                )

            with counter_lock:
                failed += 1
                processed += 1
            return (
                "failed",
                f"Error judging {sample}: {canonical_1} vs {canonical_2} (all {len(attempts)} attempts failed): {last_error}",
            )

    # Process comparisons (sequential or parallel)
    if args.concurrent_requests > 1:
        # Parallel processing
        print(
            f"Processing {len(comparison_tasks)} comparisons with {args.concurrent_requests} concurrent requests..."
        )
        with ThreadPoolExecutor(max_workers=args.concurrent_requests) as executor:
            future_to_task = {
                executor.submit(process_comparison, task): task
                for task in comparison_tasks
            }
            for future in as_completed(future_to_task):
                try:
                    status, message = future.result()
                    if status == "completed":
                        print(message)
                    else:
                        print(message, file=sys.stderr)
                except Exception as e:
                    with counter_lock:
                        failed += 1
                        processed += 1
                    print(f"Unexpected error in worker thread: {e}", file=sys.stderr)

                # Apply delay between completions if specified
                if args.delay > 0:
                    time.sleep(args.delay)
    else:
        # Sequential processing
        print(f"Processing {len(comparison_tasks)} comparisons sequentially...")
        for task in comparison_tasks:
            try:
                status, message = process_comparison(task)
                if status == "completed":
                    print(message)
                else:
                    print(message, file=sys.stderr)
            except Exception as e:
                with counter_lock:
                    failed += 1
                    processed += 1
                print(f"Unexpected error processing comparison: {e}", file=sys.stderr)

            # Apply delay between requests if specified
            if args.delay > 0:
                time.sleep(args.delay)

    summary = {
        "schema_version": 1,
        "timestamp": datetime.now().isoformat(),
        "started_at": started_at,
        "translations_dir": str(translations_dir),
        "samples_dir": str(samples_dir),
        "output_dir": str(output_dir),
        "judge_base_url": args.judge_base_url,
        "judge_model": args.judge_model,
        "models": {
            "discovered": discovered_models,
            "active": active_models,
        },
        "filters": {
            "focus_model": focus_model,
            "focus_mode": args.focus_mode if focus_model else None,
        },
        "schedule": {
            "type": schedule_type,
            "models": active_models,
            "rounds": len(rounds),
            "matches_per_round": len(rounds[0]),
            "round_offset": round_offset,
        },
        "planned": {
            "scheduled_total": scheduled_total,
            "scheduled_available": scheduled_available,
            "scheduled_missing_translation": scheduled_missing_translation,
        },
        "completed": completed,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "skipped_missing_translation": skipped_missing,
    }
    atomic_write_json(output_dir / "comparison_summary.json", summary)
    print(
        "Done. "
        f"completed={completed}, failed={failed}, skipped_existing={skipped_existing}, skipped_missing={skipped_missing}. "
        f"Summary: {output_dir / 'comparison_summary.json'}"
    )

    # Shutdown tracing if enabled
    if args.tracing and TRACING_AVAILABLE:
        shutdown_tracing()


if __name__ == "__main__":
    main()
