#!/usr/bin/env python3
"""
Local results dashboard for llm_translate_benchmark.

Reads `comparison_results/*.json`, computes Elo on startup (and on reload),
and serves an interactive HTML dashboard with:
  - Elo leaderboard + histogram
  - Pairwise heatmap (Elo-expected and observed win-rate)
  - Per-chapter drill-down showing all pairs and judge outputs

No external dependencies (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parent
STATIC_DIR = REPO_ROOT / "dashboard"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


@dataclass(frozen=True)
class ComparisonRecord:
    sample: str
    model_1: str
    model_2: str
    winner_model: Optional[str]  # None for tie
    judge_model: str
    decision: Optional[Dict[str, Any]]
    presentation: Optional[Dict[str, str]]
    translation_files: Optional[Dict[str, str]]
    source_file: Optional[str]
    completed_at: Optional[str]
    comparison_path: str  # relative path, for display/debug


def _infer_winner_model(
    *,
    model_1: str,
    model_2: str,
    winner_model: Any,
    decision: Any,
    presentation: Any,
) -> Optional[str]:
    if isinstance(winner_model, str) and winner_model in {model_1, model_2}:
        return winner_model
    if winner_model is None:
        return None

    if not isinstance(decision, dict):
        return None

    winner = decision.get("winner")
    if winner == "tie":
        return None

    if winner in {"A", "B"} and isinstance(presentation, dict):
        mapped = presentation.get(winner)
        if isinstance(mapped, str) and mapped in {model_1, model_2}:
            return mapped

    return None


def parse_comparison_file(path: Path) -> Optional[ComparisonRecord]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if payload.get("status") != "completed":
        return None

    sample = payload.get("sample")
    models = payload.get("models") or {}
    model_1 = models.get("model_1") or payload.get("model_1") or payload.get("model_a")
    model_2 = models.get("model_2") or payload.get("model_2") or payload.get("model_b")
    if not isinstance(sample, str) or not isinstance(model_1, str) or not isinstance(model_2, str):
        return None

    decision = payload.get("decision")
    decision = decision if isinstance(decision, dict) else None

    presentation = payload.get("presentation")
    presentation = presentation if isinstance(presentation, dict) else None

    judge = payload.get("judge") or {}
    judge_model = (
        (judge.get("model") if isinstance(judge, dict) else None)
        or payload.get("judge_model")
        or "unknown"
    )
    if not isinstance(judge_model, str) or not judge_model.strip():
        judge_model = "unknown"

    raw_winner_model = payload.get("winner_model")
    winner_model = _infer_winner_model(
        model_1=model_1,
        model_2=model_2,
        winner_model=raw_winner_model,
        decision=decision,
        presentation=presentation,
    )
    if winner_model is None and raw_winner_model is not None:
        # If the file claims a non-tie winner but we can't validate/resolve it, skip this record.
        # (Avoid silently turning bad data into a tie.)
        return None

    translation_files = payload.get("translation_files")
    translation_files = translation_files if isinstance(translation_files, dict) else None

    source_file = payload.get("source_file")
    source_file = source_file if isinstance(source_file, str) else None

    completed_at = payload.get("completed_at")
    completed_at = completed_at if isinstance(completed_at, str) else None

    try:
        comparison_path = str(path.relative_to(REPO_ROOT))
    except Exception:
        comparison_path = str(path)

    return ComparisonRecord(
        sample=sample,
        model_1=model_1,
        model_2=model_2,
        winner_model=winner_model,
        judge_model=judge_model,
        decision=decision,
        presentation={k: str(v) for k, v in (presentation or {}).items() if k in {"A", "B"}}
        if presentation
        else None,
        translation_files={k: str(v) for k, v in (translation_files or {}).items() if k in {"A", "B"}}
        if translation_files
        else None,
        source_file=source_file,
        completed_at=completed_at,
        comparison_path=comparison_path,
    )


def load_comparisons(comparison_dir: Path) -> List[ComparisonRecord]:
    if not comparison_dir.exists():
        raise FileNotFoundError(f"Comparison directory not found: {comparison_dir}")

    records: List[ComparisonRecord] = []
    for path in sorted(comparison_dir.glob("*.json")):
        if path.name in {"comparison_summary.json"}:
            continue
        rec = parse_comparison_file(path)
        if rec:
            records.append(rec)
    return records


def compute_elo(
    matches: Iterable[ComparisonRecord],
    *,
    initial_rating: float,
    k_factor: float,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, int]]]:
    ratings: Dict[str, float] = {}
    stats: Dict[str, Dict[str, int]] = {}

    def ensure(model: str) -> None:
        if model not in ratings:
            ratings[model] = float(initial_rating)
            stats[model] = {"wins": 0, "losses": 0, "ties": 0, "games": 0}

    for match in matches:
        ensure(match.model_1)
        ensure(match.model_2)

        rating_1 = ratings[match.model_1]
        rating_2 = ratings[match.model_2]

        exp_1 = expected_score(rating_1, rating_2)
        exp_2 = 1.0 - exp_1

        if match.winner_model is None:
            score_1 = 0.5
            score_2 = 0.5
            stats[match.model_1]["ties"] += 1
            stats[match.model_2]["ties"] += 1
        elif match.winner_model == match.model_1:
            score_1 = 1.0
            score_2 = 0.0
            stats[match.model_1]["wins"] += 1
            stats[match.model_2]["losses"] += 1
        elif match.winner_model == match.model_2:
            score_1 = 0.0
            score_2 = 1.0
            stats[match.model_1]["losses"] += 1
            stats[match.model_2]["wins"] += 1
        else:
            # Unknown winner (shouldn't happen after parsing); skip.
            continue

        stats[match.model_1]["games"] += 1
        stats[match.model_2]["games"] += 1

        ratings[match.model_1] = rating_1 + k_factor * (score_1 - exp_1)
        ratings[match.model_2] = rating_2 + k_factor * (score_2 - exp_2)

    return ratings, stats


def _histogram(values: List[float], *, bins: int) -> Dict[str, Any]:
    if not values:
        return {"bins": [], "counts": [], "min": None, "max": None}

    v_min = min(values)
    v_max = max(values)
    if math.isclose(v_min, v_max):
        v_min -= 1.0
        v_max += 1.0

    bins = max(5, min(50, bins))
    width = (v_max - v_min) / bins
    edges = [v_min + i * width for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for v in values:
        idx = int((v - v_min) / width)
        if idx == bins:
            idx = bins - 1
        counts[idx] += 1
    return {
        "bins": [round(x, 2) for x in edges],
        "counts": counts,
        "min": round(v_min, 2),
        "max": round(v_max, 2),
    }


def _pairwise_counts(
    matches: Iterable[ComparisonRecord],
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    Returns stats keyed by sorted (a,b):
      {"wins_a": int, "wins_b": int, "ties": int, "games": int}
    where a is key[0] and b is key[1].
    """
    out: Dict[Tuple[str, str], Dict[str, int]] = {}
    for m in matches:
        a, b = sorted([m.model_1, m.model_2])
        key = (a, b)
        if key not in out:
            out[key] = {"wins_a": 0, "wins_b": 0, "ties": 0, "games": 0}

        if m.winner_model is None:
            out[key]["ties"] += 1
        elif m.winner_model == a:
            out[key]["wins_a"] += 1
        elif m.winner_model == b:
            out[key]["wins_b"] += 1
        else:
            continue
        out[key]["games"] += 1
    return out


def build_summary_payload(
    matches: List[ComparisonRecord],
    *,
    judge: str,
    initial_rating: float,
    k_factor: float,
) -> Dict[str, Any]:
    ratings, stats = compute_elo(matches, initial_rating=initial_rating, k_factor=k_factor)
    rows = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)

    models_ordered = [m for (m, _) in rows]
    n = len(models_ordered)
    rating_map = {m: float(ratings[m]) for m in models_ordered}

    expected: List[List[Optional[float]]] = [[None for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            expected[i][j] = round(expected_score(rating_map[models_ordered[i]], rating_map[models_ordered[j]]), 2)

    pair_counts = _pairwise_counts(matches)
    observed: List[List[Optional[float]]] = [[None for _ in range(n)] for _ in range(n)]
    wins: List[List[Optional[int]]] = [[None for _ in range(n)] for _ in range(n)]
    ties: List[List[Optional[int]]] = [[None for _ in range(n)] for _ in range(n)]
    games: List[List[Optional[int]]] = [[None for _ in range(n)] for _ in range(n)]

    idx_of = {m: i for i, m in enumerate(models_ordered)}
    for (a, b), c in pair_counts.items():
        ia = idx_of.get(a)
        ib = idx_of.get(b)
        if ia is None or ib is None:
            continue
        ga = c["games"]
        if ga <= 0:
            continue

        wins_a = c["wins_a"]
        wins_b = c["wins_b"]
        ties_ab = c["ties"]

        # a row vs b col
        observed_a = (wins_a + 0.5 * ties_ab) / ga
        observed_b = (wins_b + 0.5 * ties_ab) / ga

        observed[ia][ib] = round(observed_a, 2)
        observed[ib][ia] = round(observed_b, 2)
        wins[ia][ib] = wins_a
        wins[ib][ia] = wins_b
        ties[ia][ib] = ties_ab
        ties[ib][ia] = ties_ab
        games[ia][ib] = ga
        games[ib][ia] = ga

    values = [float(r) for (_, r) in rows]
    histogram = _histogram(values, bins=max(10, min(24, n * 2)))

    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "judge": judge,
        "initial_rating": initial_rating,
        "k_factor": k_factor,
        "num_matches": len(matches),
        "num_models": n,
        "ratings": [
            {
                "model": model,
                "elo": round(rating, 2),
                "wins": stats.get(model, {}).get("wins", 0),
                "losses": stats.get(model, {}).get("losses", 0),
                "ties": stats.get(model, {}).get("ties", 0),
                "games": stats.get(model, {}).get("games", 0),
            }
            for model, rating in rows
        ],
        "histogram": histogram,
        "pairwise": {
            "models": models_ordered,
            "expected": expected,
            "observed": observed,
            "wins": wins,
            "ties": ties,
            "games": games,
        },
    }


class ComparisonStore:
    def __init__(
        self,
        *,
        comparison_dir: Path,
        samples_dir: Path,
        translations_dir: Path,
        initial_rating: float,
        k_factor: float,
    ) -> None:
        self._lock = threading.Lock()
        self._comparison_dir = comparison_dir
        self._samples_dir = samples_dir
        self._translations_dir = translations_dir
        self._initial_rating = initial_rating
        self._k_factor = k_factor

        self._loaded_at: Optional[str] = None
        self._comparisons: List[ComparisonRecord] = []

    @property
    def loaded_at(self) -> Optional[str]:
        with self._lock:
            return self._loaded_at

    def reload(self) -> None:
        with self._lock:
            self._comparisons = load_comparisons(self._comparison_dir)
            self._loaded_at = _now_iso()

    def judges_index(self) -> List[Dict[str, Any]]:
        with self._lock:
            by_judge: Dict[str, int] = {}
            by_judge_samples: Dict[str, set[str]] = {}
            for r in self._comparisons:
                by_judge[r.judge_model] = by_judge.get(r.judge_model, 0) + 1
                by_judge_samples.setdefault(r.judge_model, set()).add(r.sample)

            judges = sorted(by_judge.keys())
            return [
                {
                    "model": j,
                    "matches": by_judge[j],
                    "chapters": len(by_judge_samples.get(j, set())),
                }
                for j in judges
            ]

    def chapters_index(self) -> List[Dict[str, Any]]:
        with self._lock:
            by_sample: Dict[str, Dict[str, int]] = {}
            for r in self._comparisons:
                by_sample.setdefault(r.sample, {})
                by_sample[r.sample][r.judge_model] = by_sample[r.sample].get(r.judge_model, 0) + 1

            samples = sorted(by_sample.keys())
            out: List[Dict[str, Any]] = []
            for s in samples:
                per = by_sample[s]
                out.append(
                    {
                        "sample": s,
                        "total_matches": sum(per.values()),
                        "by_judge": per,
                    }
                )
            return out

    def filter_matches(self, judge: str) -> List[ComparisonRecord]:
        with self._lock:
            if judge == "combined":
                return list(self._comparisons)
            return [r for r in self._comparisons if r.judge_model == judge]

    def summary(self, judge: str) -> Dict[str, Any]:
        matches = self.filter_matches(judge)
        return build_summary_payload(
            matches,
            judge=judge,
            initial_rating=self._initial_rating,
            k_factor=self._k_factor,
        )

    def chapter(self, sample: str, judge: str) -> Dict[str, Any]:
        with self._lock:
            if judge == "combined":
                items = [r for r in self._comparisons if r.sample == sample]
            else:
                items = [r for r in self._comparisons if r.sample == sample and r.judge_model == judge]

        items_sorted = sorted(items, key=lambda r: (r.model_1, r.model_2, r.completed_at or "", r.comparison_path))

        def _decision_brief(decision: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not decision:
                return None
            # Keep payload reasonably small; preserve full decision but guard types.
            out: Dict[str, Any] = {}
            for k in ("winner", "confidence", "scores", "key_differences", "final_summary"):
                if k in decision:
                    out[k] = decision[k]
            return out

        return {
            "schema_version": 1,
            "generated_at": _now_iso(),
            "judge": judge,
            "sample": sample,
            "num_matches": len(items_sorted),
            "matches": [
                {
                    "comparison_path": r.comparison_path,
                    "model_1": r.model_1,
                    "model_2": r.model_2,
                    "winner_model": r.winner_model,
                    "judge_model": r.judge_model,
                    "completed_at": r.completed_at,
                    "presentation": r.presentation,
                    "translation_files": r.translation_files,
                    "source_file": r.source_file,
                    "decision": _decision_brief(r.decision),
                }
                for r in items_sorted
            ],
        }

    def read_text_file(self, raw_path: str, *, max_chars: int) -> Dict[str, Any]:
        max_chars = max(1_000, min(200_000, max_chars))
        requested = raw_path.strip()
        if not requested:
            raise FileNotFoundError("Missing path")

        # Allow both repo-relative and absolute paths, but only within allowed roots.
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()

        allowed_roots = [self._samples_dir.resolve(), self._translations_dir.resolve(), self._comparison_dir.resolve()]
        if not any(str(candidate).startswith(str(root) + os.sep) or candidate == root for root in allowed_roots):
            raise PermissionError("Path not allowed")

        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError("File not found")

        text = candidate.read_text(encoding="utf-8", errors="replace")
        truncated = False
        if len(text) > max_chars:
            text = text[:max_chars]
            truncated = True

        try:
            rel = str(candidate.relative_to(REPO_ROOT))
        except Exception:
            rel = str(candidate)

        return {"path": rel, "truncated": truncated, "text": text}


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "llm_translate_benchmark_dashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            if path.startswith("/api/"):
                self._handle_api(path, parsed.query)
                return
            self._handle_static(path)
        except PermissionError as e:
            self._send_json({"error": str(e)}, status=HTTPStatus.FORBIDDEN)
        except FileNotFoundError as e:
            self._send_json({"error": str(e)}, status=HTTPStatus.NOT_FOUND)
        except Exception as e:
            self._send_json({"error": f"Internal error: {e}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        # Keep logs concise.
        msg = fmt % args
        print(f"[{_now_iso()}] {self.client_address[0]} {self.command} {self.path} -> {msg}")

    @property
    def _store(self) -> ComparisonStore:
        return self.server.store  # type: ignore[attr-defined]

    def _handle_api(self, path: str, query: str) -> None:
        qs = parse_qs(query)

        if path == "/api/index":
            payload = {
                "schema_version": 1,
                "generated_at": _now_iso(),
                "loaded_at": self._store.loaded_at,
                "judges": self._store.judges_index(),
                "chapters": self._store.chapters_index(),
            }
            self._send_json(payload)
            return

        if path == "/api/reload":
            self._store.reload()
            self._send_json({"ok": True, "reloaded_at": self._store.loaded_at})
            return

        if path == "/api/summary":
            judge = (qs.get("judge", ["combined"])[0] or "combined").strip()
            payload = self._store.summary(judge)
            self._send_json(payload)
            return

        if path == "/api/chapter":
            sample = (qs.get("sample", [""])[0] or "").strip()
            if not sample:
                self._send_json({"error": "Missing sample"}, status=HTTPStatus.BAD_REQUEST)
                return
            judge = (qs.get("judge", ["combined"])[0] or "combined").strip()
            payload = self._store.chapter(sample, judge)
            self._send_json(payload)
            return

        if path == "/api/text":
            raw_path = (qs.get("path", [""])[0] or "").strip()
            max_chars = _safe_int((qs.get("max_chars", ["50000"])[0] or "50000"), 50000)
            payload = self._store.read_text_file(raw_path, max_chars=max_chars)
            self._send_json(payload)
            return

        self._send_json({"error": f"Unknown endpoint: {path}"}, status=HTTPStatus.NOT_FOUND)

    def _handle_static(self, path: str) -> None:
        # Default route.
        if path in {"", "/"}:
            path = "/index.html"

        rel = path.lstrip("/")
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
            raise PermissionError("Invalid path")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError("Not found")

        content = target.read_bytes()
        ctype, _ = mimetypes.guess_type(str(target))
        if not ctype:
            ctype = "application/octet-stream"

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8" if ctype.startswith("text/") else ctype)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: Dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = (json.dumps(payload, ensure_ascii=False, indent=None, separators=(",", ":")) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, server_address: Tuple[str, int], handler_class: type[BaseHTTPRequestHandler], store: ComparisonStore):
        super().__init__(server_address, handler_class)
        self.store = store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve a local dashboard for comparison_results (Elo + drilldowns).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8001, help="Bind port")
    parser.add_argument("--comparison-dir", default="comparison_results", help="Directory containing comparison JSONs")
    parser.add_argument("--samples-dir", default="samples", help="Directory containing source samples")
    parser.add_argument("--translations-dir", default="translated_results", help="Directory containing translated outputs")
    parser.add_argument("--initial-rating", type=float, default=1500.0, help="Initial Elo rating for each model")
    parser.add_argument("--k-factor", type=float, default=32.0, help="Elo K-factor")

    args = parser.parse_args()

    if not STATIC_DIR.exists():
        raise SystemExit(f"Missing dashboard assets directory: {STATIC_DIR}")

    store = ComparisonStore(
        comparison_dir=Path(args.comparison_dir),
        samples_dir=Path(args.samples_dir),
        translations_dir=Path(args.translations_dir),
        initial_rating=args.initial_rating,
        k_factor=args.k_factor,
    )
    store.reload()

    server = DashboardServer((args.host, args.port), DashboardHandler, store)
    url = f"http://{args.host}:{args.port}/"

    print("==========================================")
    print("LLM Translate Benchmark Dashboard")
    print("==========================================")
    print(f"Loaded comparisons: {len(store.filter_matches('combined'))}")
    print(f"Serving: {url}")
    print("Press Ctrl+C to stop.")
    print("==========================================")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
