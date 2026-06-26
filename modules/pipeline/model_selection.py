#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from eval.evaluate_bmc import compare, summarize
from support.csv_bmc import canonical_deck_id, load_gt_bmc_rows
from support.extract_common import (
    build_extraction_model,
    call_extract,
    deck_id_from_slides_path,
    load_slides,
)
from support.local_llm import check_ollama
from support.paths import (
    DEFAULT_GT_BMC_PD,
    DEFAULT_OUTPUT_DIR,
    resolve_module_01_slides,
)
from support.schema import BMC_FIELDS

DEFAULT_MODELS: tuple[str, ...] = (
    "llama3.1:8b",
    "mistral:7b",
    "qwen2.5:7b-instruct",
    "mistral-small",
    "llama3.2:3b",
    "qwen2.5:3b-instruct",
)

DEFAULT_DECKS: tuple[str, ...] = (
    "Aura",
    "Macro",
    "Vision",
    "Jobox",
    "Bespoken_spirits",
)

BMC_MODEL = build_extraction_model(BMC_FIELDS, model_name="BmcExtraction")
BMC_SYSTEM_EXTRA = (
    "Extract ONLY the nine Business Model Canvas building blocks:\n"
    "customer_segments, value_proposition, channels, customer_relationships, "
    "revenue_model, key_resources, key_activities, key_partners, cost_structure."
)
BMC_TASK = "Extract the nine Business Model Canvas fields from this pitch deck."

def _safe_model_dir(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")

def _warmup_model(model: str) -> None:
    """Load the model into RAM with a tiny call so the timed run measures
    inference speed, not cold-start load time."""
    import httpx

    from support.local_llm import ollama_host

    try:
        httpx.post(
            f"{ollama_host()}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=600.0,
        )
    except Exception as e:
        print(f"[model-sel] warmup {model} failed (non-fatal): {e}", file=sys.stderr)

def _resolve_decks(deck_names: list[str], slides_csv: Path) -> list[str]:
    from support.slides_store import deck_ids_from_slides_csv

    available = set(deck_ids_from_slides_csv(slides_csv))
    resolved: list[str] = []
    for name in deck_names:
        canon = canonical_deck_id(name)
        if canon not in available:
            print(
                f"[model-sel] WARNING: no slides for '{name}' "
                f"(canonical id '{canon}' not in {slides_csv}); skipping.",
                file=sys.stderr,
            )
            continue
        resolved.append(canon)
    return resolved

def _run_model_on_decks(
    model: str,
    deck_ids: list[str],
    slides_csv: Path,
    sandbox_dir: Path,
    warmup: bool = False,
) -> tuple[dict[str, dict[str, str]], list[dict]]:
    """Extract BMC for each deck with one model. Returns (pred_by_deck, latency_rows)."""
    out_dir = sandbox_dir / _safe_model_dir(model)
    out_dir.mkdir(parents=True, exist_ok=True)

    if warmup:
        print(f"[model-sel] {model} :: warming up (loading into RAM) ...", flush=True)
        _warmup_model(model)

    from support.slides_store import load_slides_for_deck

    pred_by_deck: dict[str, dict[str, str]] = {}
    latency_rows: list[dict] = []

    for canon_id in deck_ids:
        slides = load_slides_for_deck(canon_id, slides_csv)
        print(f"[model-sel] {model} :: {canon_id} ({len(slides)} slides) ...", flush=True)

        t0 = time.perf_counter()
        status = "ok"
        fields: dict[str, str] = {f: "" for f in BMC_FIELDS}
        try:
            extraction = call_extract(
                slides,
                canon_id,
                BMC_MODEL,
                BMC_TASK,
                extra_system=BMC_SYSTEM_EXTRA,
                model=model,
            )
            fields = {f: str(v) for f, v in extraction.fields.model_dump().items()}
        except Exception as e:
            status = "failed"
            print(f"[model-sel] {model} :: {canon_id}: FAILED ({e})", file=sys.stderr)
        elapsed = round(time.perf_counter() - t0, 2)

        pred_by_deck[canon_id] = fields
        filled = sum(1 for v in fields.values() if v.strip())
        latency_rows.append(
            {
                "model": model,
                "deck_id": canon_id,
                "n_slides": len(slides),
                "status": status,
                "seconds": elapsed,
                "fields_filled": filled,
            }
        )

        (out_dir / f"{canon_id}.bmc.json").write_text(
            json.dumps(
                {
                    "deck_id": canon_id,
                    "model": f"ollama:{model}",
                    "status": status,
                    "seconds": elapsed,
                    "fields": fields,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[model-sel] {model} :: {canon_id}: {status}, "
            f"{filled}/{len(BMC_FIELDS)} filled, {elapsed}s",
            flush=True,
        )

    return pred_by_deck, latency_rows

def _score_model(
    pred_by_deck: dict[str, dict[str, str]],
    gt_rows: list[dict[str, str]],
    use_embeddings: bool,
    embed_model: str,
) -> dict:
    results = compare(gt_rows, pred_by_deck)
    if use_embeddings:
        try:
            from eval.embedding_similarity import apply_embedding_similarity

            apply_embedding_similarity(results, model_name=embed_model)
        except ImportError as e:
            print(f"[model-sel] WARNING: skipping embeddings — {e}", file=sys.stderr)
    return summarize(results)

def _comparison_row(model: str, latency_rows: list[dict], summary: dict) -> dict:
    ok = [r for r in latency_rows if r["status"] == "ok"]
    failed = [r for r in latency_rows if r["status"] != "ok"]
    secs = [r["seconds"] for r in latency_rows]
    cm = summary.get("confusion_matrix_fill", {})
    content = summary.get("content_metrics", {})
    return {
        "model": model,
        "n_decks": len(latency_rows),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "total_seconds": round(sum(secs), 1),
        "mean_seconds_per_deck": round(sum(secs) / len(secs), 1) if secs else 0.0,
        "pred_fill_rate": summary.get("pred_fill_rate"),
        "fill_match_rate": summary.get("fill_agreement", {}).get("fill_match_rate"),
        "accuracy": cm.get("accuracy"),
        "precision": cm.get("precision"),
        "recall": cm.get("recall"),
        "f1": cm.get("f1"),
        "content_sim_both_filled": content.get("mean_lexical_sim_both_filled"),
        "embed_sim_both_filled": content.get("mean_embedding_sim_both_filled"),
        "mean_score_0_2": summary.get("mean_score_0_2"),
    }

COMPARISON_COLUMNS = [
    "model",
    "n_decks",
    "n_ok",
    "n_failed",
    "total_seconds",
    "mean_seconds_per_deck",
    "pred_fill_rate",
    "fill_match_rate",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "content_sim_both_filled",
    "embed_sim_both_filled",
    "mean_score_0_2",
]

def _merge_existing(
    path: Path,
    new_rows: list[dict],
    new_keys: set[str],
    key: str,
) -> list[dict]:
    """Keep prior rows whose key is not being re-written, then add new rows."""
    kept: list[dict] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get(key) not in new_keys:
                    kept.append(row)
    return kept + new_rows

def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("[model-sel] No models ran successfully.")
        return
    print("\n=== Model comparison (Module 02 BMC) ===")
    header = (
        f"{'model':<24} {'sec/deck':>9} {'total_s':>8} "
        f"{'fill':>6} {'acc':>6} {'F1':>6} {'embed':>6} {'score':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<24} "
            f"{r['mean_seconds_per_deck']:>9} "
            f"{r['total_seconds']:>8} "
            f"{(r['pred_fill_rate'] if r['pred_fill_rate'] is not None else 0):>6} "
            f"{(r['accuracy'] if r['accuracy'] is not None else 0):>6} "
            f"{(r['f1'] if r['f1'] is not None else 0):>6} "
            f"{(r['embed_sim_both_filled'] if r['embed_sim_both_filled'] is not None else 0):>6} "
            f"{(r['mean_score_0_2'] if r['mean_score_0_2'] is not None else 0):>6}"
        )
    print()

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark local Ollama models on Module 02 BMC extraction."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Ollama model tags to compare (default: 5 candidate models).",
    )
    parser.add_argument(
        "--decks",
        nargs="+",
        default=list(DEFAULT_DECKS),
        help="Deck ids to test (default: 5 stratified decks).",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT_BMC_PD,
        help="Pitch-deck BMC ground truth (default: data/gt/gt_pd_bmc_50.csv).",
    )
    parser.add_argument(
        "--sandbox",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "model_selection",
        help="Where to write raw per-model extractions.",
    )
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=Path("eval") / "model_selection",
        help="Where to write comparison CSV/JSON.",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip sentence-transformers embedding similarity (faster, lexical only).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge into existing comparison/latency files (dedupe by model) instead of overwriting.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Load each model with a throwaway call before timing (fair speed, excludes cold-start).",
    )
    parser.add_argument(
        "--embed-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model for content similarity.",
    )
    args = parser.parse_args(argv)

    slides_csv = resolve_module_01_slides()
    deck_ids = _resolve_decks(args.decks, slides_csv)
    if not deck_ids:
        print("[model-sel] No valid decks resolved. Run Module 01 first.", file=sys.stderr)
        return 1

    if not args.gt.exists():
        print(f"[model-sel] GT not found: {args.gt}", file=sys.stderr)
        return 1
    all_gt = load_gt_bmc_rows(args.gt)
    deck_id_set = set(deck_ids)
    gt_rows = [r for r in all_gt if r["deck_id"] in deck_id_set]
    missing_gt = deck_id_set - {r["deck_id"] for r in gt_rows}
    if missing_gt:
        print(
            f"[model-sel] WARNING: no GT rows for {sorted(missing_gt)}; "
            "those decks will not be scored.",
            file=sys.stderr,
        )

    print(
        f"[model-sel] {len(args.models)} model(s) x {len(deck_ids)} deck(s): "
        f"decks={deck_ids}"
    )

    comparison_rows: list[dict] = []
    all_latency_rows: list[dict] = []
    summaries: dict[str, dict] = {}

    for model in args.models:
        try:
            check_ollama(model)
        except RuntimeError as e:
            print(
                f"[model-sel] SKIP {model}: {e}",
                file=sys.stderr,
            )
            continue

        pred_by_deck, latency_rows = _run_model_on_decks(
            model, deck_ids, slides_csv, args.sandbox, warmup=args.warmup
        )
        all_latency_rows.extend(latency_rows)

        scored_gt = [r for r in gt_rows]
        summary = _score_model(
            pred_by_deck,
            scored_gt,
            use_embeddings=not args.no_embeddings,
            embed_model=args.embed_model,
        )
        summaries[model] = summary
        comparison_rows.append(_comparison_row(model, latency_rows, summary))

    if not comparison_rows:
        print(
            "[model-sel] No models were available. Pull at least one, e.g. "
            "`ollama pull qwen2.5:7b-instruct`.",
            file=sys.stderr,
        )
        return 1

    args.eval_out.mkdir(parents=True, exist_ok=True)
    comp_path = args.eval_out / "model_comparison.csv"
    lat_path = args.eval_out / "per_deck_latency.csv"

    if args.append:
        new_models = {r["model"] for r in comparison_rows}
        comparison_rows = _merge_existing(comp_path, comparison_rows, new_models, key="model")
        new_lat_models = {r["model"] for r in all_latency_rows}
        all_latency_rows = _merge_existing(lat_path, all_latency_rows, new_lat_models, key="model")

    comparison_rows.sort(
        key=lambda r: (
            -(float(r.get("mean_score_0_2") or 0)),
            float(r.get("mean_seconds_per_deck") or 0),
        )
    )

    with comp_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COMPARISON_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in comparison_rows:
            w.writerow(row)

    with lat_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["model", "deck_id", "n_slides", "status", "seconds", "fields_filled"],
        )
        w.writeheader()
        for row in all_latency_rows:
            w.writerow(row)

    summary_path = args.eval_out / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "gt_path": str(args.gt),
                "decks": [c for c, _ in decks],
                "models": list(args.models),
                "comparison": comparison_rows,
                "per_model_full_summary": summaries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _print_table(comparison_rows)
    print(f"[model-sel] comparison -> {comp_path}")
    print(f"[model-sel] latency    -> {lat_path}")
    print(f"[model-sel] summary    -> {summary_path}")
    print(f"[model-sel] raw output -> {args.sandbox}/")
    best = comparison_rows[0]
    print(
        f"[model-sel] Top by quality: {best['model']} "
        f"(score={best['mean_score_0_2']}, {best['mean_seconds_per_deck']}s/deck)"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
