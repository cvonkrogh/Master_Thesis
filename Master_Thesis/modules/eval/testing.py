#!/usr/bin/env python3
"""
Compare the two human full-BMC ground truth sources:

  - data/gt/gt_full_bmc.csv          (thesis GT; deck + web reference)
  - data/gt/BMC_max_enriched.csv     (alternate enriched reference)

Reports per-deck and per-field text similarity for all 10 startups (aligned by
canonical deck_id / startup name, not numeric row id — the two files use
different row numbering).

Usage:
    python modules/eval/testing.py
    python modules/eval/testing.py --embeddings
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from eval.evaluate_bmc import combined_similarity, jaccard, seq_ratio
from support.csv_bmc import canonical_deck_id, load_gt_bmc_rows
from support.schema import BMC_FIELDS
from support.paths import DEFAULT_GT_DIR, DEFAULT_GT_FULL_BMC

BMC_MAX_ENRICHED = DEFAULT_GT_DIR / "BMC_max_enriched.csv"

_STARTUP_ALIASES: dict[str, str] = {
    "connectly": "Vision",
    "bespoken spirits": "Bespoken_spirits",
    "multus": "multus",
    "morty": "morty",
    "jobox": "Jobox",
    "palta": "Palta",
    "aura": "Aura",
    "sable": "Sable",
    "sharpist": "Sharpist",
    "macro": "Macro",
}

def _norm_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())

def deck_id_from_startup_name(name: str) -> str:
    key = _norm_name(name)
    if key in _STARTUP_ALIASES:
        return _STARTUP_ALIASES[key]

    token = name.strip().replace(" ", "_")
    return canonical_deck_id(token)

def _detect_delimiter(first_line: str) -> str:
    return ";" if first_line.count(";") > first_line.count(",") else ","

def load_bmc_max_enriched_rows(path: Path = BMC_MAX_ENRICHED) -> dict[str, dict[str, str]]:
    """Load BMC_max_enriched.csv keyed by canonical deck_id."""
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        first = f.readline()
        f.seek(0)
        reader = csv.DictReader(f, delimiter=_detect_delimiter(first))
        if reader.fieldnames:
            reader.fieldnames = [(n or "").strip() for n in reader.fieldnames]

        by_deck: dict[str, dict[str, str]] = {}
        for row in reader:
            startup = (row.get("startup_name") or "").strip()
            if not startup:
                continue
            deck_id = deck_id_from_startup_name(startup)
            rec = {"deck_id": deck_id, "startup_name": startup}
            for field in BMC_FIELDS:
                rec[field] = (row.get(field) or "").strip()
            by_deck[deck_id] = rec
    return by_deck

def _maybe_embed_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    if not pairs:
        return []
    try:
        from eval.embedding_similarity import batch_cosine_similarity

        gt_texts = [a for a, _ in pairs]
        pred_texts = [b for _, b in pairs]
        scores, method = batch_cosine_similarity(gt_texts, pred_texts)
        print(f"[testing] Embeddings: {method}", flush=True)
        return scores
    except Exception as e:
        print(f"[testing] Embeddings skipped ({e})", file=sys.stderr)
        return []

def compare_gt_sources(
    gt_path: Path = DEFAULT_GT_FULL_BMC,
    alt_path: Path = BMC_MAX_ENRICHED,
    use_embeddings: bool = False,
) -> int:
    gt_rows = load_gt_bmc_rows(gt_path)
    gt_by_deck = {r["deck_id"]: r for r in gt_rows}
    alt_by_deck = load_bmc_max_enriched_rows(alt_path)

    gt_decks = set(gt_by_deck)
    alt_decks = set(alt_by_deck)
    missing_in_alt = sorted(gt_decks - alt_decks)
    missing_in_gt = sorted(alt_decks - gt_decks)

    print(f"[testing] GT A: {gt_path}")
    print(f"[testing] GT B: {alt_path}")
    print(f"[testing] Decks in GT A: {len(gt_decks)} | in GT B: {len(alt_decks)}")
    if missing_in_alt:
        print(f"[testing] Missing in GT B: {missing_in_alt}")
    if missing_in_gt:
        print(f"[testing] Extra in GT B only: {missing_in_gt}")

    common = sorted(gt_decks & alt_decks)
    if len(common) != 10:
        print(f"[testing] WARNING: expected 10 aligned decks, got {len(common)}", file=sys.stderr)

    embed_pairs: list[tuple[str, str]] = []
    deck_scores: list[tuple[str, float, float, int]] = []

    print()
    print(f"{'Deck':<18} {'Field':<22} {'Fill A/B':<10} {'Jacc':>5} {'Seq':>5} {'Comb':>5}")
    print("-" * 72)

    field_combined: list[float] = []
    both_filled_combined: list[float] = []

    for deck_id in common:
        gt = gt_by_deck[deck_id]
        alt = alt_by_deck[deck_id]
        deck_sims: list[float] = []
        filled_both = 0

        for field in BMC_FIELDS:
            a = gt.get(field, "")
            b = alt.get(field, "")
            af, bf = bool(a.strip()), bool(b.strip())
            j = jaccard(a, b)
            s = seq_ratio(a, b)
            c = combined_similarity(a, b)
            field_combined.append(c)
            if af and bf:
                both_filled_combined.append(c)
                filled_both += 1
                embed_pairs.append((a, b))
            fill_tag = f"{'Y' if af else 'n'}/{'Y' if bf else 'n'}"
            print(
                f"{deck_id:<18} {field:<22} {fill_tag:<10} {j:5.2f} {s:5.2f} {c:5.2f}",
            )
            deck_sims.append(c)

        mean_deck = sum(deck_sims) / len(deck_sims) if deck_sims else 0.0
        deck_scores.append((deck_id, mean_deck, mean_deck, filled_both))

    embed_scores: list[float] = []
    if use_embeddings and embed_pairs:
        embed_scores = _maybe_embed_pairs(embed_pairs)

    print()
    print("Per-deck mean combined similarity (all 9 fields):")
    for deck_id, mean_c, _, n_both in deck_scores:
        print(f"  {deck_id:<18} {mean_c:5.2f}  (both filled: {n_both}/9)")

    overall = sum(field_combined) / len(field_combined) if field_combined else 0.0
    both_mean = sum(both_filled_combined) / len(both_filled_combined) if both_filled_combined else 0.0
    high = sum(1 for c in both_filled_combined if c >= 0.75)
    partial = sum(1 for c in both_filled_combined if 0.35 <= c < 0.75)

    print()
    print(f"Overall mean combined sim (all cells):     {overall:.3f}")
    print(f"Mean combined sim (both filled only):      {both_mean:.3f}")
    print(f"Both filled cells:                         {len(both_filled_combined)}/90")
    print(f"High match (comb ≥0.75, both filled):      {high}")
    print(f"Partial match (0.35–0.75, both filled):    {partial}")

    if embed_scores:
        embed_mean = sum(embed_scores) / len(embed_scores)
        print(f"Mean embedding sim (both filled only):   {embed_mean:.3f}")

    print()
    if both_mean >= 0.5 and len(common) == 10:
        print("[testing] Verdict: same 10 startups, broadly consistent wording (not identical copies).")
    elif len(common) == 10:
        print("[testing] Verdict: same 10 startups, but descriptions differ materially — review before merging GTs.")
    else:
        print("[testing] Verdict: deck alignment incomplete — fix startup name mapping first.")
        return 1

    return 0

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare gt_full_bmc.csv vs BMC_max_enriched.csv (10 startups).",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT_FULL_BMC,
        help="Primary full BMC GT (default: data/gt/gt_full_bmc.csv).",
    )
    parser.add_argument(
        "--alt",
        type=Path,
        default=BMC_MAX_ENRICHED,
        help="Alternate GT (default: data/gt/BMC_max_enriched.csv).",
    )
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Also compute sentence-transformer cosine on both-filled pairs.",
    )
    args = parser.parse_args(argv)
    return compare_gt_sources(args.gt, args.alt, use_embeddings=args.embeddings)

if __name__ == "__main__":
    raise SystemExit(main())
