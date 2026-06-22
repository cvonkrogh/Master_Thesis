#!/usr/bin/env python3
"""
Build a qualitative evaluation template for Module 04 peer rankings.

Module 04 has no ground-truth competitor list. Use this CSV to rate each top peer
with a simple rubric (fill ``relevance_code`` and optional ``notes`` by hand).

Relevance codes:
  C — Credible startup peer (same sector, comparable product/stage)
  A — Adjacent (news, directory, profile of target, big incumbent)
  W — Wrong entity (homonym, unrelated topic)
  U — Unusable (empty/nonsense BMC, fetch failure residue)

Inputs (first found):
  output/module_04/similar_top5_all_decks.csv
  or output/module_04/*_similar_ranked.csv (top 5 per deck)

Output:
  eval/module_04/peer_relevance_rubric.csv

Usage:
  python modules/eval/build_m04_rubric_template.py
  python modules/eval/build_m04_rubric_template.py --top-k 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from support.paths import DEFAULT_MODULE_04_DIR  # noqa: E402

RUBRIC_FIELDS = [
    "target_deck_id",
    "rank",
    "peer_name",
    "url",
    "rank_score",
    "rank_method",
    "core_embed_sim",
    "embed_sim",
    "tfidf_cosine",
    "combined_score",
    "discovery_source",
    "bmc_fields_filled",
    "relevance_code",
    "notes",
]


def _load_rows(module_04_dir: Path, top_k: int) -> list[dict[str, str]]:
    summary = module_04_dir / "similar_top5_all_decks.csv"
    if summary.exists():
        with summary.open(encoding="utf-8", newline="") as f:
            summary_rows = list(csv.DictReader(f))
        if summary_rows:
            return summary_rows

    ranked = module_04_dir / "peers_ranked.csv"
    if ranked.exists():
        rows: list[dict[str, str]] = []
        by_deck: dict[str, list[dict[str, str]]] = {}
        with ranked.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                did = (row.get("target_deck_id") or "").strip()
                if did:
                    by_deck.setdefault(did, []).append(row)
        for did in sorted(by_deck.keys()):
            rows.extend(by_deck[did][:top_k])
        if rows:
            return rows

    rows = []
    for path in sorted(module_04_dir.glob("*_similar_ranked.csv")):
        with path.open(encoding="utf-8", newline="") as f:
            deck_rows = list(csv.DictReader(f))
        if deck_rows:
            rows.extend(deck_rows[:top_k])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Module 04 qualitative rubric template.")
    parser.add_argument(
        "--module-04-dir",
        type=Path,
        default=DEFAULT_MODULE_04_DIR,
        help="Module 04 output directory.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Peers per deck to include.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval/module_04/peer_relevance_rubric.csv"),
        help="Output rubric CSV path.",
    )
    args = parser.parse_args(argv)

    rows = _load_rows(args.module_04_dir, args.top_k)
    if not rows:
        print(f"[m04-rubric] No ranked peers in {args.module_04_dir}. Run Module 04 first.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUBRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in RUBRIC_FIELDS}
            out.setdefault("relevance_code", "")
            out.setdefault("notes", "")
            writer.writerow(out)

    print(f"[m04-rubric] {len(rows)} row(s) -> {args.out}")
    print("[m04-rubric] Fill relevance_code: C=credible peer, A=adjacent, W=wrong, U=unusable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
