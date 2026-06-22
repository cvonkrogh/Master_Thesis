#!/usr/bin/env python3
"""Rebuild Module 04 aggregate CSVs from pipeline_50decks.log + peer_bmc_cache.csv.

Use when peers_ranked / similar_top5 / vc_diligence_summary were overwritten but
the original full pipeline log and peer cache still exist.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

_M04_PATH = _MODULES_DIR / "pipeline" / "04_find_similar_startups.py"
_spec = importlib.util.spec_from_file_location("m04_pipeline", _M04_PATH)
_m04 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_m04)

QUERY_LOG_FIELDNAMES = _m04.QUERY_LOG_FIELDNAMES
RANKED_FIELDNAMES = _m04.RANKED_FIELDNAMES
TOP5_FIELDNAMES = _m04.TOP5_FIELDNAMES
VC_SUMMARY_FIELDNAMES = _m04.VC_SUMMARY_FIELDNAMES
_build_vc_summary_row = _m04._build_vc_summary_row
from support.csv_bmc import bmc_by_deck_from_csv, startup_name_for_deck  # noqa: E402
from support.paths import (  # noqa: E402
    DEFAULT_MODULE_04_DIR,
    DEFAULT_PEER_BMC_CACHE_CSV,
    resolve_enriched_bmc,
)
from support.peer_bmc_cache import get_cached_peer_bmc, load_peer_bmc_cache  # noqa: E402
from support.profile_text import bmc_fields_filled_count, bmc_row_to_profile_text  # noqa: E402
from support.schema import BMC_FIELDS  # noqa: E402
from support.similar_score import score_peer  # noqa: E402
from support.similar_search import brand_from_domain, target_exclude_domains  # noqa: E402
from support.websites import load_websites_csv, lookup_website_info  # noqa: E402

PEER_RE = re.compile(
    r"^\[04\] (.+?): peer (\d+)/(\d+) (.+?) \((https?://[^\)]+)\) prefilter=([\d.]+)"
)
TOP_RE = re.compile(
    r"^\[04\] (.+?): top match (.+?) \((https?://[^\)]+)\) "
    r"rank_score=([\d.]+) core_embed=([\d.]+)"
)
CAND_RE = re.compile(r"^\[04\] (.+?): (\d+) candidate URL\(s\)")
RANKED_RE = re.compile(r"^\[04\] (.+?): (\d+) ranked peer\(s\)")
DECK_RE = re.compile(r"^\[04\] ===== (.+?) =====$")
ANCHOR_RE = re.compile(r"^\[04\] (.+?): anchor '(.+?)', queries")
QUERY_RE = re.compile(r"^\s+- \[(bmc|competitor|vc)\] (.+)$")
VC_UNIQUE_RE = re.compile(
    r"^\[04\] (.+?): VC signal — unique \((.+)\)$"
)
VC_STD_RE = re.compile(
    r"^\[04\] (.+?): VC signal — (\w+) \((\d+) strong in top-5\)$"
)


@dataclass
class PendingPeer:
    name: str
    url: str
    prefilter: float
    from_cache: bool = False
    skipped: bool = False


@dataclass
class DeckRun:
    deck_id: str
    anchor: str = ""
    candidates: int = 0
    queries: list[tuple[int, str, str]] = field(default_factory=list)
    peers: list[PendingPeer] = field(default_factory=list)
    ranked_count: int = 0
    no_peers: bool = False
    vc_label: str = ""
    vc_note: str = ""


def _parse_log(path: Path) -> list[DeckRun]:
    runs: list[DeckRun] = []
    current: DeckRun | None = None
    query_rank = 0

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := DECK_RE.match(raw):
            current = DeckRun(deck_id=m.group(1))
            runs.append(current)
            query_rank = 0
            continue
        if current is None:
            continue

        if m := ANCHOR_RE.match(raw):
            current.anchor = m.group(2)
            continue
        if m := QUERY_RE.match(raw):
            query_rank += 1
            kind, q = m.group(1), m.group(2)
            current.queries.append((query_rank, kind, q))
            continue
        if m := CAND_RE.match(raw):
            current.candidates = int(m.group(2))
            continue
        if m := PEER_RE.match(raw):
            current.peers.append(
                PendingPeer(
                    name=m.group(4).strip(),
                    url=m.group(5).strip(),
                    prefilter=float(m.group(6)),
                )
            )
            continue
        if raw.strip() == "[04]   cache hit (" or "cache hit (" in raw:
            if current.peers:
                current.peers[-1].from_cache = True
            continue
        if "cache hit (" in raw and current.peers:
            current.peers[-1].from_cache = True
            continue
        if "skip:" in raw and current.peers:
            current.peers[-1].skipped = True
            continue
        if "no peers scored." in raw:
            current.no_peers = True
            continue
        if m := RANKED_RE.match(raw):
            current.ranked_count = int(m.group(2))
            continue
        if m := VC_STD_RE.match(raw):
            current.vc_label = m.group(2)
            current.vc_note = f"{m.group(3)} strong in top-5"
            continue
        if m := VC_UNIQUE_RE.match(raw):
            current.vc_label = "unique"
            current.vc_note = m.group(2)
            continue

    return runs


def _peer_bmc_from_cache(cache_row: dict[str, str]) -> dict[str, str]:
    return {f: (cache_row.get(f) or "").strip() for f in BMC_FIELDS}


def _target_labels(deck_id: str, websites_by: dict) -> list[str]:
    info = lookup_website_info(deck_id, websites_by)
    label = (info.get("startup_name") or startup_name_for_deck(deck_id)).strip()
    domain = (
        info.get("discovered_website") or info.get("website_url") or ""
    ).strip()
    return [deck_id, label, brand_from_domain(domain)]


def recover(
    *,
    log_path: Path,
    out_dir: Path,
    enriched_path: Path,
    cache_path: Path,
    top_k: int = 5,
    min_peer_fields: int = 2,
) -> None:
    by_deck = bmc_by_deck_from_csv(enriched_path)
    websites_by = {}
    websites_csv = enriched_path.parent / "websites.csv"
    if websites_csv.exists():
        websites_by = load_websites_csv(websites_csv)
    peer_cache = load_peer_bmc_cache(cache_path)

    runs = _parse_log(log_path)
    all_peers: list[dict[str, object]] = []
    all_top5: list[dict[str, object]] = []
    vc_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []

    for run in runs:
        deck_id = run.deck_id
        if deck_id not in by_deck:
            print(f"[recover] skip unknown deck {deck_id!r}", file=sys.stderr)
            continue

        target_row = by_deck[deck_id]
        target_profile = bmc_row_to_profile_text(target_row, deck_id)
        labels = _target_labels(deck_id, websites_by)
        exclude = target_exclude_domains(
            deck_id,
            run.anchor if run.anchor.startswith("http") else "",
        )

        scored: list[dict[str, object]] = []
        for peer in run.peers:
            if peer.skipped:
                continue
            cache_bmc = get_cached_peer_bmc(
                peer.url, peer_cache, min_fields=min_peer_fields
            )
            if not cache_bmc:
                print(
                    f"[recover] {deck_id}: missing cache for {peer.url}",
                    file=sys.stderr,
                )
                continue

            peer_profile = bmc_row_to_profile_text(cache_bmc, peer.name)
            scores = score_peer(
                target_profile,
                target_row,
                peer_profile,
                cache_bmc,
                peer_url=peer.url,
                target_labels=labels,
            )
            row: dict[str, object] = {
                "target_deck_id": deck_id,
                "peer_name": peer.name,
                "url": peer.url,
                "discovery_source": "",
                "snippet_prefilter_score": peer.prefilter,
                "from_cache": "yes" if peer.from_cache else "no",
                "bmc_fields_filled": bmc_fields_filled_count(cache_bmc),
                **scores,
                **cache_bmc,
            }
            scored.append(row)

        scored.sort(key=lambda r: float(r["rank_score"]), reverse=True)
        for rank, row in enumerate(scored, 1):
            row["rank"] = rank

        if scored:
            all_peers.extend(scored)
            all_top5.extend(scored[:top_k])
            vc_rows.append(
                _build_vc_summary_row(
                    deck_id,
                    candidates_discovered=run.candidates,
                    scored_rows=scored,
                    top_k=top_k,
                )
            )
        else:
            vc_rows.append(
                {
                    "target_deck_id": deck_id,
                    "candidates_discovered": run.candidates,
                    "peers_scored": 0,
                    "top5_strong_peers": 0,
                    "top5_mean_rank_score": 0.0,
                    "uniqueness_label": run.vc_label or "unique",
                    "vc_note": run.vc_note
                    or "No peers recovered from log/cache.",
                }
            )

        for qrank, kind, q in run.queries:
            query_rows.append(
                {
                    "target_deck_id": deck_id,
                    "anchor": run.anchor,
                    "query_rank": qrank,
                    "query": f"[{kind}] {q}",
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(name: str, fields: list[str], rows: list[dict[str, object]]) -> Path:
        path = out_dir / name
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fields})
        return path

    p1 = write_csv("peers_ranked.csv", RANKED_FIELDNAMES, all_peers)
    p2 = write_csv("similar_top5_all_decks.csv", TOP5_FIELDNAMES, all_top5)
    p3 = write_csv("vc_diligence_summary.csv", VC_SUMMARY_FIELDNAMES, vc_rows)
    p4 = write_csv("search_queries.csv", QUERY_LOG_FIELDNAMES, query_rows)

    print(f"[recover] peers_ranked.csv -> {len(all_peers)} rows ({p1})")
    print(f"[recover] similar_top5_all_decks.csv -> {len(all_top5)} rows ({p2})")
    print(f"[recover] vc_diligence_summary.csv -> {len(vc_rows)} rows ({p3})")
    print(f"[recover] search_queries.csv -> {len(query_rows)} rows ({p4})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover M04 CSVs from pipeline log.")
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/pipeline_50decks.log"),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MODULE_04_DIR)
    parser.add_argument("--enriched-in", type=Path, default=None)
    parser.add_argument("--peer-cache", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)

    enriched = args.enriched_in or resolve_enriched_bmc()
    cache = args.peer_cache or (args.out_dir / DEFAULT_PEER_BMC_CACHE_CSV.name)
    if not args.log.exists():
        print(f"[recover] Missing log: {args.log}", file=sys.stderr)
        return 1
    if not cache.exists():
        print(f"[recover] Missing cache: {cache}", file=sys.stderr)
        return 1

    recover(
        log_path=args.log,
        out_dir=args.out_dir,
        enriched_path=enriched,
        cache_path=cache,
        top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
