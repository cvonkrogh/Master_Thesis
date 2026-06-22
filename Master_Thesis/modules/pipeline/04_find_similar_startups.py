#!/usr/bin/env python3
"""
Module 04 — find similar startups online and rank by BMC profile similarity.

Uses the target's enriched BMC as the query profile, discovers peer startups via
BMC-driven + competitor web search (no ground truth), extracts each peer's BMC
from public web pages (local LLM), then ranks by core-field embedding cosine
(customer_segments + value_proposition) minus host/mention/incumbent penalties.

Inputs:
    output/module_03/enriched_bmc.csv   — target BMC profile (Module 03)

Outputs:
    output/module_04/peers_ranked.csv          (all scored peers, all decks)
    output/module_04/similar_top5_all_decks.csv
    output/module_04/vc_diligence_summary.csv
    output/module_04/search_queries.csv
    output/module_04/peer_bmc_cache.csv        (reused peer BMC across decks)

Usage:
    python modules/pipeline/04_find_similar_startups.py --deck Aura
    python modules/pipeline/04_find_similar_startups.py --all
    python modules/pipeline/04_find_similar_startups.py --all --force
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from support.csv_bmc import bmc_by_deck_from_csv  # noqa: E402
from support.local_llm import check_ollama, ollama_model  # noqa: E402
from support.paths import (  # noqa: E402
    DEFAULT_MODULE_04_DIR,
    DEFAULT_PEER_BMC_CACHE_CSV,
    DEFAULT_PEERS_RANKED_CSV,
    DEFAULT_SEARCH_QUERIES_CSV,
    DEFAULT_SIMILAR_TOP5_CSV,
    DEFAULT_VC_DILIGENCE_CSV,
    resolve_enriched_bmc,
    resolve_websites_csv,
)
from support.peer_bmc_cache import (  # noqa: E402
    get_cached_peer_bmc,
    load_peer_bmc_cache,
    peer_domain_key,
    upsert_peer_bmc_cache,
    write_peer_bmc_cache,
)
from support.peer_bmc import extract_peer_bmc  # noqa: E402
from support.profile_text import bmc_fields_filled_count, bmc_row_to_profile_text  # noqa: E402
from support.schema import BMC_FIELDS  # noqa: E402
from support.similar_score import (  # noqa: E402
    PREFILTER_MIN_SCORE,
    batch_snippet_prefilter_scores,
    score_peer,
    select_candidates_for_deep_scoring,
)
from support.similar_search import (  # noqa: E402
    brand_from_domain,
    build_all_search_queries,
    collect_peer_candidates,
    target_exclude_domains,
)
from support.web_fetch import USER_AGENT, fetch_homepage_snippet, fetch_pages_with_retry  # noqa: E402
from support.websites import load_websites_csv, lookup_website_info  # noqa: E402

TAG = "04"

RANKED_FIELDNAMES = [
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
    "host_penalty",
    "mention_penalty",
    "incumbent_penalty",
    "discovery_source",
    "snippet_prefilter_score",
    "from_cache",
    "bmc_fields_filled",
    *BMC_FIELDS,
]

TOP5_FIELDNAMES = [
    "target_deck_id",
    "rank",
    "peer_name",
    "url",
    "rank_score",
    "rank_method",
    "core_embed_sim",
    "embed_sim",
    "discovery_source",
    "bmc_fields_filled",
]

VC_SUMMARY_FIELDNAMES = [
    "target_deck_id",
    "candidates_discovered",
    "peers_scored",
    "top5_strong_peers",
    "top5_mean_rank_score",
    "uniqueness_label",
    "vc_note",
]

QUERY_LOG_FIELDNAMES = ["target_deck_id", "anchor", "query_rank", "query"]


def _slug(name: str) -> str:
    import re

    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    return s.strip("_") or "peer"


def _target_website(deck_id: str, websites_by: dict[str, dict[str, str]]) -> str:
    info = lookup_website_info(deck_id, websites_by)
    return (info.get("discovered_website") or info.get("website_url") or "").strip()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _peers_by_deck_from_rows(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        did = str(row.get("target_deck_id") or "")
        if did:
            out.setdefault(did, []).append(row)
    return out


def _rows_excluding_decks(
    rows: list[dict[str, object]],
    deck_ids: set[str],
    *,
    deck_key: str = "target_deck_id",
) -> list[dict[str, object]]:
    return [r for r in rows if str(r.get(deck_key) or "") not in deck_ids]


def _startup_label(deck_id: str, websites_by_pipeline: dict[str, dict[str, str]]) -> str:
    from support.csv_bmc import startup_name_for_deck

    info = lookup_website_info(deck_id, websites_by_pipeline)
    return (info.get("startup_name") or startup_name_for_deck(deck_id)).strip()


def _uniqueness_label(
    strong_count: int,
    peers_scored: int,
    *,
    candidates_discovered: int = 0,
) -> str:
    if peers_scored == 0:
        if candidates_discovered >= 3:
            return "unique"
        return "unclear"
    if peers_scored < 2:
        return "unclear"
    if strong_count == 0:
        return "unique"
    if strong_count <= 1:
        return "high"
    if strong_count <= 3:
        return "moderate"
    return "crowded"


def _vc_note(
    label: str,
    strong_count: int,
    peers_scored: int,
    top_names: list[str],
    *,
    candidates_discovered: int = 0,
) -> str:
    uniq = _uniqueness_label(
        strong_count,
        peers_scored,
        candidates_discovered=candidates_discovered,
    )
    names = ", ".join(top_names[:3]) if top_names else "none"
    if uniq == "unique" and peers_scored == 0:
        return (
            f"Searched {candidates_discovered} candidate URL(s) but none yielded a "
            f"usable peer BMC — startup may be relatively new or differentiated; "
            f"verify manually."
        )
    if uniq == "unique":
        return (
            f"No strong startup peers in top-5 (0 above similarity threshold); "
            f"idea may be relatively new or niche. Weak matches: {names}."
        )
    if uniq == "high":
        return (
            f"Few strong startup peers in top-5 ({strong_count}); "
            f"idea may be relatively differentiated — verify manually. Top: {names}."
        )
    if uniq == "moderate":
        return (
            f"Some comparable startups found ({strong_count} strong in top-5); "
            f"competitive but not saturated. Top: {names}."
        )
    if uniq == "crowded":
        return (
            f"Several strong peers ({strong_count} in top-5); "
            f"space looks competitive — benchmark vs {names}."
        )
    return f"Insufficient peer signal ({peers_scored} scored); manual comp research needed."


def _build_vc_summary_row(
    deck_id: str,
    *,
    candidates_discovered: int,
    scored_rows: list[dict[str, object]],
    top_k: int,
) -> dict[str, object]:
    top = scored_rows[:top_k]
    strong = 0
    rank_scores: list[float] = []
    top_names: list[str] = []
    for row in top:
        rs = float(row.get("rank_score") or 0)
        hp = float(row.get("host_penalty") or 0)
        ip = float(row.get("incumbent_penalty") or 0)
        rank_scores.append(rs)
        name = str(row.get("peer_name") or "")
        if name:
            top_names.append(name)
        if rs >= 0.45 and hp < 0.08 and ip == 0.0:
            strong += 1
    mean_rs = round(sum(rank_scores) / len(rank_scores), 3) if rank_scores else 0.0
    uniq = _uniqueness_label(
        strong,
        len(scored_rows),
        candidates_discovered=candidates_discovered,
    )
    return {
        "target_deck_id": deck_id,
        "candidates_discovered": candidates_discovered,
        "peers_scored": len(scored_rows),
        "top5_strong_peers": strong,
        "top5_mean_rank_score": mean_rs,
        "uniqueness_label": uniq,
        "vc_note": _vc_note(
            uniq,
            strong,
            len(scored_rows),
            top_names,
            candidates_discovered=candidates_discovered,
        ),
    }


def _deep_score_candidate(
    deck_id: str,
    cand: object,
    *,
    target_profile: str,
    target_row: dict[str, str],
    target_labels: list[str],
    client: httpx.Client,
    model: str,
    fetch_retries: int,
    min_peer_fields: int,
    peer_cache: dict[str, dict[str, str]],
    prefilter_score: float,
) -> dict[str, object] | None:
    """Fetch (or cache) peer BMC and compute full rank score."""
    peer_name = getattr(cand, "peer_name", "")
    peer_url = getattr(cand, "url", "")
    source = getattr(cand, "source", "")

    peer_bmc = get_cached_peer_bmc(peer_url, peer_cache, min_fields=min_peer_fields)
    from_cache = peer_bmc is not None

    if not from_cache:
        pages = fetch_pages_with_retry(
            peer_url,
            client,
            tag=TAG,
            retries=fetch_retries,
            verbose=False,
            homepage_only=True,
        )
        if not pages:
            print(f"[{TAG}]   skip: no pages fetched", flush=True)
            return None

        try:
            peer_bmc = extract_peer_bmc(peer_name, peer_url, pages, model=model)
        except Exception as e:
            print(f"[{TAG}]   skip: BMC extraction failed ({e})", file=sys.stderr)
            return None

        filled = bmc_fields_filled_count(peer_bmc)
        if filled >= min_peer_fields:
            upsert_peer_bmc_cache(
                peer_cache,
                url=peer_url,
                peer_name=peer_name,
                peer_bmc=peer_bmc,
                bmc_fields_filled=filled,
                model=model,
            )
    else:
        print(f"[{TAG}]   cache hit ({peer_domain_key(peer_url)})", flush=True)

    filled = bmc_fields_filled_count(peer_bmc)
    if filled < min_peer_fields:
        print(
            f"[{TAG}]   skip: thin BMC ({filled}<{min_peer_fields} fields filled)",
            flush=True,
        )
        return None

    peer_profile = bmc_row_to_profile_text(peer_bmc, peer_name)
    scores = score_peer(
        target_profile,
        target_row,
        peer_profile,
        peer_bmc,
        peer_url=peer_url,
        target_labels=target_labels,
    )

    return {
        "target_deck_id": deck_id,
        "peer_name": peer_name,
        "url": peer_url,
        "discovery_source": source,
        "snippet_prefilter_score": prefilter_score,
        "from_cache": "yes" if from_cache else "no",
        "bmc_fields_filled": filled,
        **scores,
        **peer_bmc,
    }


def process_deck(
    deck_id: str,
    target_row: dict[str, str],
    *,
    websites_by: dict[str, dict[str, str]],
    client: httpx.Client,
    model: str,
    max_candidates: int,
    startup_name: str = "",
    fetch_retries: int = 2,
    min_peer_fields: int = 2,
    top_k: int = 5,
    query_log: Optional[list[dict[str, object]]] = None,
    peer_cache: Optional[dict[str, dict[str, str]]] = None,
    two_stage: bool = True,
    max_llm_peers: int = 5,
    min_prefilter_score: float = PREFILTER_MIN_SCORE,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Discover peers, prefilter cheaply, deep-score selected candidates."""
    cache = peer_cache if peer_cache is not None else {}
    target_profile = bmc_row_to_profile_text(target_row, deck_id)
    empty_summary = {
        "target_deck_id": deck_id,
        "candidates_discovered": 0,
        "peers_scored": 0,
        "top5_strong_peers": 0,
        "top5_mean_rank_score": 0.0,
        "uniqueness_label": "unclear",
        "vc_note": "No enriched BMC — skipped.",
    }
    if not target_profile.strip():
        print(f"[{TAG}] {deck_id}: enriched BMC empty — skip.", file=sys.stderr)
        return [], empty_summary

    target_domain = _target_website(deck_id, websites_by)
    label = startup_name or deck_id
    all_queries = build_all_search_queries(
        target_row,
        deck_id=deck_id,
        startup_name=label,
        target_domain=target_domain,
    )
    anchor = target_domain or f"name:{label}"
    print(f"[{TAG}] {deck_id}: anchor '{anchor}', queries ({len(all_queries)}):", flush=True)
    for rank, (q, kind) in enumerate(all_queries, 1):
        print(f"  - [{kind}] {q}", flush=True)
        if query_log is not None:
            query_log.append(
                {
                    "target_deck_id": deck_id,
                    "anchor": anchor,
                    "query_rank": rank,
                    "query": f"[{kind}] {q}",
                }
            )

    exclude = target_exclude_domains(deck_id, target_domain)
    target_labels = [deck_id, label, brand_from_domain(target_domain)]

    print(f"[{TAG}] {deck_id}: discovering peers (strict URLs + VC seed) ...", flush=True)
    candidates = collect_peer_candidates(
        target_row,
        client,
        exclude,
        max_candidates=max_candidates,
        deck_id=deck_id,
        startup_name=label,
        target_domain=target_domain,
    )
    print(f"[{TAG}] {deck_id}: {len(candidates)} candidate URL(s)", flush=True)
    if not candidates:
        summary_row = _build_vc_summary_row(
            deck_id,
            candidates_discovered=0,
            scored_rows=[],
            top_k=top_k,
        )
        return [], summary_row

    print(f"[{TAG}] {deck_id}: stage A — homepage snippet prefilter ...", flush=True)
    snippet_pairs: list[tuple[str, str]] = []
    for cand in candidates:
        snippet = fetch_homepage_snippet(cand.url, client) or ""
        snippet_pairs.append((cand.url, snippet))

    prefilter_scores = batch_snippet_prefilter_scores(target_row, snippet_pairs)
    ranked = sorted(
        zip(candidates, prefilter_scores, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    prefilter_by_url = {cand.url: score for cand, score in ranked}
    snippets_by_url = {url: snippet for url, snippet in snippet_pairs}

    if two_stage:
        cache_domains = set(cache.keys())
        to_score = select_candidates_for_deep_scoring(
            [(cand, score) for cand, score in ranked],
            cache_domains,
            max_llm_peers=max_llm_peers,
            min_prefilter_score=min_prefilter_score,
            snippets_by_url=snippets_by_url,
        )
        n_cache = sum(1 for c in to_score if peer_domain_key(c.url) in cache_domains)
        n_llm = len(to_score) - n_cache
        print(
            f"[{TAG}] {deck_id}: stage B — LLM on {n_llm} finalist(s) + {n_cache} cache "
            f"(of {len(candidates)} discovered; min prefilter {min_prefilter_score})",
            flush=True,
        )
    else:
        to_score = [cand for cand, _score in ranked]
        print(
            f"[{TAG}] {deck_id}: deep scoring ALL {len(to_score)} candidates "
            f"(two-stage disabled)",
            flush=True,
        )

    scored_rows: list[dict[str, object]] = []
    for i, cand in enumerate(to_score, 1):
        print(
            f"[{TAG}] {deck_id}: peer {i}/{len(to_score)} "
            f"{cand.peer_name} ({cand.url}) "
            f"prefilter={prefilter_by_url.get(cand.url, 0):.3f} ...",
            flush=True,
        )
        row = _deep_score_candidate(
            deck_id,
            cand,
            target_profile=target_profile,
            target_row=target_row,
            target_labels=target_labels,
            client=client,
            model=model,
            fetch_retries=fetch_retries,
            min_peer_fields=min_peer_fields,
            peer_cache=cache,
            prefilter_score=prefilter_by_url.get(cand.url, 0.0),
        )
        if row:
            scored_rows.append(row)

    scored_rows.sort(key=lambda r: float(r["rank_score"]), reverse=True)
    for rank, row in enumerate(scored_rows, 1):
        row["rank"] = rank

    summary_row = _build_vc_summary_row(
        deck_id,
        candidates_discovered=len(candidates),
        scored_rows=scored_rows,
        top_k=top_k,
    )

    if not scored_rows:
        print(f"[{TAG}] {deck_id}: no peers scored.", file=sys.stderr)
        print(
            f"[{TAG}] {deck_id}: VC signal — {summary_row['uniqueness_label']} "
            f"({summary_row['vc_note']})",
            flush=True,
        )
        return scored_rows, summary_row

    print(
        f"[{TAG}] {deck_id}: {len(scored_rows)} ranked peer(s)",
        flush=True,
    )
    top = scored_rows[0]
    print(
        f"[{TAG}] {deck_id}: top match {top['peer_name']} ({top['url']}) "
        f"rank_score={top['rank_score']} core_embed={top.get('core_embed_sim')}",
        flush=True,
    )
    print(
        f"[{TAG}] {deck_id}: VC signal — {summary_row['uniqueness_label']} "
        f"({summary_row['top5_strong_peers']} strong in top-{top_k})",
        flush=True,
    )
    return scored_rows, summary_row


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Module 04 — find similar startups online."
    )
    parser.add_argument("--deck", default="", help="Single target deck_id.")
    parser.add_argument("--all", action="store_true", help="Process all decks.")
    parser.add_argument("--enriched-in", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_MODULE_04_DIR,
        help="Output directory (default: output/module_04).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=15,
        help="Max peer URLs to discover per deck (default: 15).",
    )
    parser.add_argument(
        "--max-llm-peers",
        type=int,
        default=5,
        help="Finalists for LLM BMC extraction after prefilter (default: 5).",
    )
    parser.add_argument(
        "--min-prefilter",
        type=float,
        default=PREFILTER_MIN_SCORE,
        help="Min Stage-A score to qualify for LLM (default: 0.20).",
    )
    parser.add_argument(
        "--no-two-stage",
        action="store_true",
        help="Disable prefilter; LLM-extract BMC for every discovered candidate.",
    )
    parser.add_argument(
        "--peer-cache",
        type=Path,
        default=None,
        help="Peer BMC cache CSV (default: output/module_04/peer_bmc_cache.csv).",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Peers in summary CSV.")
    parser.add_argument(
        "--deck-delay",
        type=float,
        default=30.0,
        help="Seconds between decks on --all.",
    )
    parser.add_argument("--fetch-retries", type=int, default=2)
    parser.add_argument("--min-peer-fields", type=int, default=2)
    parser.add_argument("--model", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    enriched_in = args.enriched_in or resolve_enriched_bmc()

    run_all = args.all or not args.deck.strip()
    deck_id = args.deck.strip()
    if args.force:
        skip_existing = False
    elif args.skip_existing is not None:
        skip_existing = args.skip_existing
    else:
        skip_existing = run_all

    if not enriched_in.exists():
        print(f"[{TAG}] Missing {enriched_in}. Run Module 03 first.", file=sys.stderr)
        return 1

    by_deck = bmc_by_deck_from_csv(enriched_in)
    if run_all:
        deck_ids = sorted(by_deck.keys())
        print(f"[{TAG}] Processing all {len(deck_ids)} deck(s) from {enriched_in}", flush=True)
        if skip_existing:
            print(f"[{TAG}] Skip existing ranked CSVs (use --force to redo)", flush=True)
    else:
        if deck_id not in by_deck:
            print(f"[{TAG}] {deck_id} not in {enriched_in}.", file=sys.stderr)
            return 1
        deck_ids = [deck_id]

    model = args.model or ollama_model()
    try:
        check_ollama(model)
    except RuntimeError as e:
        print(f"[{TAG}] {e}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.peer_cache or (args.out_dir / DEFAULT_PEER_BMC_CACHE_CSV.name)
    peer_cache = load_peer_bmc_cache(cache_path)
    if peer_cache:
        print(f"[{TAG}] Loaded {len(peer_cache)} cached peer BMC row(s) from {cache_path}", flush=True)

    websites_path = resolve_websites_csv()
    websites_by_pipeline = load_websites_csv(websites_path) if websites_path.exists() else {}
    peers_ranked_path = args.out_dir / DEFAULT_PEERS_RANKED_CSV.name
    top5_path = args.out_dir / DEFAULT_SIMILAR_TOP5_CSV.name
    vc_path = args.out_dir / DEFAULT_VC_DILIGENCE_CSV.name
    queries_path = args.out_dir / DEFAULT_SEARCH_QUERIES_CSV.name

    reprocess = set(deck_ids)
    wipe_all = args.force and run_all

    if wipe_all:
        peers_by_deck: dict[str, list[dict[str, object]]] = {}
        kept_queries: list[dict[str, object]] = []
        kept_vc_by_deck: dict[str, dict[str, object]] = {}
    else:
        existing_peer_rows = _load_csv_rows(peers_ranked_path)
        if args.force:
            existing_peer_rows = _rows_excluding_decks(existing_peer_rows, reprocess)
        peers_by_deck = _peers_by_deck_from_rows(existing_peer_rows)

        existing_query_rows = _load_csv_rows(queries_path)
        kept_queries = (
            _rows_excluding_decks(existing_query_rows, reprocess)
            if args.force
            else existing_query_rows
        )

        existing_vc_rows = _load_csv_rows(vc_path)
        if args.force:
            existing_vc_rows = _rows_excluding_decks(existing_vc_rows, reprocess)
        kept_vc_by_deck = {
            str(r.get("target_deck_id") or ""): r
            for r in existing_vc_rows
            if r.get("target_deck_id")
        }

    deck_query_log: list[dict[str, object]] = []
    summary_by_deck: dict[str, dict[str, object]] = {}
    failures = 0
    skipped = 0
    min_rows = args.top_k
    headers = {"User-Agent": USER_AGENT}
    delay_before_next = False

    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for did in deck_ids:
            deck_existing = peers_by_deck.get(did, [])
            if skip_existing and not args.force and len(deck_existing) >= min_rows:
                skipped += 1
                print(f"\n[{TAG}] ===== {did} ===== skip ({len(deck_existing)} rows)", flush=True)
                summary_by_deck[did] = _build_vc_summary_row(
                    did,
                    candidates_discovered=0,
                    scored_rows=deck_existing,
                    top_k=args.top_k,
                )
                continue

            if delay_before_next and args.deck_delay > 0:
                print(f"[{TAG}] waiting {args.deck_delay:.0f}s before next deck ...", flush=True)
                time.sleep(args.deck_delay)

            print(f"\n[{TAG}] ===== {did} =====", flush=True)
            label = _startup_label(did, websites_by_pipeline)
            scored, summary = process_deck(
                did,
                by_deck[did],
                websites_by=websites_by_pipeline,
                client=client,
                model=model,
                max_candidates=args.max_candidates,
                startup_name=label,
                fetch_retries=args.fetch_retries,
                min_peer_fields=args.min_peer_fields,
                top_k=args.top_k,
                query_log=deck_query_log,
                peer_cache=peer_cache,
                two_stage=not args.no_two_stage,
                max_llm_peers=args.max_llm_peers,
                min_prefilter_score=args.min_prefilter,
            )
            delay_before_next = True
            summary_by_deck[did] = summary
            if not scored:
                if summary.get("vc_note") == "No enriched BMC — skipped.":
                    failures += 1
                peers_by_deck.pop(did, None)
                continue
            peers_by_deck[did] = scored

    all_peers: list[dict[str, object]] = []
    all_top5: list[dict[str, object]] = []
    vc_summaries: list[dict[str, object]] = []
    for did in sorted(by_deck.keys()):
        rows = peers_by_deck.get(did, [])
        all_peers.extend(rows)
        for row in rows[: args.top_k]:
            all_top5.append(row)
        if did in summary_by_deck:
            vc_summaries.append(summary_by_deck[did])
        elif did in kept_vc_by_deck:
            vc_summaries.append(kept_vc_by_deck[did])
        elif rows:
            vc_summaries.append(
                _build_vc_summary_row(
                    did,
                    candidates_discovered=0,
                    scored_rows=rows,
                    top_k=args.top_k,
                )
            )

    query_log = kept_queries + deck_query_log

    if skipped:
        print(f"\n[{TAG}] Skipped {skipped} deck(s) with existing peer rows", flush=True)

    if all_peers:
        _write_csv(peers_ranked_path, RANKED_FIELDNAMES, all_peers)
        print(f"[{TAG}] All ranked peers -> {peers_ranked_path} ({len(all_peers)} rows)", flush=True)

    if query_log:
        _write_csv(queries_path, QUERY_LOG_FIELDNAMES, query_log)
        print(f"[{TAG}] Search queries -> {queries_path}", flush=True)

    if all_top5:
        _write_csv(top5_path, TOP5_FIELDNAMES, all_top5)
        print(
            f"\n[{TAG}] Top {args.top_k} per deck -> {top5_path} ({len(all_top5)} rows)",
            flush=True,
        )

    if vc_summaries:
        _write_csv(vc_path, VC_SUMMARY_FIELDNAMES, vc_summaries)
        print(f"[{TAG}] VC diligence summary -> {vc_path}", flush=True)

    if peer_cache:
        write_peer_bmc_cache(cache_path, peer_cache)
        print(f"[{TAG}] Peer BMC cache -> {cache_path} ({len(peer_cache)} domains)", flush=True)

    if all_top5:
        eval_dir = Path("eval") / "module_04"
        eval_dir.mkdir(parents=True, exist_ok=True)
        from eval.build_m04_rubric_template import main as build_rubric_main  # noqa: E402

        build_rubric_main(
            [
                "--module-04-dir",
                str(args.out_dir),
                "--top-k",
                str(args.top_k),
                "--out",
                str(eval_dir / "peer_relevance_rubric.csv"),
            ]
        )

    attempted = len(deck_ids) - skipped
    if attempted > 0 and failures == attempted:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
