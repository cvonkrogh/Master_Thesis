#!/usr/bin/env python3
"""
Module 03 — discover startup websites and enrich BMC from public web data (local LLM).

Inputs:
    output/module_01/slides.csv
    output/module_02/screening_bmc.csv

Outputs:
    output/module_03/websites.csv
    output/module_03/enriched_bmc.csv

Usage:
    python modules/pipeline/03_enrich_bmc.py
    python modules/pipeline/03_enrich_bmc.py --decks Palta,Sable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field, create_model

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from support.csv_bmc import bmc_by_deck_from_csv, load_screening_bmc_rows, write_enriched_bmc
from support.local_llm import chat_json, check_ollama, ollama_model
from support.paths import (
    DEFAULT_ENRICHED_BMC,
    DEFAULT_MODULE_01_SLIDES_CSV,
    DEFAULT_SCREENING_BMC,
    DEFAULT_WEBSITES_CSV,
)
from support.web_fetch import USER_AGENT, discover_website, fetch_pages
from support.website_validate import (
    discover_website_validated,
    load_deck_context,
)
from support.websites import (
    load_slides_for_deck_from_store,
    website_seed_from_slides,
    write_websites_csv,
)
from support.bmc_clamp import clamp_bmc_field
from support.schema import BMC_FIELDS, FIELD_DEFINITIONS

def build_web_enrichment_model() -> type[BaseModel]:
    """BMC fields + URL evidence lists (not slide numbers)."""
    fields_model = create_model(
        "BmcEnrichmentFields",
        **{
            name: (str, Field(default="", description=FIELD_DEFINITIONS[name]))
            for name in BMC_FIELDS
        },
    )
    evidence_model = create_model(
        "BmcEnrichmentEvidence",
        **{
            name: (
                list[str],
                Field(
                    default_factory=list,
                    description=f"URLs supporting `{name}`. Empty if value is ''.",
                ),
            )
            for name in BMC_FIELDS
        },
    )

    class BmcEnrichment(BaseModel):
        fields: fields_model = Field(
            description="BMC answers from website text. Use '' if unsupported.",
        )
        evidence: evidence_model = Field(
            description="Supporting URLs for each field.",
        )

    BmcEnrichment.model_rebuild()
    return BmcEnrichment

BMC_ENRICHMENT_MODEL = build_web_enrichment_model()

SYSTEM_PROMPT = """You enrich empty Business Model Canvas fields using ONLY text scraped from a company's public website.

Rules:
1. You receive (a) existing BMC values from the pitch deck — some fields filled, some empty, and (b) website page text with URLs.
2. Fill ONLY empty BMC fields when the website text explicitly supports an answer. Never overwrite non-empty deck values.
3. No outside knowledge. If unsupported, leave "".
4. Short phrases, under 25 words. Preserve numbers verbatim.
5. For each field you fill from the web, list supporting URLs. Empty fields → empty URL lists.
6. For fields already filled from the deck, return the same text verbatim with an empty URL list."""

def _build_user_prompt(
    deck_id: str,
    existing: dict[str, str],
    pages: list[tuple[str, str]],
) -> str:
    empty = [f for f in BMC_FIELDS if not existing.get(f, "").strip()]
    filled = [f for f in BMC_FIELDS if existing.get(f, "").strip()]
    lines = [
        f"Deck: {deck_id}",
        "",
        "BMC fields already filled from pitch deck (do NOT change):",
    ]
    for f in filled:
        lines.append(f"  {f}: {existing[f]}")
    lines += ["", "Empty BMC fields to try filling from website:"]
    for f in empty:
        lines.append(f"  {f}")
    lines += ["", "Website text:"]
    for url, text in pages:
        lines.append(f"=== URL: {url} ===")
        lines.append(text)
        lines.append("")
    lines.append("Return structured BMC fields + evidence URLs.")
    return "\n".join(lines)

def _merge_bmc(
    deck_row: dict[str, str],
    enrichment: BaseModel,
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    enriched_fields = enrichment.fields.model_dump()
    enriched_evidence = enrichment.evidence.model_dump()
    merged: dict[str, str] = {"deck_id": deck_row["deck_id"]}
    provenance: dict[str, dict[str, object]] = {}

    for name in BMC_FIELDS:
        deck_val = (deck_row.get(name) or "").strip()
        web_val = (enriched_fields.get(name) or "").strip()
        urls = enriched_evidence.get(name) or []

        if deck_val:
            merged[name] = clamp_bmc_field(deck_val)
            provenance[name] = {"value": merged[name], "source": "deck", "evidence_urls": []}
        elif web_val:
            merged[name] = clamp_bmc_field(web_val)
            provenance[name] = {
                "value": web_val,
                "source": "web",
                "evidence_urls": list(urls),
            }
        else:
            merged[name] = ""
            provenance[name] = {"value": "", "source": "none", "evidence_urls": []}

    return merged, provenance

def _summarize(prov: dict[str, dict[str, object]]) -> str:
    deck = sum(1 for p in prov.values() if p["source"] == "deck")
    web = sum(1 for p in prov.values() if p["source"] == "web")
    none = sum(1 for p in prov.values() if p["source"] == "none")
    return f"deck={deck}, web={web}, empty={none}"

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Module 03 — discover websites and enrich BMC from public web data.",
    )
    parser.add_argument(
        "--bmc-in",
        type=Path,
        default=DEFAULT_SCREENING_BMC,
        help="Deck-only BMC CSV (default: screening_bmc.csv).",
    )
    parser.add_argument(
        "--slides-csv",
        type=Path,
        default=DEFAULT_MODULE_01_SLIDES_CSV,
        help="Module 01 slides CSV.",
    )
    parser.add_argument(
        "--websites-out",
        type=Path,
        default=DEFAULT_WEBSITES_CSV,
        help="Website seeds CSV (default: output/module_03/websites.csv).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=DEFAULT_ENRICHED_BMC,
        help="Enriched BMC CSV (default: enriched_bmc.csv).",
    )
    parser.add_argument("--model", default=None, help="Ollama model (default: llama3.1:8b).")
    parser.add_argument("--decks", default="", help="Comma-separated deck_ids (default: all in BMC CSV).")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip LLM deck-alignment check (use first reachable URL).",
    )
    args = parser.parse_args(argv)

    if not args.bmc_in.exists():
        print(f"[03] BMC input not found: {args.bmc_in}. Run Module 02 first.", file=sys.stderr)
        return 1

    bmc_rows = load_screening_bmc_rows(args.bmc_in)
    all_bmc_rows = list(bmc_rows)
    subset_decks: set[str] = set()
    if args.decks.strip():
        subset_decks = {d.strip() for d in args.decks.split(",") if d.strip()}
        bmc_rows = [r for r in bmc_rows if r["deck_id"] in subset_decks]

    model = args.model or ollama_model()
    try:
        check_ollama(model)
    except RuntimeError as e:
        print(f"[03] {e}", file=sys.stderr)
        return 1

    model = args.model or ollama_model()
    try:
        check_ollama(model)
    except RuntimeError as e:
        print(f"[03] {e}", file=sys.stderr)
        return 1

    enriched_rows: list[dict[str, str]] = []
    website_rows: list[dict[str, str]] = []

    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for row in bmc_rows:
            deck_id = row["deck_id"]
            slides = load_slides_for_deck_from_store(deck_id, args.slides_csv)
            if not slides:
                print(f"[03] {deck_id}: no slides in {args.slides_csv}", flush=True)

            seed = website_seed_from_slides(slides, deck_id) if slides else {
                "deck_id": deck_id,
                "startup_name": deck_id,
                "website_url": "",
                "website_source": "",
                "discovered_website": "",
            }
            startup_name = (seed.get("startup_name") or deck_id).strip() or deck_id
            seed_url = (seed.get("website_url") or "").strip()
            if seed_url:
                print(
                    f"[03] {deck_id}: deck seed URL {seed_url} ({seed.get('website_source', '')})",
                    flush=True,
                )

            print(f"[03] {deck_id}: discover website for '{startup_name}' ...", flush=True)
            if args.skip_validation:
                base_url = discover_website(startup_name, seed_url, client, tag="03")
            else:
                deck_context = load_deck_context(
                    deck_id,
                    args.slides_csv,
                    row,
                    startup_name,
                )
                base_url, _discovery_meta = discover_website_validated(
                    startup_name,
                    seed_url,
                    deck_context,
                    client,
                    model=model,
                    tag="03",
                )
                if not base_url:
                    print(
                        f"[03] {deck_id}: no validated website — keeping deck BMC only",
                        flush=True,
                    )

            seed["discovered_website"] = base_url or ""
            if base_url and not seed_url:
                seed["website_url"] = base_url
                seed["website_source"] = "search"

            pages: list[tuple[str, str]] = []
            if base_url:
                pages = fetch_pages(base_url, client, tag="03")

            if not pages:
                if base_url or args.skip_validation:
                    print(f"[03] {deck_id}: no web pages — keeping deck BMC only", flush=True)
                merged = {**row}
            else:
                try:
                    enrichment = chat_json(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": _build_user_prompt(deck_id, row, pages),
                            },
                        ],
                        BMC_ENRICHMENT_MODEL,
                        model=model,
                        temperature=0.0,
                    )
                except Exception as e:
                    print(f"[03] {deck_id}: FAILED ({e})", file=sys.stderr)
                    merged = {**row}
                else:
                    merged, provenance = _merge_bmc(row, enrichment)
                    print(f"[03] {deck_id}: {_summarize(provenance)}", flush=True)

            enriched_rows.append(merged)
            website_rows.append(seed)

    write_websites_csv(website_rows, args.websites_out)
    print(f"[03] websites.csv -> {args.websites_out}", flush=True)

    if subset_decks and args.csv_out.exists():
        merged_by_deck = bmc_by_deck_from_csv(args.csv_out)
        for row in enriched_rows:
            merged_by_deck[row["deck_id"]] = row
        enriched_rows = [merged_by_deck[r["deck_id"]] for r in all_bmc_rows if r["deck_id"] in merged_by_deck]

    write_enriched_bmc(enriched_rows, args.csv_out)
    print(f"[03] enriched_bmc.csv (GT order) -> {args.csv_out}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
