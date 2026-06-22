#!/usr/bin/env python3
"""
Module 02 — extract Business Model Canvas fields from pitch decks (9 fields).

Uses Module 01 slide CSV by default (no duplicate PDF/OCR). Optional --with-pdf
re-reads the PDF for a second text source (slower).

Inputs:  output/module_01/slides.csv
         data/pitch_decks/{deck}.pdf  (optional, --with-pdf only)

Outputs: output/module_02/screening_bmc.csv  (GT order, all companies)

Usage:
    python modules/pipeline/02_bmc_extract.py
    python modules/pipeline/02_bmc_extract.py --decks Aura,Macro
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from support.bmc_clamp import clamp_bmc_row
from support.csv_bmc import canonical_deck_id, write_screening_bmc_from_preds
from support.local_llm import check_ollama, ollama_model
from support.pdf_deck import DEFAULT_PDF_DIR, extract_pdf_slides, resolve_pdf_path
from support.paths import (
    DEFAULT_GT_BMC_PD,
    DEFAULT_MODULE_01_SLIDES_CSV,
    DEFAULT_SCREENING_BMC,
)
from support.schema import BMC_FIELDS
from support.extract_common import (
    build_extraction_model,
    call_extract,
    resolve_deck_ids,
    summarize_extraction,
)
from support.slides_store import load_slides_for_deck

BMC_MODEL = build_extraction_model(BMC_FIELDS, model_name="BmcExtraction")

BMC_SYSTEM_EXTRA = (
    "Extract ONLY the nine Business Model Canvas building blocks:\n"
    "customer_segments, value_proposition, channels, customer_relationships, "
    "revenue_model, key_resources, key_activities, key_partners, cost_structure."
)

BMC_TASK = "Extract the nine Business Model Canvas fields from this pitch deck."

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Module 02 — BMC deck extraction (CSV + PDF).")
    parser.add_argument(
        "--slides-csv",
        type=Path,
        default=DEFAULT_MODULE_01_SLIDES_CSV,
        help="Module 01 slides CSV.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help="Pitch-deck PDF directory.",
    )
    parser.add_argument(
        "--bmc-csv",
        type=Path,
        default=DEFAULT_SCREENING_BMC,
        help="BMC CSV aligned to GT (default: screening_bmc.csv).",
    )
    parser.add_argument("--decks", default="", help="Comma-separated deck_ids (default: all in slides CSV).")
    parser.add_argument("--model", default=None, help="Ollama model (default: llama3.1:8b).")
    parser.add_argument(
        "--with-pdf",
        action="store_true",
        help="Also re-read PDF text (duplicates Module 01 OCR — slower).",
    )
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR when using --with-pdf.")
    args = parser.parse_args(argv)

    model = args.model or ollama_model()
    try:
        check_ollama(model)
    except RuntimeError as e:
        print(f"[02] {e}", file=sys.stderr)
        return 1

    deck_filter = [d.strip() for d in args.decks.split(",") if d.strip()] or None
    deck_ids = resolve_deck_ids(args.slides_csv, deck_filter=deck_filter)
    pred_by_deck: dict[str, dict[str, str]] = {}

    for deck_id in deck_ids:
        canon = canonical_deck_id(deck_id)
        slides = load_slides_for_deck(canon, args.slides_csv)
        pdf_path = resolve_pdf_path(canon, args.pdf_dir)
        pdf_slides: list[dict] | None = None
        pdf_label = ""

        if args.with_pdf and pdf_path:
            print(
                f"[02] {canon}: CSV ({len(slides)} slides) + PDF {pdf_path.name} -> {model} ...",
                flush=True,
            )
            pdf_slides = extract_pdf_slides(pdf_path, use_ocr=not args.no_ocr, verbose=False)
            pdf_label = str(pdf_path)
        elif not slides and pdf_path:
            print(
                f"[02] {canon}: no CSV slides — falling back to PDF {pdf_path.name} -> {model} ...",
                flush=True,
            )
            pdf_slides = extract_pdf_slides(pdf_path, use_ocr=not args.no_ocr, verbose=False)
            pdf_label = str(pdf_path)
        else:
            print(f"[02] {canon}: {len(slides)} CSV slides -> {model} ...", flush=True)

        try:
            extraction = call_extract(
                slides,
                canon,
                BMC_MODEL,
                BMC_TASK,
                extra_system=BMC_SYSTEM_EXTRA,
                model=model,
                pdf_slides=pdf_slides,
                pdf_path=pdf_label,
            )
        except Exception as e:
            print(f"[02] {canon}: FAILED ({e})", file=sys.stderr)
            continue

        fields = extraction.fields.model_dump()
        pred_by_deck[canon] = clamp_bmc_row(
            {f: str(fields.get(f) or "").strip() for f in BMC_FIELDS}
        )
        print(f"[02] {canon}: {summarize_extraction(extraction)}", flush=True)

    if not pred_by_deck:
        print("[02] No decks extracted.", file=sys.stderr)
        return 1

    write_screening_bmc_from_preds(pred_by_deck, args.bmc_csv, gt_path=DEFAULT_GT_BMC_PD)
    print(f"[02] screening_bmc.csv ({len(pred_by_deck)} decks, GT order) -> {args.bmc_csv}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
