"""Resolve pitch-deck PDF paths and extract page text for Module 02."""

from __future__ import annotations

from pathlib import Path

from support.csv_bmc import canonical_deck_id

DEFAULT_PDF_DIR = Path("data/pitch_decks")

# Filename aliases (Vision deck file may be Vision.pdf; GT id is Vision).
PDF_NAME_ALIASES: dict[str, list[str]] = {
    "Vision": ["Vision", "Connectly"],
}


def pdf_lookup_names(deck_id: str) -> list[str]:
    canon = canonical_deck_id(deck_id)
    return PDF_NAME_ALIASES.get(canon, [canon])


def resolve_pdf_path(deck_id: str, pdf_dir: Path = DEFAULT_PDF_DIR) -> Path | None:
    for name in pdf_lookup_names(deck_id):
        path = pdf_dir / f"{name}.pdf"
        if path.exists():
            return path
    return None


def extract_pdf_slides(pdf_path: Path, *, use_ocr: bool = True, verbose: bool = False) -> list[dict]:
    """Extract slides from PDF using Module 01 logic."""
    import importlib.util
    import sys

    mod_path = Path(__file__).resolve().parent.parent / "pipeline" / "01_pdf_to_slides.py"
    spec = importlib.util.spec_from_file_location("pdf_to_slides", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    slides = mod.extract_slides(pdf_path, use_ocr=use_ocr, verbose=verbose)
    return [
        {
            "page": s.page,
            "title": s.title,
            "body": s.body,
            "source": s.source,
            "char_count": s.char_count,
        }
        for s in slides
    ]
