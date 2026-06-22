"""Read/write Module 01 slide text as a single CSV (all decks)."""

from __future__ import annotations

import csv
from pathlib import Path

from support.csv_bmc import canonical_deck_id

SLIDES_CSV_FIELDS = ("deck_id", "page", "title", "body", "source", "char_count")

def slide_row(deck_id: str, slide: dict) -> dict[str, str | int]:
    return {
        "deck_id": canonical_deck_id(deck_id),
        "page": int(slide.get("page") or 0),
        "title": (slide.get("title") or "").strip(),
        "body": (slide.get("body") or "").strip(),
        "source": (slide.get("source") or "").strip(),
        "char_count": int(slide.get("char_count") or 0),
    }

def write_slides_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SLIDES_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SLIDES_CSV_FIELDS})

def load_slides_by_deck(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    by_deck: dict[str, list[dict]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            deck_id = canonical_deck_id((row.get("deck_id") or "").strip())
            if not deck_id:
                continue
            slide = {
                "page": int(row.get("page") or 0),
                "title": (row.get("title") or "").strip(),
                "body": (row.get("body") or "").strip(),
                "source": (row.get("source") or "").strip(),
                "char_count": int(row.get("char_count") or 0),
            }
            by_deck.setdefault(deck_id, []).append(slide)
    for slides in by_deck.values():
        slides.sort(key=lambda s: s["page"])
    return by_deck

def load_slides_for_deck(deck_id: str, path: Path) -> list[dict]:
    canon = canonical_deck_id(deck_id)
    return load_slides_by_deck(path).get(canon, [])

def deck_ids_from_slides_csv(path: Path) -> list[str]:
    return sorted(load_slides_by_deck(path).keys())
