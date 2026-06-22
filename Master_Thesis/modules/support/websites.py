"""Website seeds and lookup for Module 03 (discovery) and Module 04 (search labels)."""

from __future__ import annotations

import csv
from pathlib import Path

from support.csv_bmc import canonical_deck_id, startup_name_for_deck
from support.paths import DEFAULT_WEBSITES_CSV, resolve_module_01_slides
from support.slides_store import deck_ids_from_slides_csv, load_slides_for_deck
from support.web_fetch import extract_urls_from_slides, pick_deck_url

WEBSITE_CSV_FIELDS = (
    "deck_id",
    "startup_name",
    "website_url",
    "website_source",
    "discovered_website",
)


def website_seed_from_slides(slides: list, deck_id: str) -> dict[str, str]:
    """Extract startup label + deck URL hints from pitch-deck slides (no LLM)."""
    canon = canonical_deck_id(deck_id)
    startup_name = startup_name_for_deck(canon)

    website_url = ""
    source = ""
    urls = extract_urls_from_slides(slides)
    if urls:
        website_url = pick_deck_url(urls, startup_name)
        source = "deck_regex"

    return {
        "deck_id": canon,
        "startup_name": startup_name,
        "website_url": website_url,
        "website_source": source,
        "discovered_website": "",
    }


def lookup_website_info(
    deck_id: str,
    by_pipeline_deck: dict[str, dict[str, str]],
) -> dict[str, str]:
    canon = canonical_deck_id(deck_id)
    return by_pipeline_deck.get(canon, {})


def load_websites_csv(path: Path = DEFAULT_WEBSITES_CSV) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_deck: dict[str, dict[str, str]] = {}
    for row in rows:
        deck_id = canonical_deck_id((row.get("deck_id") or "").strip())
        if deck_id:
            by_deck[deck_id] = {**row, "deck_id": deck_id}
    return by_deck


def write_websites_csv(rows: list[dict[str, str]], path: Path = DEFAULT_WEBSITES_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(WEBSITE_CSV_FIELDS), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in WEBSITE_CSV_FIELDS})


def load_slides_for_deck_from_store(deck_id: str, slides_csv: Path | None = None) -> list[dict]:
    path = slides_csv or resolve_module_01_slides()
    return load_slides_for_deck(deck_id, path)


def build_websites_csv(
    deck_ids: list[str] | None = None,
    slides_csv: Path | None = None,
    out_path: Path = DEFAULT_WEBSITES_CSV,
) -> list[dict[str, str]]:
    """Build websites.csv from Module 01 slides (regex URL + brand names)."""
    slides_csv = slides_csv or resolve_module_01_slides()
    if deck_ids is None:
        deck_ids = deck_ids_from_slides_csv(slides_csv)

    rows_out: list[dict[str, str]] = []
    for deck_id in deck_ids:
        canon = canonical_deck_id(deck_id)
        slides = load_slides_for_deck(canon, slides_csv)
        if slides:
            rows_out.append(website_seed_from_slides(slides, canon))
        else:
            rows_out.append(
                {
                    "deck_id": canon,
                    "startup_name": startup_name_for_deck(canon),
                    "website_url": "",
                    "website_source": "",
                    "discovered_website": "",
                }
            )

    write_websites_csv(rows_out, out_path)
    return rows_out
