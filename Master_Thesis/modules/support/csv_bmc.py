"""Shared helpers for BMC-only CSV files (9 Business Model Canvas fields)."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

from support.paths import DEFAULT_GT_BMC_PD  # noqa: TC001 — runtime import
from support.schema import BMC_FIELDS

# Legacy numeric ids 1–10 in older GT exports; prefer startup_name when present.
GT_ROW_TO_DECK: dict[str, str] = {
    "1": "Palta",
    "2": "Aura",
    "3": "Bespoken_spirits",
    "4": "Jobox",
    "5": "Macro",
    "6": "Sable",
    "7": "Sharpist",
    "8": "Vision",
    "9": "morty",
    "10": "multus",
}

# PDF / pipeline deck_id stems that differ from GT startup_name.
DECK_ALIASES: dict[str, str] = {
    "Connectly": "Vision",
    "AI_rudder": "AI Rudder",
    "mustart": "Mustard",
    "pilgrim": "Pilgrim Soul",
    "scipher": "Scipher Medicine",
}

# Public brand name for website/search (GT deck_id → startup name).
DECK_STARTUP_NAMES: dict[str, str] = {
    "Vision": "Connectly",
}

_gt_startup_names_cache: list[str] | None = None


def _normalize_deck_key(text: str) -> str:
    """Fold accents/punctuation for PDF stem ↔ GT startup_name matching."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("Ÿ", "u").replace("ÿ", "u")
    s = re.sub(r"[^a-z0-9]+", "", s.lower())
    s = re.sub(r"gr[uuy]n$", "grun", s)  # Liefergrün / LiefergrŸn
    return s


def _gt_startup_names(gt_path: Path = DEFAULT_GT_BMC_PD) -> list[str]:
    global _gt_startup_names_cache
    if _gt_startup_names_cache is None:
        _gt_startup_names_cache = [r["deck_id"] for r in load_gt_bmc_rows(gt_path)]
    return _gt_startup_names_cache


def _deck_id_from_gt_row(row: dict[str, str]) -> str:
    startup = (row.get("startup_name") or "").strip()
    raw_id = (row.get("deck_id") or "").strip()
    if startup:
        return startup
    return GT_ROW_TO_DECK.get(raw_id, raw_id)


def canonical_deck_id(deck_id: str) -> str:
    """Normalize pipeline deck id to GT startup_name (Connectly → Vision, etc.)."""
    if deck_id in DECK_ALIASES:
        return DECK_ALIASES[deck_id]
    key = _normalize_deck_key(deck_id)
    if not key:
        return deck_id
    try:
        for gt_name in _gt_startup_names():
            if _normalize_deck_key(gt_name) == key:
                return gt_name
        for gt_name in _gt_startup_names():
            gk = _normalize_deck_key(gt_name)
            if len(key) >= 5 and gk.startswith(key):
                return gt_name
    except FileNotFoundError:
        pass
    return deck_id


def startup_name_for_deck(deck_id: str) -> str:
    """Brand name for web discovery / peer search (Vision → Connectly)."""
    return DECK_STARTUP_NAMES.get(canonical_deck_id(deck_id), canonical_deck_id(deck_id))


def write_bmc_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write deck_id + 9 BMC columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["deck_id", *BMC_FIELDS]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _open_gt_csv(path: Path):
    """Open GT file trying common Excel export encodings."""
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            f = path.open("r", encoding=encoding, newline="")
            f.read(4096)
            f.seek(0)
            return f
        except UnicodeDecodeError:
            continue
    raise ValueError(f"{path}: could not decode as UTF-8 or Latin-1/Windows-1252")


def _detect_delimiter(first_line: str) -> str:
    if first_line.count(";") > first_line.count(","):
        return ";"
    return ","


def load_gt_bmc_rows(gt_path: Path = DEFAULT_GT_BMC_PD) -> list[dict[str, str]]:
    """Load pitch-deck BMC GT CSV and return BMC fields per deck in GT order."""
    f = _open_gt_csv(gt_path)
    with f:
        first = f.readline()
        f.seek(0)
        delimiter = _detect_delimiter(first)
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames:
            reader.fieldnames = [(n or "").strip() for n in reader.fieldnames]

        records: list[dict[str, str]] = []
        for row in reader:
            deck_id = _deck_id_from_gt_row(row)
            if not deck_id:
                continue
            rec: dict[str, str] = {"deck_id": deck_id}
            for field in BMC_FIELDS:
                rec[field] = (row.get(field) or "").strip()
            records.append(rec)
    return records


def load_gt_row(deck_id: str, gt_path: Path = DEFAULT_GT_BMC_PD) -> dict[str, str]:
    """Load one GT row (all columns present in the CSV)."""
    f = _open_gt_csv(gt_path)
    with f:
        first = f.readline()
        f.seek(0)
        delimiter = _detect_delimiter(first)
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames:
            reader.fieldnames = [(n or "").strip() for n in reader.fieldnames]

        for row in reader:
            mapped = _deck_id_from_gt_row(row)
            if mapped != canonical_deck_id(deck_id):
                continue
            out: dict[str, str] = {"deck_id": mapped}
            for key in reader.fieldnames or []:
                if not key:
                    continue
                out[key] = (row.get(key) or "").strip()
            return out
    return {}


def write_gt_bmc(
    gt_path: Path = DEFAULT_GT_BMC_PD,
    out_path: Path | None = None,
) -> list[dict[str, str]]:
    """Re-write pitch-deck BMC GT (optional copy path)."""
    rows = load_gt_bmc_rows(gt_path)
    if out_path is not None:
        write_bmc_csv(out_path, rows)
    return rows


def pred_by_deck_from_screening(screening_csv: Path) -> dict[str, dict[str, str]]:
    """Read full screening.csv; return canonical deck_id -> BMC fields."""
    with screening_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    by_deck: dict[str, dict[str, str]] = {}
    for row in rows:
        deck_id = (row.get("deck_id") or "").strip()
        if not deck_id:
            continue
        canonical = canonical_deck_id(deck_id)
        if canonical in by_deck and deck_id != canonical:
            continue
        by_deck[canonical] = {f: (row.get(f) or "").strip() for f in BMC_FIELDS}
    return by_deck


def screening_bmc_rows_aligned(
    gt_rows: list[dict[str, str]],
    pred_by_deck: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Build screening_bmc rows: GT deck order, AI values or empty."""
    empty = {f: "" for f in BMC_FIELDS}
    return [
        {"deck_id": g["deck_id"], **pred_by_deck.get(g["deck_id"], empty)}
        for g in gt_rows
    ]


def write_screening_bmc_from_preds(
    pred_by_deck: dict[str, dict[str, str]],
    out_path: Path,
    gt_path: Path = DEFAULT_GT_BMC_PD,
) -> list[dict[str, str]]:
    """Write screening_bmc.csv aligned to GT deck order from in-memory predictions."""
    gt_rows = load_gt_bmc_rows(gt_path)
    rows = screening_bmc_rows_aligned(gt_rows, pred_by_deck)
    write_bmc_csv(out_path, rows)
    return rows


def write_screening_bmc(
    screening_csv: Path,
    out_path: Path,
    gt_path: Path = DEFAULT_GT_BMC_PD,
) -> list[dict[str, str]]:
    """Write screening_bmc.csv aligned to GT deck order (all companies in GT file).

    Prefer reading *.bmc.json files when ``screening_csv`` is a directory;
    otherwise extract BMC columns from the full screening CSV.
    """
    gt_rows = load_gt_bmc_rows(gt_path)
    if screening_csv.is_dir():
        pred_by_deck = pred_by_deck_from_bmc_json(screening_csv)
    else:
        pred_by_deck = pred_by_deck_from_screening(screening_csv)
    rows = screening_bmc_rows_aligned(gt_rows, pred_by_deck)
    write_bmc_csv(out_path, rows)
    return rows


def pred_by_deck_from_bmc_json(json_dir: Path) -> dict[str, dict[str, str]]:
    """Load {deck}.bmc.json files; map Connectly -> Vision for GT alignment."""
    by_deck: dict[str, dict[str, str]] = {}
    for path in json_dir.glob("*.bmc.json"):
        deck_id = path.name[: -len(".bmc.json")]
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = data.get("fields") or {}
        canonical = canonical_deck_id(deck_id)
        by_deck[canonical] = {f: str(fields.get(f) or "").strip() for f in BMC_FIELDS}
    return by_deck


def load_screening_bmc_rows(path: Path) -> list[dict[str, str]]:
    """Load screening_bmc.csv as ordered list of rows."""
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, str]] = []
    for row in rows:
        deck_id = (row.get("deck_id") or "").strip()
        if not deck_id:
            continue
        rec = {"deck_id": deck_id}
        for field in BMC_FIELDS:
            rec[field] = (row.get(field) or "").strip()
        out.append(rec)
    return out


def bmc_by_deck_from_csv(path: Path) -> dict[str, dict[str, str]]:
    return {r["deck_id"]: r for r in load_screening_bmc_rows(path)}


def write_enriched_bmc(
    rows: list[dict[str, str]],
    out_path: Path,
) -> None:
    write_bmc_csv(out_path, rows)
