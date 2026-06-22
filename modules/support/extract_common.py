"""Shared helpers for Module 02 (BMC) and legacy 02b screening extraction."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, create_model

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from support.local_llm import chat_json, ollama_model
from support.csv_bmc import canonical_deck_id
from support.paths import resolve_module_01_slides
from support.schema import BMC_FIELDS, FIELD_DEFINITIONS, FIELD_NAMES, META_COLUMNS, SCREENING_FIELDS
from support.slides_store import deck_ids_from_slides_csv, load_slides_for_deck

DEFAULT_SLIDES_CSV = resolve_module_01_slides()

BASE_SYSTEM_PROMPT = """You are an analyst that extracts structured information from startup pitch decks for early-stage VC screening.

You will receive TWO text sources for the same pitch deck:
  (A) Structured slide text from Module 01 (title + body per slide).
  (B) Direct text extracted from the pitch-deck PDF file (native text + OCR per page).

Use BOTH sources together. If one source has text the other missed, use it. Prefer the clearer wording when they disagree.

RULES (very important):
1. Use ONLY information explicitly stated or directly implied by the deck sources. Do NOT use outside knowledge.
2. If a field is not mentioned, or you are not confident, set the field to "" (empty string). Never guess.
3. Write short, clean phrases — not full paragraphs. MAX 25 words per field. Never use bullet lists.
4. Do NOT paste pricing tables, competitor lists, or notes into any field. Put only the startup's own BMC summary.
5. Preserve numeric values as they appear in the slides when relevant (one short phrase).
6. For each field, return slide numbers (1-indexed) used as evidence. If the field is "", return [].
7. Slide text may contain OCR artefacts. Reconstruct meaning conservatively.

Return only the structured object. Do not add commentary outside the JSON fields."""

def _str_field(description: str):
    return (str, Field(default="", description=description))

def _evidence_field(field_name: str):
    return (
        list[int],
        Field(
            default_factory=list,
            description=(
                f"Slide numbers supporting `{field_name}`. Empty list if value is ''."
            ),
        ),
    )

def build_extraction_model(
    field_names: tuple[str, ...],
    model_name: str = "Extraction",
) -> type[BaseModel]:
    """Pydantic model with `fields` + `evidence` for the given field subset."""
    defs = {n: FIELD_DEFINITIONS[n] for n in field_names}
    fields_model = create_model(
        f"{model_name}Fields",
        **{name: _str_field(desc) for name, desc in defs.items()},
    )
    evidence_model = create_model(
        f"{model_name}Evidence",
        **{name: _evidence_field(name) for name in field_names},
    )

    class Extraction(BaseModel):
        fields: fields_model = Field(
            description="Extracted answers. Use '' if not in the deck.",
        )
        evidence: evidence_model = Field(
            description="Slide numbers (1-indexed) justifying each field.",
        )

    Extraction.model_rebuild()
    return Extraction

def format_slides(slides: list[dict], *, label: str = "", max_body_chars: int = 2000) -> str:
    blocks = []
    prefix = f"{label}: " if label else ""
    for s in slides:
        page = s.get("page", "?")
        title = (s.get("title") or "").strip()
        body = (s.get("body") or "").strip()
        if len(body) > max_body_chars:
            body = body[: max_body_chars - 3].rsplit(" ", 1)[0] + "..."
        block = [f"=== Slide {page} ==="]
        if title:
            block.append(f"Title: {title}")
        if body:
            block.append(f"Body:\n{body}")
        if not title and not body:
            block.append("(no extractable text on this slide)")
        blocks.append("\n".join(block))
    header = f"{prefix}({len(slides)} slides)\n\n" if prefix else ""
    return header + "\n\n".join(blocks)

def format_combined_deck_text(
    slides_json: list[dict],
    pdf_slides: list[dict],
    *,
    pdf_path: str = "",
) -> str:
    parts = [
        "SOURCE A — Structured slide text (Module 01 CSV):",
        format_slides(slides_json) if slides_json else "(empty)",
        "",
        f"SOURCE B — Direct pitch-deck PDF text{f' ({pdf_path})' if pdf_path else ''}:",
        format_slides(pdf_slides, label="PDF") if pdf_slides else "(PDF not available or empty)",
    ]
    return "\n".join(parts)

def build_user_prompt(deck_id: str, slides: list[dict], task_hint: str) -> str:
    return (
        f"Pitch deck: {deck_id}\n"
        f"Task: {task_hint}\n"
        f"Slides ({len(slides)} total):\n\n"
        f"{format_slides(slides)}\n\n"
        "Fill the schema. Use empty string for anything not in the slides."
    )

def build_bmc_user_prompt(
    deck_id: str,
    slides_json: list[dict],
    pdf_slides: list[dict],
    task_hint: str,
    *,
    pdf_path: str = "",
) -> str:
    return (
        f"Pitch deck: {deck_id}\n"
        f"Task: {task_hint}\n\n"
        f"{format_combined_deck_text(slides_json, pdf_slides, pdf_path=pdf_path)}\n\n"
        "Fill the schema from BOTH sources. Use empty string for anything not supported by the deck."
    )

def call_extract(
    slides: list[dict],
    deck_id: str,
    extraction_model: type[BaseModel],
    task_hint: str,
    extra_system: str = "",
    model: str | None = None,
    *,
    pdf_slides: list[dict] | None = None,
    pdf_path: str = "",
) -> BaseModel:
    system = BASE_SYSTEM_PROMPT
    if extra_system:
        system = system + "\n\n" + extra_system
    if pdf_slides is not None:
        user_content = build_bmc_user_prompt(
            deck_id, slides, pdf_slides, task_hint, pdf_path=pdf_path
        )
    else:
        user_content = build_user_prompt(deck_id, slides, task_hint)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    return chat_json(messages, extraction_model, model=model or ollama_model(), temperature=0.0)

def resolve_deck_ids(
    slides_csv: Path | None = None,
    deck_filter: list[str] | None = None,
) -> list[str]:
    """Deck ids from slides.csv (or all decks in the file)."""
    path = slides_csv or DEFAULT_SLIDES_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Slides CSV not found: {path}. Run Module 01 first."
        )
    ids = deck_ids_from_slides_csv(path)
    if not ids:
        raise FileNotFoundError(f"No decks in {path}. Run Module 01 first.")
    if deck_filter:
        wanted = {canonical_deck_id(d) for d in deck_filter}
        ids = [d for d in ids if d in wanted]
        if not ids:
            raise FileNotFoundError(f"No matching decks in {path} for {deck_filter!r}.")
    return ids

def _dedupe_slide_inputs(paths: list[Path]) -> list[Path]:
    """One slides file per GT deck — prefer Vision.slides.json over Connectly.slides.json."""
    by_canonical: dict[str, Path] = {}
    for path in paths:
        raw_id = deck_id_from_slides_path(path)
        canon = canonical_deck_id(raw_id)
        existing = by_canonical.get(canon)
        if existing is None:
            by_canonical[canon] = path
            continue
        if raw_id == canon and deck_id_from_slides_path(existing) != canon:
            by_canonical[canon] = path
    return sorted(by_canonical.values())

def load_slides(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a list of slides.")
    return data

def deck_id_from_slides_path(path: Path) -> str:
    name = path.name
    if name.endswith(".slides.json"):
        return name[: -len(".slides.json")]
    return path.stem

def save_partial_json(
    extraction: BaseModel,
    deck_id: str,
    n_slides: int,
    model: str,
    out_path: Path,
    extract_type: str,
) -> None:
    payload = {
        "deck_id": deck_id,
        "n_slides": n_slides,
        "model": model,
        "extract_type": extract_type,
        "fields": extraction.fields.model_dump(),
        "evidence": extraction.evidence.model_dump(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

def summarize_extraction(extraction: BaseModel) -> str:
    fields = extraction.fields.model_dump()
    filled = sum(1 for v in fields.values() if v)
    return f"{filled}/{len(fields)} fields filled"

def upsert_csv_row(csv_path: Path, row: dict[str, str], key: str = "deck_id") -> None:
    rows: list[dict[str, str]] = []
    existing_fields: list[str] = []
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            rows = list(reader)

    replaced = False
    for i, r in enumerate(rows):
        if r.get(key) == row[key]:
            rows[i] = {**r, **row}
            replaced = True
            break
    if not replaced:
        rows.append(row)

    canonical = list(META_COLUMNS) + list(FIELD_NAMES)
    fieldnames = list(canonical)
    for f in existing_fields:
        if f not in fieldnames:
            fieldnames.append(f)
    for k in row.keys():
        if k not in fieldnames:
            fieldnames.append(k)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({fn: r.get(fn, "") for fn in fieldnames})

def load_partial_fields(json_path: Path) -> dict[str, str]:
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {k: str(v) for k, v in (data.get("fields") or {}).items()}

def merge_deck_fields(
    deck_id: str,
    json_dir: Path,
) -> dict[str, str]:
    """Merge fields from {deck}.bmc.json and {deck}.screening.json."""
    merged: dict[str, str] = {n: "" for n in FIELD_NAMES}
    for suffix in (".bmc.json", ".screening.json"):
        path = json_dir / f"{deck_id}{suffix}"
        merged.update(load_partial_fields(path))
    return merged

def row_for_screening_csv(
    deck_id: str,
    n_slides: int,
    model: str,
    fields: dict[str, str],
) -> dict[str, str]:
    row: dict[str, str] = {
        "deck_id": deck_id,
        "n_slides": str(n_slides),
        "model": model,
    }
    for name in FIELD_NAMES:
        row[name] = str(fields.get(name, ""))
    return row

def migrate_extracted_json(json_dir: Path) -> int:
    """Split legacy {deck}.extracted.json into .bmc.json + .screening.json."""
    count = 0
    for path in json_dir.glob("*.extracted.json"):
        deck_id = path.name[: -len(".extracted.json")]
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = data.get("fields") or {}
        evidence = data.get("evidence") or {}
        base = {
            "deck_id": deck_id,
            "n_slides": data.get("n_slides", 0),
            "model": data.get("model", ""),
        }
        for extract_type, names in (("bmc", BMC_FIELDS), ("screening", SCREENING_FIELDS)):
            out = json_dir / f"{deck_id}.{extract_type}.json"
            if out.exists():
                continue
            subset = {n: fields.get(n, "") for n in names}
            ev_subset = {n: evidence.get(n, []) for n in names}
            payload = {
                **base,
                "extract_type": extract_type,
                "fields": subset,
                "evidence": ev_subset,
            }
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1
    return count

def rebuild_screening_csv(json_dir: Path, csv_path: Path, model_label: str) -> None:
    deck_ids: set[str] = set()
    for path in json_dir.glob("*.bmc.json"):
        deck_ids.add(path.name.replace(".bmc.json", ""))
    for path in json_dir.glob("*.screening.json"):
        deck_ids.add(path.name.replace(".screening.json", ""))

    for deck_id in sorted(deck_ids):
        bmc_path = json_dir / f"{deck_id}.bmc.json"
        scr_path = json_dir / f"{deck_id}.screening.json"
        n_slides = 0
        model = model_label
        for p in (bmc_path, scr_path):
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                n_slides = int(data.get("n_slides") or n_slides)
                model = str(data.get("model") or model)
        fields = merge_deck_fields(deck_id, json_dir)
        upsert_csv_row(
            csv_path,
            row_for_screening_csv(deck_id, n_slides, model, fields),
        )
