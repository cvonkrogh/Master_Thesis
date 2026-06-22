"""Validate that a candidate website matches the startup described in a pitch deck."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Optional

import httpx
from pydantic import BaseModel, Field

_MODULES_DIR = Path(__file__).resolve().parent.parent
if str(_MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULES_DIR))

from support.local_llm import chat_json
from support.web_fetch import (
    fetch_homepage_snippet,
    iter_website_candidates,
    try_url,
)
from support.schema import BMC_FIELDS

BMC_CONTEXT_FIELDS = (
    "value_proposition",
    "customer_segments",
    "revenue_model",
    "key_activities",
    "channels",
)

VALIDATION_SYSTEM = """You decide whether a candidate website belongs to the SAME startup described in a pitch deck.

Use ONLY the pitch deck context and the homepage text. Do NOT use outside knowledge.

Mark matches=true only when the website clearly describes the same company/product:
- industry and product type align with the deck
- geography or customer focus is compatible (not contradictory)
- team/traction/competition hints match when present

Mark matches=false when:
- it is a different company that happens to share a name
- the business category is unrelated (e.g. deck is health app, site is food brand)
- the page is domain parking, a generic landing page, or too vague to confirm
- evidence is weak or contradictory

Be conservative: when unsure, set matches=false and confidence=low."""

class WebsiteMatchResult(BaseModel):
    matches: bool = Field(description="True if the website belongs to the pitch-deck startup.")
    confidence: Literal["high", "medium", "low"] = Field(
        description="How confident you are in the matches decision.",
    )
    reason: str = Field(description="One short sentence explaining the decision.")

def load_deck_context(
    deck_id: str,
    slides_csv: Path,
    bmc_row: dict[str, str],
    startup_name: str,
) -> dict[str, str]:
    """Merge BMC row and slide excerpts for website validation."""
    from support.websites import load_slides_for_deck_from_store

    ctx: dict[str, str] = {
        "deck_id": deck_id,
        "startup_name": startup_name,
    }

    for field in BMC_FIELDS:
        val = (bmc_row.get(field) or "").strip()
        if val:
            ctx[field] = val

    slides = load_slides_for_deck_from_store(deck_id, slides_csv)
    excerpts: list[str] = []
    for i, slide in enumerate(slides[:4], 1):
        title = (slide.get("title") or "").strip()
        body = (slide.get("body") or "").strip()[:400]
        if title or body:
            excerpts.append(f"Slide {i}: {title}\n{body}")
    if excerpts:
        ctx["slide_excerpts"] = "\n\n".join(excerpts)

    return ctx

def format_deck_context(ctx: dict[str, str]) -> str:
    lines = [
        f"Startup name: {ctx.get('startup_name', '')}",
        f"Deck id: {ctx.get('deck_id', '')}",
    ]
    for key in BMC_CONTEXT_FIELDS:
        val = ctx.get(key, "")
        if val:
            lines.append(f"{key}: {val}")
    if ctx.get("slide_excerpts"):
        lines.append("")
        lines.append("Pitch deck excerpts:")
        lines.append(ctx["slide_excerpts"])
    return "\n".join(lines)

def validate_website_match(
    deck_context: dict[str, str],
    url: str,
    homepage_text: str,
    model: str | None = None,
) -> WebsiteMatchResult:
    user = "\n".join(
        [
            format_deck_context(deck_context),
            "",
            f"Candidate website URL: {url}",
            "",
            "Homepage text snippet:",
            homepage_text[:3000],
            "",
            "Does this website belong to the pitch-deck startup?",
        ]
    )
    return chat_json(
        [
            {"role": "system", "content": VALIDATION_SYSTEM},
            {"role": "user", "content": user},
        ],
        WebsiteMatchResult,
        model=model,
        temperature=0.0,
    )

def _accept_match(result: WebsiteMatchResult) -> bool:
    if not result.matches:
        return False
    return result.confidence in ("high", "medium")

def discover_website_validated(
    startup_name: str,
    seed_url: str,
    deck_context: dict[str, str],
    client: httpx.Client,
    model: str | None = None,
    tag: str = "03",
    verbose: bool = True,
) -> tuple[Optional[str], dict]:
    """Try candidate URLs in order; return the first that passes LLM deck alignment."""
    candidates_tried: list[dict[str, object]] = []

    for url, source in iter_website_candidates(startup_name, seed_url):
        resolved = try_url(client, url)
        if not resolved:
            if verbose:
                print(f"  [{tag}] skip {url} ({source}): not reachable", flush=True)
            candidates_tried.append(
                {"url": url, "source": source, "accepted": False, "reason": "not reachable"}
            )
            continue

        snippet = fetch_homepage_snippet(resolved, client)
        if not snippet:
            if verbose:
                print(f"  [{tag}] skip {resolved} ({source}): empty page", flush=True)
            candidates_tried.append(
                {"url": resolved, "source": source, "accepted": False, "reason": "empty page"}
            )
            continue

        try:
            result = validate_website_match(deck_context, resolved, snippet, model=model)
        except Exception as e:
            if verbose:
                print(f"  [{tag}] skip {resolved} ({source}): validation error ({e})", flush=True)
            candidates_tried.append(
                {
                    "url": resolved,
                    "source": source,
                    "accepted": False,
                    "reason": f"validation error: {e}",
                }
            )
            continue

        accepted = _accept_match(result)
        entry: dict[str, object] = {
            "url": resolved,
            "source": source,
            "accepted": accepted,
            "reason": result.reason,
            "confidence": result.confidence,
            "matches": result.matches,
        }
        candidates_tried.append(entry)

        if verbose:
            status = "ACCEPTED" if accepted else "rejected"
            print(f"  [{tag}] {status} {resolved} ({source}): {result.reason}", flush=True)

        if accepted:
            return resolved, {
                "source": source,
                "validated": True,
                "match_reason": result.reason,
                "confidence": result.confidence,
                "candidates_tried": candidates_tried,
            }

    return None, {
        "source": None,
        "validated": False,
        "match_reason": "no candidate passed deck alignment",
        "confidence": None,
        "candidates_tried": candidates_tried,
    }
