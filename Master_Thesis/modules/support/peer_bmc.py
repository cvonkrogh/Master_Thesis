"""Extract BMC profiles from peer startup websites (local LLM)."""

from __future__ import annotations

from pydantic import BaseModel, Field, create_model

from support.bmc_clamp import clamp_bmc_row
from support.local_llm import chat_json
from support.schema import BMC_FIELDS, FIELD_DEFINITIONS

PEER_SYSTEM_PROMPT = """You extract a Business Model Canvas from a company's public website text.

Rules:
1. Use ONLY the website text provided. No outside knowledge.
2. Fill all nine BMC fields when supported; use "" if not stated.
3. Short phrases only — MAX 25 words per field. No lists, notes, or commentary.
4. For each field, list supporting page URLs in evidence (empty list if value is "").
"""

def build_peer_bmc_model() -> type[BaseModel]:
    fields_model = create_model(
        "PeerBmcFields",
        **{
            name: (str, Field(default="", description=FIELD_DEFINITIONS[name]))
            for name in BMC_FIELDS
        },
    )
    evidence_model = create_model(
        "PeerBmcEvidence",
        **{
            name: (
                list[str],
                Field(default_factory=list, description=f"URLs supporting `{name}`."),
            )
            for name in BMC_FIELDS
        },
    )

    class PeerBmcExtraction(BaseModel):
        fields: fields_model = Field(
            description="BMC fields inferred from the website.",
        )
        evidence: evidence_model = Field(
            description="Supporting URLs per field.",
        )

    PeerBmcExtraction.model_rebuild()
    return PeerBmcExtraction

PEER_BMC_MODEL = build_peer_bmc_model()

def _build_user_prompt(peer_name: str, url: str, pages: list[tuple[str, str]]) -> str:
    lines = [
        f"Company: {peer_name}",
        f"Primary URL: {url}",
        "",
        "Website text:",
    ]
    for page_url, text in pages:
        lines.append(f"=== URL: {page_url} ===")
        snippet = (text or "")[:4000]
        if len(text or "") > 4000:
            snippet = snippet[:3997] + "..."
        lines.append(snippet)
        lines.append("")
    lines.append("Return all nine BMC fields plus evidence URLs.")
    return "\n".join(lines)

def extract_peer_bmc(
    peer_name: str,
    url: str,
    pages: list[tuple[str, str]],
    model: str | None = None,
) -> dict[str, str]:
    """Run local LLM to produce a 9-field BMC dict from scraped pages."""
    if not pages:
        return {f: "" for f in BMC_FIELDS}

    result = chat_json(
        [
            {"role": "system", "content": PEER_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(peer_name, url, pages)},
        ],
        PEER_BMC_MODEL,
        model=model,
        temperature=0.0,
    )
    fields = result.fields.model_dump()
    raw = {name: str(fields.get(name) or "").strip() for name in BMC_FIELDS}
    return clamp_bmc_row(raw)
