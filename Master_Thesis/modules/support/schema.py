"""
Shared schema for the screening pipeline.

Both Module 02 (deck-only extraction) and Module 03 (public-data enrichment)
fill the same set of fields, so the canonical definitions live here. Keeping a
single source of truth means the screening.csv and screening_enriched.csv have
identical column orders and the diff between them is meaningful.
"""

from __future__ import annotations


# Field name -> short definition shown to the LLM in the structured-output
# JSON schema's `description` slot. The wording matters: it tells the model
# what each Business-Model-Canvas term means in the context of a pitch deck.
FIELD_DEFINITIONS: dict[str, str] = {
    "startup_name": "Official company / product name as it appears on the deck.",
    "website_url": "Public website URL of the company (e.g. https://palta.com). Empty if not on the deck.",
    "founding_year": "Year the company was founded, as stated on the deck (e.g. 2019). Empty if not mentioned.",
    "customer_segments": "Which customer groups the company serves (demographics, geography, B2C/B2B, vertical).",
    "value_proposition": "What problem the product solves and why customers choose it. Short, concrete phrasing.",
    "channels": "How the product reaches users — app stores, direct sales, partnerships, paid ads, etc.",
    "customer_relationships": "Type of relationship with users — subscription, self-service, community, account managers, etc.",
    "revenue_model": "How money is made — subscription, transaction fee, ads, freemium, licensing, marketplace take rate.",
    "key_resources": "Critical assets the company relies on — proprietary data, IP, brand, team, technology, licences.",
    "key_activities": "What the company actually does to deliver value — product development, content production, ML/AI, sales motion.",
    "key_partners": "Important external parties — distribution partners, suppliers, integrations, investors-as-partners.",
    "cost_structure": "Main cost drivers — CAC, R&D, infrastructure, content licensing, COGS, salaries.",
    "traction_users": "User-side traction with numbers if stated — MAU, DAU, subscribers, downloads, retention.",
    "traction_revenue": "Revenue-side traction with numbers — ARR, MRR, GMV, growth rate, gross margin.",
    "stage": "Funding stage — pre-seed / seed / Series A / B / etc., either explicit or inferable from ARR/round size.",
    "funding_ask": "What is being raised — round size, valuation, use of funds, lead investor sought.",
    "team_summary": "One- or two-line summary of the founding team — names, roles, prior companies.",
    "claimed_TAM": "Market-size figures stated by the company — TAM/SAM/SOM with currency and year if given.",
    "competition": "Named competitors or competitive positioning the deck mentions.",
    "geography": "Geographic markets the company operates in or targets — countries, regions, expansion plans.",
}

FIELD_NAMES: tuple[str, ...] = tuple(FIELD_DEFINITIONS.keys())

# The nine Business Model Canvas building blocks (Osterwalder & Pigneur, 2010).
# Used for Module 02.1 evaluation: compare deck-only extraction vs human GT on
# BMC fields only, separate from traction / market / identity fields.
BMC_FIELDS: tuple[str, ...] = (
    "customer_segments",
    "value_proposition",
    "channels",
    "customer_relationships",
    "revenue_model",
    "key_resources",
    "key_activities",
    "key_partners",
    "cost_structure",
)

# Non-BMC screening fields (Module 02b): identity, traction, market, team.
SCREENING_FIELDS: tuple[str, ...] = tuple(n for n in FIELD_NAMES if n not in BMC_FIELDS)

# Metadata columns that come before the schema fields in the screening CSVs.
META_COLUMNS: tuple[str, ...] = ("deck_id", "n_slides", "model")
