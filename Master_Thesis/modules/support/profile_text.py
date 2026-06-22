"""Convert BMC rows to structured text profiles for similarity scoring."""

from __future__ import annotations

from support.schema import BMC_FIELDS


def bmc_row_to_profile_text(row: dict[str, str], deck_id: str | None = None) -> str:
    """Fixed-order BMC text: one `field: value` line per non-empty cell."""
    label = deck_id or (row.get("deck_id") or "").strip()
    lines: list[str] = []
    if label:
        lines.append(f"deck_id: {label}")
    for field in BMC_FIELDS:
        value = (row.get(field) or "").strip()
        if value:
            lines.append(f"{field}: {value}")
    return "\n".join(lines)


def bmc_fields_filled_count(row: dict[str, str]) -> int:
    return sum(1 for f in BMC_FIELDS if (row.get(f) or "").strip())
