"""Clamp BMC field text to short screening phrases (post-LLM safety net)."""

from __future__ import annotations

import re

from support.schema import BMC_FIELDS

MAX_BMC_FIELD_WORDS = 25
MAX_BMC_FIELD_CHARS = 280

# Model rants / commentary that sometimes leak into a field value.
_NOTE_MARKERS = re.compile(
    r"\n\s*(Note:|Additionally:|For example:|In summary:|It's important to note)",
    re.IGNORECASE,
)


def clamp_bmc_field(
    value: str,
    *,
    max_words: int = MAX_BMC_FIELD_WORDS,
    max_chars: int = MAX_BMC_FIELD_CHARS,
) -> str:
    """Trim a single BMC cell to a short phrase."""
    text = re.sub(r"\s+", " ", (value or "").strip())
    if not text:
        return ""

    match = _NOTE_MARKERS.search(text)
    if match:
        text = text[: match.start()].strip()
    else:
        for marker in ("Note:", "Additionally:", "For example:", "In summary:"):
            idx = text.find(marker)
            if idx > 20:
                text = text[:idx].strip()
                break

    if "\n\n" in text and len(text) > max_chars:
        text = text.split("\n\n", 1)[0].strip()

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    if len(text) > max_chars:
        cut = text[: max_chars - 3].rsplit(" ", 1)[0]
        text = f"{cut}..." if cut else text[:max_chars]

    return text.strip()


def clamp_bmc_row(
    row: dict[str, str],
    fields: tuple[str, ...] = BMC_FIELDS,
) -> dict[str, str]:
    """Clamp all BMC columns in a row."""
    out = dict(row)
    for name in fields:
        if name in out:
            out[name] = clamp_bmc_field(out.get(name, ""))
    return out
