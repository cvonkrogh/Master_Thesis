"""Cross-deck cache for peer BMC extractions (keyed by homepage domain)."""

from __future__ import annotations

import csv
from pathlib import Path
from urllib.parse import urlparse

from support.schema import BMC_FIELDS
from support.web_fetch import normalize_url

CACHE_FIELDNAMES = [
    "domain",
    "url",
    "peer_name",
    "bmc_fields_filled",
    "model",
    *BMC_FIELDS,
]


def peer_domain_key(url: str) -> str:
    """Normalized registrable domain for cache lookup."""
    norm = normalize_url(url) or (url or "").strip()
    host = (urlparse(norm).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def load_peer_bmc_cache(path: Path) -> dict[str, dict[str, str]]:
    """Return {domain: row} from cache CSV."""
    if not path.exists():
        return {}
    cache: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            domain = (row.get("domain") or "").strip().lower()
            if not domain:
                domain = peer_domain_key(row.get("url") or "")
            if not domain:
                continue
            cache[domain] = {k: str(row.get(k) or "").strip() for k in CACHE_FIELDNAMES}
            cache[domain]["domain"] = domain
    return cache


def get_cached_peer_bmc(
    url: str,
    cache: dict[str, dict[str, str]],
    *,
    min_fields: int = 2,
) -> dict[str, str] | None:
    """Return 9-field BMC dict if cache hit is usable."""
    domain = peer_domain_key(url)
    if not domain:
        return None
    row = cache.get(domain)
    if not row:
        return None
    try:
        filled = int(row.get("bmc_fields_filled") or 0)
    except ValueError:
        filled = sum(1 for f in BMC_FIELDS if (row.get(f) or "").strip())
    if filled < min_fields:
        return None
    return {name: (row.get(name) or "").strip() for name in BMC_FIELDS}


def upsert_peer_bmc_cache(
    cache: dict[str, dict[str, str]],
    *,
    url: str,
    peer_name: str,
    peer_bmc: dict[str, str],
    bmc_fields_filled: int,
    model: str = "",
) -> None:
    domain = peer_domain_key(url)
    if not domain:
        return
    cache[domain] = {
        "domain": domain,
        "url": normalize_url(url) or url,
        "peer_name": peer_name,
        "bmc_fields_filled": str(bmc_fields_filled),
        "model": model,
        **{name: (peer_bmc.get(name) or "").strip() for name in BMC_FIELDS},
    }


def write_peer_bmc_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(cache.values(), key=lambda r: (r.get("domain") or "").lower())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CACHE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CACHE_FIELDNAMES})
