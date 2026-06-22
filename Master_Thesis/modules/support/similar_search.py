"""Discover candidate peer startup URLs for Module 04.

BMC-driven search, competitor-style queries, always-on VC seed pass, and
stricter homepage URL filtering.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from support.web_fetch import (
    HTTP_TIMEOUT,
    extract_urls_from_text,
    normalize_url,
    search_engine_results,
    search_engine_text,
    slugify,
    try_url,
)

QUERY_SLEEP_SEC = 3.0
MAX_URLS_PER_QUERY = 4

SKIP_HOST_SUFFIXES = (
    "wikipedia.org",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
    "reddit.com",
    "medium.com",
    "ideasai.com",
    "substack.com",
    "forbes.com",
    "techcrunch.com",
    "crunchbase.com",
    "pitchbook.com",
    "g2.com",
    "capterra.com",
    "apple.com",
    "play.google.com",
    "apps.apple.com",
    "google.com",
    "grokipedia.com",
    "quora.com",
    "wellfound.com",
    "angel.co",
    "sourceforge.net",
    "slashdot.org",
    "producthunt.com",
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "inc42.com",
    "theverge.com",
    "wired.com",
    "startuptostartup.com",
    "promoteproject.com",
    "theonestartup.com",
    "topbusinesssoftware.com",
    "softwareadvice.com",
    "getapp.com",
    "trustpilot.com",
    "startupnewswire.in",
    "worldmetrics.org",
    "explodingtopics.com",
    "bennetttechinnovation.com",
    "startupsavant.com",
    "topstartups.io",
    "ideasai.com",
)

SKIP_HOST_TLDS = (".gov", ".edu")

LISTICLE_PATH_HINTS = (
    "/blog/",
    "/blog",
    "/news/",
    "/article/",
    "/articles/",
    "/press/",
    "/top-",
    "/best-",
    "/list/",
    "/review/",
    "/reviews/",
    "/category",
    "/categories/",
    "/tag/",
    "/tags/",
    "/directory",
    "/companies",
    "/company/",
    "/software/",
    "/startup/",
    "/startups/",
    "/orgs/",
    "/p/",
    "/post/",
    "/posts/",
    "/insights/",
    "/compare/",
    "/comparison",
    "/alternative",
    "/alternatives",
    "/competitor",
    "/competitors",
    "/similar",
    "/vs-",
    "-vs-",
    "/wiki/",
    "/wiki",
    "-alternatives",
    "-competitors",
)

EXTRA_SKIP_HOST_SUFFIXES = (
    "yahoo.com",
    "businessnewsdaily.com",
    "businessinsider.com",
    "indianstartuptimes.com",
    "retailboss.co",
    "mindmybusinessnyc.com",
    "intellias.com",
    "agfundernews.com",
    "sentientmedia.org",
    "gfi.org",
    "hitlab.org",
    "imd.org",
    "finance.via.news",
    "nelincs.gov.uk",
    "jobcrusher.com",
    "flrepoter.com",
    "aivaagency.com",
    "digistreetmedia.com",
    "thepromota.co.uk",
    "jamesryall.com",
    "futurealternative.com.au",
    "tasteradio.com",
    "daily.sevenfifty.com",
    "myimg.ai",
)

EXTRA_LISTICLE_PATH_HINTS = (
    "/episodes/",
    "/episode/",
    "/interviews/",
    "/interview/",
    "/video/",
    "/videos/",
    "/watch/",
    "/guide/",
    "/guides/",
    "/resources/",
    "/learn/",
    "/solutions/",
    "/solutions-for-",
    "/scaling-",
    "/launching-",
    "/starting-",
    "/building-",
    "/how-to-",
    "/what-to-",
)

GENERIC_VP_PHRASES = frozenset(
    {
        "intelligent safety category",
        "category",
        "platform",
        "solution",
        "software",
        "technology",
        "startup",
        "company",
    }
)

AMBIGUOUS_BRAND_NAMES = frozenset(
    {
        "aura",
        "macro",
        "mojo",
        "mustard",
        "circular",
        "cowboy",
        "fluence",
        "morty",
        "pilgrim",
        "soul",
    }
)

LISTICLE_PATH_MARKERS = (
    "/alternatives",
    "/alternative",
    "/competitors",
    "/competitor",
    "/similar",
    "/best-",
    "/top-",
    "/vs-",
    "-vs-",
    "/compare",
    "/comparison",
    "/review/",
    "/reviews/",
)

@dataclass
class PeerCandidate:
    peer_name: str
    url: str
    source: str

def _first_phrase(text: str, max_len: int = 80) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    for sep in (",", ";", "|"):
        if sep in text:
            return text.split(sep)[0].strip()[:max_len]
    return text[:max_len].strip()

def _short_phrase(text: str, max_words: int = 6) -> str:
    """First clause of a phrase, capped to ``max_words`` words.

    Splits off only the first claim (before ``;``/``|``/``.``) so multi-claim
    value propositions don't become long, search-unfriendly queries, then keeps
    at most ``max_words`` words.
    """
    t = (text or "").strip()
    if not t:
        return ""

    cut = min(
        (t.find(sep) for sep in (",", ";", "|", ".") if sep in t),
        default=-1,
    )
    if cut > 0:
        t = t[:cut].strip()
    words = t.split()
    return " ".join(words[:max_words]).strip(" ,;:-")

def _is_generic_vp(vp: str) -> bool:
    v = vp.strip().lower()
    if not v or len(v) < 12:
        return True
    if v in GENERIC_VP_PHRASES:
        return True
    if v.endswith(" category") or " category " in v:
        return True
    return False

def _searchable_vp(bmc_row: dict[str, str], max_words: int = 6) -> str:
    """Pick the best short, specific phrase for search.

    Genericness is checked on the *shortened* phrase (the first clause), so a
    field like "Monitoring, Alerting, ..." is rejected as generic instead of
    becoming the bare word "Monitoring". When the first clause is generic, later
    clauses in the same field are tried (e.g. Aura's "proactive protection…").
    """
    fields = ("value_proposition", "key_resources", "key_activities")
    fallback = ""
    for field in fields:
        raw = (bmc_row.get(field) or "").strip()
        if not raw:
            continue
        clauses = re.split(r"[,;|]", raw)
        for clause in clauses:
            short = _short_phrase(clause.strip(), max_words)
            if not short:
                continue
            if not _is_generic_vp(short):
                return short
            fallback = fallback or short
    return fallback

def is_ambiguous_brand(name: str) -> bool:
    """True when a bare brand token is likely to match unrelated web results."""
    token = (name or "").strip().lower()
    if not token:
        return True
    if token in AMBIGUOUS_BRAND_NAMES:
        return True

    if " " not in token and len(token) <= 6:
        return True
    return False

def _bmc_context_label(bmc_row: dict[str, str], max_words: int = 5) -> str:
    """Short sector/product phrase to disambiguate name-based queries."""
    seg = _short_phrase(bmc_row.get("customer_segments", ""), 3)
    vp = _searchable_vp(bmc_row, max_words)
    if seg and vp:
        return f"{seg} {vp}"
    return vp or seg

def is_listicle_url(url: str) -> bool:
    path = (urlparse(url).path or "").lower()
    return any(marker in path for marker in LISTICLE_PATH_MARKERS)

def brand_from_domain(url: str) -> str:
    """Registrable brand token from a URL/domain (e.g. https://connectly.ai -> connectly)."""
    domain = _domain_key(url) if url else ""
    if not domain:
        return ""
    base = domain.split(".")[0]
    return base.replace("-", " ").strip()

def build_search_queries(
    bmc_row: dict[str, str],
    *,
    deck_id: str = "",
    startup_name: str = "",
    target_domain: str = "",
) -> list[str]:
    """Build web search queries from BMC + canonical name (no GT competitor list).

    When the target's verified website (``target_domain``) is known, prefer the
    domain brand as the strongest name token; this avoids homonym collisions for
    short/generic deck ids (e.g. Aura, Macro, morty).
    """
    seg = _short_phrase(bmc_row.get("customer_segments", ""), 4)
    vp = _searchable_vp(bmc_row, 6)
    rev = _first_phrase(bmc_row.get("revenue_model", ""))
    act = _short_phrase(bmc_row.get("key_activities", ""), 5)
    brand = brand_from_domain(target_domain)
    name = (brand or startup_name or deck_id).strip()

    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        q = re.sub(r"\s+", " ", q.strip())[:160]
        if len(q) < 8:
            return
        key = q.lower()
        if key in seen:
            return
        seen.add(key)
        queries.append(q)

    if seg and vp:
        add(f"{seg} {vp} startup")
    if vp:
        add(f"{vp} startup company")
    if act and seg:
        add(f"{act} for {seg} startup")
    if seg:
        add(f"{seg} B2B startup official website")

    context = _bmc_context_label(bmc_row)
    ambiguous = is_ambiguous_brand(name)

    if name and len(name) > 2:
        if ambiguous:
            if context:
                add(f"startups similar to {name} {context}")
                add(f"{name} {context} competitors")
            if target_domain and vp:
                add(f"{brand_from_domain(target_domain)} {vp} competitors")
        else:
            if context:
                add(f"startups similar to {name} {context}")
            add(f"{name} competitors")
            if " " not in name:
                add(f"startups similar to {name}")

    if vp and rev and "subscription" in rev.lower():
        add(f"{vp} subscription startup")

    return queries[:8]

def build_vc_sector_queries(
    bmc_row: dict[str, str],
    *,
    seg: str = "",
    vp: str = "",
) -> list[str]:
    """Queries aimed at VC/sector lists to harvest startup names (second pass)."""
    seg = seg or _short_phrase(bmc_row.get("customer_segments", ""), 4)
    vp = vp or _searchable_vp(bmc_row, 6)
    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        q = re.sub(r"\s+", " ", q.strip())[:160]
        if len(q) < 8:
            return
        key = q.lower()
        if key in seen:
            return
        seen.add(key)
        queries.append(q)

    if vp:
        add(f"{vp} startup seed funding")
        add(f"{vp} early stage startup")
    if seg and vp:
        add(f"{seg} {vp} startups to watch")
    if seg:
        add(f"top {seg} startups")

    return queries[:4]

def build_competitor_queries(
    bmc_row: dict[str, str],
    *,
    deck_id: str = "",
    startup_name: str = "",
    target_domain: str = "",
) -> list[str]:
    """VC-style competitor queries (sector + product type)."""
    seg = _short_phrase(bmc_row.get("customer_segments", ""), 4)
    vp = _searchable_vp(bmc_row, 6)
    act = _short_phrase(bmc_row.get("key_activities", ""), 4)
    brand = brand_from_domain(target_domain)
    name = (brand or startup_name or deck_id).strip()

    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        q = re.sub(r"\s+", " ", q.strip())[:160]
        if len(q) < 8:
            return
        key = q.lower()
        if key in seen:
            return
        seen.add(key)
        queries.append(q)

    if vp:
        add(f"{vp} startup competitors")
        add(f"companies like {vp} startup")
    if seg and vp:
        add(f"{seg} {vp} companies startup")
    if seg:
        add(f"{seg} startup landscape competitors")
    if act and seg:
        add(f"{act} {seg} startup company")
    context = _bmc_context_label(bmc_row)
    if name and len(name) > 3:
        if is_ambiguous_brand(name) and context:
            add(f"{name} {context} alternatives startup")
        elif not is_ambiguous_brand(name):
            add(f"{name} alternatives startup")

    return queries[:6]

def build_all_search_queries(
    bmc_row: dict[str, str],
    *,
    deck_id: str = "",
    startup_name: str = "",
    target_domain: str = "",
) -> list[tuple[str, str]]:
    """Return (query, kind) pairs for logging."""
    bmc = build_search_queries(
        bmc_row,
        deck_id=deck_id,
        startup_name=startup_name,
        target_domain=target_domain,
    )
    comp = build_competitor_queries(
        bmc_row,
        deck_id=deck_id,
        startup_name=startup_name,
        target_domain=target_domain,
    )
    vc = build_vc_sector_queries(bmc_row)
    return (
        [(q, "bmc") for q in bmc]
        + [(q, "competitor") for q in comp]
        + [(q, "vc") for q in vc]
    )

_NAME_STOP = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "startup",
        "startups",
        "company",
        "companies",
        "best",
        "top",
        "watch",
        "list",
        "guide",
        "news",
        "blog",
        "how",
        "what",
        "why",
        "your",
        "their",
        "these",
        "those",
        "from",
        "into",
        "about",
    }
)

def extract_startup_names_from_results(results: list[dict], max_names: int = 8) -> list[str]:
    """Pull plausible company names from search titles/snippets (VC/sector lists)."""
    names: list[str] = []
    seen: set[str] = set()

    def add(raw: str) -> None:
        raw = re.sub(r"\s+", " ", (raw or "").strip())
        if not raw or len(raw) < 3 or len(raw) > 48:
            return
        words = raw.split()
        if len(words) < 1 or len(words) > 5:
            return
        if any(w.lower() in _NAME_STOP for w in words):
            return
        if not words[0][0].isupper():
            return
        key = raw.lower()
        if key in seen:
            return
        seen.add(key)
        names.append(raw)

    for r in results:
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        for chunk in (title, body[:200]):
            if not chunk:
                continue
            for part in re.split(r"[-–|:•]", chunk):
                add(part.strip())
            for m in re.finditer(
                r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\b",
                chunk,
            ):
                add(m.group(1))
        if len(names) >= max_names:
            break

    return names[:max_names]

def resolve_startup_homepage(
    name: str,
    client: httpx.Client,
    exclude_domains: set[str],
) -> Optional[str]:
    """Find a reachable homepage for a harvested startup name."""
    for query in (f"{name} official website startup", f"{name} startup company"):
        for url in search_engine_text(query, max_results=4):
            if should_skip_url(url, exclude_domains):
                continue
            resolved = try_url(client, url)
            if resolved and not should_skip_url(resolved, exclude_domains):
                return resolved
    return None

def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()

def _domain_key(url: str) -> str:
    host = _host(url)
    if host.startswith("www."):
        host = host[4:]
    return host

def _path_depth(url: str) -> int:
    path = (urlparse(url).path or "").strip("/")
    if not path:
        return 0
    return len(path.split("/"))

def _matches_excluded(domain: str, exclude_domains: set[str]) -> bool:
    if not domain:
        return False
    base = domain.split(".")[0]
    for ex in exclude_domains:
        if not ex:
            continue
        if domain == ex or base == ex or ex in base or base in ex:
            return True
    return False

def should_skip_url(url: str, exclude_domains: set[str]) -> bool:
    norm = normalize_url(url)
    if not norm:
        return True
    host = _host(norm)
    if not host:
        return True
    if any(host.endswith(s) or host == s for s in SKIP_HOST_SUFFIXES):
        return True
    if any(host.endswith(tld) for tld in SKIP_HOST_TLDS):
        return True
    if host == "ycombinator.com":
        return True
    domain = _domain_key(norm)
    if _matches_excluded(domain, exclude_domains):
        return True
    path = (urlparse(norm).path or "").lower()
    if any(h in path for h in LISTICLE_PATH_HINTS):
        return True

    if _path_depth(norm) > 2:
        return True
    return False

def should_skip_url_strict(url: str, exclude_domains: set[str]) -> bool:
    """Prefer company homepages; reject articles and deep paths."""
    norm = normalize_url(url)
    if not norm:
        return True
    host = _host(norm)
    if not host:
        return True
    if any(host.endswith(s) or host == s for s in SKIP_HOST_SUFFIXES):
        return True
    if any(host.endswith(s) or s in host for s in EXTRA_SKIP_HOST_SUFFIXES):
        return True
    if any(host.endswith(tld) for tld in SKIP_HOST_TLDS):
        return True
    if host == "ycombinator.com":
        return True
    domain = _domain_key(norm)
    if _matches_excluded(domain, exclude_domains):
        return True
    path = (urlparse(norm).path or "").lower()
    if any(h in path for h in LISTICLE_PATH_HINTS):
        return True
    if any(h in path for h in EXTRA_LISTICLE_PATH_HINTS):
        return True
    if _path_depth(norm) > 1:
        return True
    return False

def peer_name_from_url(url: str) -> str:
    domain = _domain_key(url)
    base = domain.split(".")[0] if domain else "unknown"
    return base.replace("-", " ").title()

def _resolve_strict(
    query: str,
    client: httpx.Client,
    exclude_domains: set[str],
    max_urls: int = MAX_URLS_PER_QUERY,
) -> list[str]:
    urls: list[str] = []
    for url in search_engine_text(query, max_results=max_urls + 12):
        if should_skip_url_strict(url, exclude_domains):
            continue
        resolved = try_url(client, url)
        if not resolved or should_skip_url_strict(resolved, exclude_domains):
            continue
        urls.append(resolved)
        if len(urls) >= max_urls:
            break
    return urls

def _find_listicle_urls(
    query: str,
    client: httpx.Client,
    exclude_domains: set[str],
    *,
    max_urls: int = 3,
) -> list[str]:
    """Search hits that look like competitor roundups (not startup homepages)."""
    found: list[str] = []
    seen: set[str] = set()
    for url in search_engine_text(query, max_results=max_urls + 14):
        norm = normalize_url(url)
        if not norm or norm in seen:
            continue
        host = _host(norm)
        if not host:
            continue
        if any(host.endswith(s) or host == s for s in SKIP_HOST_SUFFIXES):
            continue
        if _matches_excluded(_domain_key(norm), exclude_domains):
            continue
        if not is_listicle_url(norm):
            continue
        seen.add(norm)
        found.append(norm)
        if len(found) >= max_urls:
            break
    return found

def extract_startup_homepage_links(
    html: str,
    base_url: str,
    exclude_domains: set[str],
    *,
    max_links: int = 12,
) -> list[str]:
    """Pull external shallow URLs from a competitor listicle page."""
    from urllib.parse import urljoin

    soup = BeautifulSoup(html, "lxml")
    base_host = _domain_key(base_url)
    found: list[str] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = (anchor.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        full = normalize_url(urljoin(base_url, href))
        if not full:
            continue
        host = _domain_key(full)
        if not host or host == base_host or host in seen:
            continue
        if should_skip_url_strict(full, exclude_domains):
            continue
        seen.add(host)
        found.append(full)
        if len(found) >= max_links:
            break
    return found

def _fetch_listicle_html(url: str, client: httpx.Client) -> str:
    try:
        resp = client.get(url, follow_redirects=True, timeout=HTTP_TIMEOUT)
    except httpx.HTTPError:
        return ""
    if resp.status_code >= 400:
        return ""
    ct = resp.headers.get("content-type", "")
    if "html" not in ct.lower():
        return ""
    return resp.text

def _run_listicle_harvest_pass(
    bmc_row: dict[str, str],
    client: httpx.Client,
    exclude_domains: set[str],
    add,
    *,
    deck_id: str = "",
    startup_name: str = "",
    target_domain: str = "",
    max_new: int,
) -> int:
    """Mine competitor listicles for startup homepages (what Google shows humans)."""
    if max_new <= 0:
        return 0

    brand = brand_from_domain(target_domain)
    name = (brand or startup_name or deck_id).strip()
    vp = _searchable_vp(bmc_row, 6)
    context = _bmc_context_label(bmc_row)
    queries: list[str] = []
    seen_q: set[str] = set()

    def add_q(q: str) -> None:
        q = re.sub(r"\s+", " ", q.strip())[:160]
        if len(q) < 8:
            return
        key = q.lower()
        if key in seen_q:
            return
        seen_q.add(key)
        queries.append(q)

    if name and context:
        add_q(f"startups similar to {name} {context}")
    if vp:
        add_q(f"best {vp} startups")
        add_q(f"{vp} startup competitors list")
    if name and vp and is_ambiguous_brand(name):
        add_q(f"{name} {vp} competitors")

    added = 0
    for i, query in enumerate(queries):
        if added >= max_new:
            break
        if i > 0:
            time.sleep(QUERY_SLEEP_SEC)

        for result in search_engine_results(query, max_results=10):
            if added >= max_new:
                break
            body = (result.get("body") or "")[:500]
            for raw in extract_urls_from_text(body):
                if added >= max_new:
                    break
                resolved = try_url(client, raw)
                if not resolved or should_skip_url_strict(resolved, exclude_domains):
                    continue
                peer = PeerCandidate(
                    peer_name=peer_name_from_url(resolved),
                    url=resolved,
                    source=f"listicle_snippet:{query[:48]}",
                )
                if add(peer):
                    added += 1

        for listicle_url in _find_listicle_urls(query, client, exclude_domains, max_urls=2):
            if added >= max_new:
                break
            html = _fetch_listicle_html(listicle_url, client)
            if not html:
                continue
            for link in extract_startup_homepage_links(
                html, listicle_url, exclude_domains, max_links=10
            ):
                if added >= max_new:
                    break
                resolved = try_url(client, link)
                if not resolved or should_skip_url_strict(resolved, exclude_domains):
                    continue
                peer = PeerCandidate(
                    peer_name=peer_name_from_url(resolved),
                    url=resolved,
                    source=f"listicle_page:{query[:40]}",
                )
                if add(peer):
                    added += 1

    return added

def _run_vc_seed_pass(
    bmc_row: dict[str, str],
    client: httpx.Client,
    exclude_domains: set[str],
    add,
    *,
    seg: str,
    vp: str,
    max_new: int,
) -> int:
    added = 0
    vc_queries = build_vc_sector_queries(bmc_row, seg=seg, vp=vp)
    for i, query in enumerate(vc_queries):
        if added >= max_new:
            break
        if i > 0:
            time.sleep(QUERY_SLEEP_SEC)
        results = search_engine_results(query, max_results=8)
        names = extract_startup_names_from_results(results, max_names=10)
        for name in names:
            if added >= max_new:
                break
            homepage = resolve_startup_homepage(name, client, exclude_domains)
            if not homepage or should_skip_url_strict(homepage, exclude_domains):
                continue
            peer = PeerCandidate(
                peer_name=name,
                url=homepage,
                source=f"vc_seed:{query[:48]}",
            )
            if add(peer):
                added += 1
    return added

def collect_peer_candidates(
    bmc_row: dict[str, str],
    client: httpx.Client,
    exclude_domains: set[str],
    max_candidates: int = 20,
    *,
    deck_id: str = "",
    startup_name: str = "",
    target_domain: str = "",
) -> list[PeerCandidate]:
    """Gather peer URLs: BMC + competitor queries + always-on VC seed pass."""
    candidates: list[PeerCandidate] = []
    seen_domains: set[str] = set()

    def add(candidate: PeerCandidate) -> bool:
        domain = _domain_key(candidate.url)
        if not domain or domain in seen_domains:
            return False
        if should_skip_url_strict(candidate.url, exclude_domains):
            return False
        seen_domains.add(domain)
        candidates.append(candidate)
        return True

    seg = _short_phrase(bmc_row.get("customer_segments", ""), 4)
    vp = _searchable_vp(bmc_row, 6)

    bmc_queries = build_search_queries(
        bmc_row,
        deck_id=deck_id,
        startup_name=startup_name,
        target_domain=target_domain,
    )
    comp_queries = build_competitor_queries(
        bmc_row,
        deck_id=deck_id,
        startup_name=startup_name,
        target_domain=target_domain,
    )
    all_bmc = [(q, "bmc") for q in bmc_queries] + [(q, "competitor") for q in comp_queries]

    for i, (query, kind) in enumerate(all_bmc):
        if len(candidates) >= max_candidates:
            break
        if i > 0:
            time.sleep(QUERY_SLEEP_SEC)
        for url in _resolve_strict(query, client, exclude_domains):
            add(
                PeerCandidate(
                    peer_name=peer_name_from_url(url),
                    url=url,
                    source=f"{kind}_search:{query[:72]}",
                )
            )
            if len(candidates) >= max_candidates:
                break

    remaining = max(0, max_candidates - len(candidates))
    if remaining > 0:
        _run_listicle_harvest_pass(
            bmc_row,
            client,
            exclude_domains,
            add,
            deck_id=deck_id,
            startup_name=startup_name,
            target_domain=target_domain,
            max_new=min(remaining, 8),
        )
        remaining = max(0, max_candidates - len(candidates))

    if remaining > 0:
        _run_vc_seed_pass(
            bmc_row,
            client,
            exclude_domains,
            add,
            seg=seg,
            vp=vp,
            max_new=remaining,
        )

    return candidates[:max_candidates]

def target_exclude_domains(deck_id: str, target_url: str = "") -> set[str]:
    domains = {slugify(deck_id)}
    if target_url:
        domains.add(_domain_key(target_url))
    domains.discard("")
    return domains
