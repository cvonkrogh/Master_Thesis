"""Website discovery and page scraping for Module 03."""

from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

USER_AGENT = (
    "AI-Investment-Screening/0.1 (+thesis prototype; "
    "github.com/Constantinvonkrogh/AI-investment-screening)"
)
HTTP_TIMEOUT = 10.0
MAX_PAGE_CHARS = 8_000
MAX_TOTAL_CHARS = 30_000

_HTTP_FETCH_ERRORS = (httpx.HTTPError, UnicodeEncodeError)

def _clear_client_cookies(client: httpx.Client) -> None:
    try:
        client.cookies.clear()
    except Exception:
        pass

CANDIDATE_TLDS = (".com", ".io", ".ai", ".app", ".co", ".com.br")
SUBPATHS = ("", "/about", "/about-us", "/team", "/company", "/partners", "/our-team")
DROP_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "form", "svg")

URL_IN_TEXT_RE = re.compile(
    r"(?:https?://|www\.)[a-z0-9][-a-z0-9.]*[a-z0-9](?:/[^\s)\]\"'<>]*)?",
    re.IGNORECASE,
)

def slugify(name: str) -> str:
    s = name.lower()
    return re.sub(r"[^a-z0-9]+", "", s)

def candidate_urls(startup_name: str) -> list[str]:
    slug = slugify(startup_name)
    if not slug:
        return []
    return [f"https://www.{slug}{tld}" for tld in CANDIDATE_TLDS] + [
        f"https://{slug}{tld}" for tld in CANDIDATE_TLDS
    ]

def normalize_url(url: str) -> Optional[str]:
    url = url.strip()
    if not url:
        return None
    if url.startswith("//"):
        url = "https:" + url
    elif not url.startswith(("http://", "https://")):
        if url.startswith("/") or " " in url:
            return None
        url = "https://" + url.lstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    return url

def extract_urls_from_text(text: str) -> list[str]:
    found: list[str] = []
    for match in URL_IN_TEXT_RE.findall(text or ""):
        norm = normalize_url(match if match.startswith("http") else f"https://{match}")
        if norm:
            found.append(norm)
    return found

def extract_urls_from_slides(slides: list[dict]) -> list[str]:
    urls: list[str] = []
    for slide in slides:
        block = f"{slide.get('title', '')}\n{slide.get('body', '')}"
        urls.extend(extract_urls_from_text(block))
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def pick_deck_url(urls: list[str], deck_id: str) -> str:
    """Prefer a URL whose host matches the deck / startup name."""
    slug = slugify(deck_id)
    if slug:
        for u in urls:
            host = slugify(urlparse(u).hostname or "")
            if slug in host or host in slug:
                return u
    return urls[0] if urls else ""

def try_url(client: httpx.Client, url: str) -> Optional[str]:
    url = normalize_url(url) or ""
    if not url:
        return None
    try:
        resp = client.head(url, follow_redirects=True, timeout=HTTP_TIMEOUT)
        if resp.status_code >= 400:
            return None
        ct = resp.headers.get("content-type", "")
        if "html" not in ct.lower() and ct:
            return None
        return str(resp.url)
    except (*_HTTP_FETCH_ERRORS, httpx.InvalidURL, ValueError):
        _clear_client_cookies(client)
        return None

def search_engine_lookup(startup_name: str) -> Optional[str]:
    urls = search_engine_urls(startup_name, max_results=5)
    return urls[0] if urls else None

def search_engine_urls(startup_name: str, max_results: int = 5) -> list[str]:
    return search_engine_text(f"{startup_name} official website", max_results=max_results)

def _run_ddg_text(query: str, max_results: int) -> list[dict]:
    """Run DuckDuckGo text search via ddgs package."""
    try:
        from ddgs import DDGS

        with DDGS(timeout=HTTP_TIMEOUT) as client:
            return list(client.text(query, max_results=max_results))
    except Exception:
        return []

def search_engine_results(query: str, max_results: int = 8) -> list[dict]:
    """DuckDuckGo text hits with href, title, and body (for seed-name extraction)."""
    return _run_ddg_text(query, max_results)

def search_engine_text(query: str, max_results: int = 5) -> list[str]:
    results = search_engine_results(query, max_results)
    skip_hosts = (
        "wikipedia.org",
        "linkedin.com",
        "twitter.com",
        "x.com",
        "facebook.com",
        "reddit.com",
        "grokipedia.com",
        "quora.com",
    )
    urls: list[str] = []
    seen: set[str] = set()
    for r in results:
        href = r.get("href") or r.get("url")
        if not href:
            continue
        host = urlparse(href).hostname or ""
        if any(host.endswith(s) or host == s for s in skip_hosts):
            continue
        norm = normalize_url(href)
        if norm and norm not in seen:
            seen.add(norm)
            urls.append(norm)
    return urls

def iter_website_candidates(
    startup_name: str,
    seed_url: str,
) -> list[tuple[str, str]]:
    """Ordered (url, source) pairs: deck seed → heuristics → search results."""
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(raw: str, source: str) -> None:
        norm = normalize_url(raw.strip()) if raw else None
        if norm and norm not in seen:
            seen.add(norm)
            candidates.append((norm, source))

    if seed_url.strip():
        add(seed_url.strip(), "deck")
    for cand in candidate_urls(startup_name):
        add(cand, "heuristic")
    for url in search_engine_urls(startup_name):
        add(url, "search")
    return candidates

def fetch_homepage_snippet(
    url: str,
    client: httpx.Client,
    max_chars: int = 3_000,
) -> Optional[str]:
    """Fetch homepage text only (used for validation before full scrape)."""
    url = normalize_url(url) or ""
    if not url:
        return None
    try:
        resp = client.get(url, follow_redirects=True, timeout=HTTP_TIMEOUT)
    except _HTTP_FETCH_ERRORS:
        _clear_client_cookies(client)
        return None
    if resp.status_code >= 400:
        return None
    ct = resp.headers.get("content-type", "")
    if "html" not in ct.lower():
        return None
    text = strip_html(resp.text).strip()
    if not text:
        return None
    return text[:max_chars]

def discover_website(
    startup_name: str,
    existing_url: str,
    client: httpx.Client,
    tag: str = "03",
    verbose: bool = True,
) -> Optional[str]:
    """Return first reachable candidate without deck alignment (legacy fast path)."""
    for url, source in iter_website_candidates(startup_name, existing_url):
        resolved = try_url(client, url)
        if resolved:
            if verbose:
                print(f"  [{tag}] using URL ({source}): {resolved}", flush=True)
            return resolved
    return None

def strip_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text(separator="\n").splitlines() if ln.strip()]
    return "\n".join(lines)

def fetch_pages(
    base_url: str,
    client: httpx.Client,
    tag: str = "03",
    verbose: bool = True,
    homepage_only: bool = False,
) -> list[tuple[str, str]]:
    from urllib.parse import urljoin

    pages: list[tuple[str, str]] = []
    seen: set[str] = set()
    total = 0
    subpaths = ("",) if homepage_only else SUBPATHS
    for sub in subpaths:
        url = urljoin(base_url, sub) if sub else base_url
        if url in seen:
            continue
        seen.add(url)
        try:
            resp = client.get(url, follow_redirects=True, timeout=HTTP_TIMEOUT)
        except _HTTP_FETCH_ERRORS:
            _clear_client_cookies(client)
            continue
        if resp.status_code >= 400:
            continue
        ct = resp.headers.get("content-type", "")
        if "html" not in ct.lower():
            continue
        text = strip_html(resp.text)[:MAX_PAGE_CHARS]
        if not text:
            continue
        final_url = str(resp.url)
        pages.append((final_url, text))
        total += len(text)
        if verbose:
            print(f"  [{tag}] fetched {final_url} ({len(text)} chars)", flush=True)
        if total >= MAX_TOTAL_CHARS:
            break
    return pages

def fetch_pages_with_retry(
    base_url: str,
    client: httpx.Client,
    tag: str = "03",
    *,
    retries: int = 2,
    verbose: bool = True,
    homepage_only: bool = False,
) -> list[tuple[str, str]]:
    """Fetch site pages; retry and fall back to homepage-only snippet."""
    import time

    _clear_client_cookies(client)
    for attempt in range(max(1, retries)):
        pages = fetch_pages(
            base_url,
            client,
            tag=tag,
            verbose=verbose and attempt == 0,
            homepage_only=homepage_only,
        )
        if pages:
            return pages
        if attempt + 1 < retries:
            time.sleep(1.5)
    snippet = fetch_homepage_snippet(base_url, client, max_chars=MAX_PAGE_CHARS)
    if snippet:
        norm = normalize_url(base_url) or base_url
        return [(norm, snippet)]
    return []
