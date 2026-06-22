"""BMC profile similarity for Module 04.

Ranking uses core-field embedding cosine (customer_segments + value_proposition)
minus host, mention, and incumbent penalties. Full-profile ``embed_sim`` and
``combined_score`` are kept for debugging.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from eval.embedding_similarity import (
    DEFAULT_EMBED_MODEL,
    batch_cosine_similarity,
    pair_cosine_similarity,
)
from support.schema import BMC_FIELDS

CORE_FIELDS = ("customer_segments", "value_proposition")

INCUMBENT_HOSTS = frozenset(
    {
        "stripe.com",
        "twilio.com",
        "fiverr.com",
        "github.com",
        "zapier.com",
        "sentinelone.com",
        "circle.com",
        "webroot.com",
        "sap.com",
        "microsoft.com",
        "google.com",
        "amazon.com",
        "apple.com",
    }
)

INCUMBENT_PENALTY = 0.15

NON_STARTUP_HOST_SUFFIXES = (
    ".gov",
    ".edu",
    "startupnewswire",
    "worldmetrics.org",
    "explodingtopics.com",
    "bennetttechinnovation.com",
    "blumbergcapital.com",
    "topstartups.io",
    "startupsavant.com",
    "businessinsider.com",
    "techcrunch.com",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _norm(text)) if len(t) > 1}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def seq_ratio(a: str, b: str) -> float:
    if not a.strip() and not b.strip():
        return 1.0
    if not a.strip() or not b.strip():
        return 0.0
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def jaccard_combined(a: str, b: str) -> float:
    return 0.5 * jaccard(a, b) + 0.5 * seq_ratio(a, b)


def _field_line(field: str, row: dict[str, str]) -> str:
    value = (row.get(field) or "").strip()
    return f"{field}: {value}" if value else ""


def _fit_cosine(docs: list[str]) -> list[float]:
    if len(docs) < 2:
        return []
    cleaned = [d if d.strip() else "empty" for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(cleaned)
    sims = cosine_similarity(matrix[0:1], matrix[1:])[0]
    return [float(s) for s in sims]


def _host_penalty(url: str) -> float:
    host = (urlparse(url).hostname or "").lower()
    path = (urlparse(url).path or "").lower()
    penalty = 0.0
    if any(host.endswith(s) or s in host for s in NON_STARTUP_HOST_SUFFIXES):
        penalty += 0.12
    if any(h in path for h in ("/post/", "/news/", "/blog/", "/article/", "/insights/")):
        penalty += 0.08
    return penalty


def _target_mention_penalty(peer_profile: str, target_labels: list[str]) -> float:
    text = _norm(peer_profile)
    if not text:
        return 0.0
    mentions = 0
    for label in target_labels:
        label = (label or "").strip().lower()
        if len(label) < 3:
            continue
        mentions += text.count(label)
    return min(0.15, mentions * 0.04)


def _incumbent_penalty(url: str) -> float:
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    for inc in INCUMBENT_HOSTS:
        if host == inc or host.endswith("." + inc):
            return INCUMBENT_PENALTY
    return 0.0


def _exploratory_combined(doc_tfidf: float, field_tfidf: float, doc_jaccard: float) -> float:
    return 0.50 * doc_tfidf + 0.30 * field_tfidf + 0.20 * doc_jaccard


def bmc_core_profile_text(row: dict[str, str], deck_id: str | None = None) -> str:
    """Segment + value proposition only — what the company does."""
    label = deck_id or (row.get("deck_id") or "").strip()
    lines: list[str] = []
    if label:
        lines.append(f"deck_id: {label}")
    for field in CORE_FIELDS:
        value = (row.get(field) or "").strip()
        if value:
            lines.append(f"{field}: {value}")
    return "\n".join(lines)


PREFILTER_MIN_SCORE = 0.20
MIN_SNIPPET_CHARS = 80


def select_candidates_for_deep_scoring(
    ranked: list[tuple[object, float]],
    cache_domains: set[str],
    *,
    max_llm_peers: int,
    min_prefilter_score: float = PREFILTER_MIN_SCORE,
    snippets_by_url: dict[str, str] | None = None,
) -> list[object]:
    """Pick finalists for full BMC extraction (LLM).

  1. Cached domains — score from cache, no LLM.
  2. Top ``max_llm_peers`` by Stage-A prefilter among the rest.
     Skips URLs below ``min_prefilter_score`` or with too-short snippets.
    """
    selected: list[object] = []
    seen_domains: set[str] = set()
    snippets = snippets_by_url or {}

    def domain_of(cand: object) -> str:
        from support.peer_bmc_cache import peer_domain_key

        return peer_domain_key(getattr(cand, "url", ""))

    def add(cand: object) -> None:
        domain = domain_of(cand)
        if not domain or domain in seen_domains:
            return
        seen_domains.add(domain)
        selected.append(cand)

    for cand, _score in ranked:
        if domain_of(cand) in cache_domains:
            add(cand)

    llm_slots = 0
    for cand, score in ranked:
        if domain_of(cand) in seen_domains:
            continue
        if llm_slots >= max_llm_peers:
            break
        url = getattr(cand, "url", "")
        snippet = (snippets.get(url) or "").strip()
        if score < min_prefilter_score:
            continue
        if snippet and len(snippet) < MIN_SNIPPET_CHARS:
            continue
        add(cand)
        llm_slots += 1

    return selected


def _snippet_penalties(peer_url: str) -> float:
    return _host_penalty(peer_url) + _incumbent_penalty(peer_url)


def batch_snippet_prefilter_scores(
    target_row: dict[str, str],
    candidates: list[tuple[str, str]],
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> list[float]:
    """Stage-A scores: target core BMC vs homepage snippets (no LLM).

    ``candidates`` is a list of (peer_url, homepage_snippet).
    """
    target_core = bmc_core_profile_text(target_row)
    if not target_core.strip() or not candidates:
        return [0.0] * len(candidates)

    pred_texts = [
        f"peer homepage:\n{(snippet or '')[:3000]}" if snippet else ""
        for _, snippet in candidates
    ]
    gt_texts = [target_core] * len(candidates)
    sims, _ = batch_cosine_similarity(gt_texts, pred_texts, model_name=embed_model)

    scores: list[float] = []
    for sim, (url, snippet) in zip(sims, candidates, strict=True):
        if not snippet.strip():
            scores.append(0.0)
            continue
        adjusted = max(0.0, float(sim) - _snippet_penalties(url))
        scores.append(round(adjusted, 4))
    return scores


def score_peer(
    target_profile: str,
    target_row: dict[str, str],
    peer_profile: str,
    peer_row: dict[str, str],
    *,
    peer_url: str = "",
    target_labels: list[str] | None = None,
    use_embeddings: bool = True,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> dict[str, float | str]:
    """Score peer vs target. ``rank_score`` = core embed sim minus penalties."""
    target_core = bmc_core_profile_text(target_row)
    peer_core = bmc_core_profile_text(peer_row)

    doc_tfidf = _fit_cosine([target_profile, peer_profile])[0] if target_profile.strip() else 0.0
    doc_jaccard = jaccard_combined(target_profile, peer_profile)

    embed_sim = 0.0
    core_embed_sim = 0.0
    rank_method = "tfidf_cosine"
    rank_score = doc_tfidf

    if use_embeddings and target_core.strip() and peer_core.strip():
        raw_core, method = pair_cosine_similarity(
            target_core, peer_core, model_name=embed_model
        )
        if method != "tfidf_fallback":
            core_embed_sim = raw_core
            rank_score = core_embed_sim
            rank_method = embed_model
            if target_profile.strip() and peer_profile.strip():
                raw_full, _ = pair_cosine_similarity(
                    target_profile, peer_profile, model_name=embed_model
                )
                embed_sim = raw_full

    field_tfidf_scores: list[float] = []
    field_jaccard_scores: list[float] = []
    for field in BMC_FIELDS:
        t_line = _field_line(field, target_row)
        p_line = _field_line(field, peer_row)
        if not t_line and not p_line:
            continue
        if not t_line or not p_line:
            field_tfidf_scores.append(0.0)
            field_jaccard_scores.append(0.0)
            continue
        field_tfidf_scores.append(_fit_cosine([t_line, p_line])[0])
        field_jaccard_scores.append(jaccard_combined(t_line, p_line))

    field_tfidf = sum(field_tfidf_scores) / len(field_tfidf_scores) if field_tfidf_scores else 0.0
    field_jaccard = sum(field_jaccard_scores) / len(field_jaccard_scores) if field_jaccard_scores else 0.0

    labels = target_labels or []
    host_pen = _host_penalty(peer_url)
    mention_pen = _target_mention_penalty(peer_profile, labels)
    inc_pen = _incumbent_penalty(peer_url)
    combined = _exploratory_combined(doc_tfidf, field_tfidf, doc_jaccard)

    adjusted = float(rank_score) - host_pen - mention_pen - inc_pen
    adjusted = max(0.0, round(adjusted, 4))

    return {
        "rank_score": adjusted,
        "rank_method": rank_method,
        "core_embed_sim": round(core_embed_sim, 4),
        "tfidf_cosine": round(doc_tfidf, 4),
        "tfidf_fieldwise": round(field_tfidf, 4),
        "jaccard_combined": round(doc_jaccard, 4),
        "jaccard_fieldwise": round(field_jaccard, 4),
        "embed_sim": round(embed_sim, 4),
        "host_penalty": round(host_pen, 4),
        "mention_penalty": round(mention_pen, 4),
        "incumbent_penalty": round(inc_pen, 4),
        "combined_score": round(combined, 4),
    }
