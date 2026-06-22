"""Semantic similarity for BMC eval — sentence-transformers with sklearn TF-IDF fallback."""

from __future__ import annotations

import sys

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

_model = None
_model_name: str | None = None

def _tfidf_pairwise_batch(gt_texts: list[str], pred_texts: list[str]) -> list[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    sims: list[float] = []
    for gt, pred in zip(gt_texts, pred_texts, strict=True):
        mat = TfidfVectorizer().fit_transform([gt, pred])
        sims.append(round(float(cosine_similarity(mat[0], mat[1])[0][0]), 3))
    return sims

def _sentence_transformer_batch(
    gt_texts: list[str],
    pred_texts: list[str],
    model_name: str,
) -> list[float]:
    global _model, _model_name
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for embedding similarity. "
            "Install: pip install sentence-transformers numpy"
        ) from e
    if _model is None or _model_name != model_name:
        _model = SentenceTransformer(model_name)
        _model_name = model_name
    gt_emb = _model.encode(gt_texts, normalize_embeddings=True, show_progress_bar=False)
    pred_emb = _model.encode(pred_texts, normalize_embeddings=True, show_progress_bar=False)
    return [round(float(a @ b), 3) for a, b in zip(gt_emb, pred_emb, strict=True)]

def pair_cosine_similarity(
    text_a: str,
    text_b: str,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> tuple[float, str]:
    """Single-pair cosine similarity (0–1). Returns (score, method_label)."""
    scores, method = batch_cosine_similarity([text_a], [text_b], model_name=model_name)
    return scores[0], method

def batch_cosine_similarity(
    gt_texts: list[str],
    pred_texts: list[str],
    model_name: str = DEFAULT_EMBED_MODEL,
) -> tuple[list[float], str]:
    """Cosine similarity per pair (0–1). Returns (scores, method_label)."""
    if len(gt_texts) != len(pred_texts):
        raise ValueError("gt_texts and pred_texts must have the same length")
    if not gt_texts:
        return [], "none"
    try:
        return _sentence_transformer_batch(gt_texts, pred_texts, model_name), model_name
    except Exception as e:
        print(
            f"[eval] WARNING: sentence-transformers failed ({e}); "
            "using TF-IDF cosine fallback (weaker semantic match).",
            file=sys.stderr,
        )
        return _tfidf_pairwise_batch(gt_texts, pred_texts), "tfidf_fallback"

def apply_embedding_similarity(
    results: list,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> str:
    """
    Set embed_sim on each result.
    Computed only when both GT and pred have text (TP); otherwise embed_sim stays None.
    Returns method label (model name or tfidf_fallback).
    """
    indices: list[int] = []
    gt_texts: list[str] = []
    pred_texts: list[str] = []
    for i, r in enumerate(results):
        if r.gt_filled and r.pred_filled:
            indices.append(i)
            gt_texts.append(r.gt.strip())
            pred_texts.append(r.pred.strip())

    sims, method = batch_cosine_similarity(gt_texts, pred_texts, model_name=model_name)
    for i, sim in zip(indices, sims, strict=True):
        results[i].embed_sim = sim
    return method
