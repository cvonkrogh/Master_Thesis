"""
Evaluate Module 02 BMC extraction vs pitch-deck ground truth (gt_pd_bmc_50.csv).

Deck-only AI output (screening_bmc.csv) is compared to human BMC labels from the
pitch deck only — not the full multi-source reference (gt_full_bmc.csv).

Usage:
    python modules/eval/evaluate_bmc.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from support.csv_bmc import load_gt_bmc_rows, load_screening_bmc_rows  # noqa: E402
from support.paths import (  # noqa: E402
    DEFAULT_EVAL_FILL_METRICS,
    DEFAULT_EVAL_MODULE_02,
    DEFAULT_GT_BMC_PD,
    resolve_screening_bmc,
)
from support.schema import BMC_FIELDS  # noqa: E402

# Content similarity thresholds (combined Jaccard + sequence ratio, 0–1).
SIM_HIGH_MATCH = 0.75
SIM_PARTIAL_MATCH = 0.35


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


def combined_similarity(a: str, b: str) -> float:
    return 0.5 * jaccard(a, b) + 0.5 * seq_ratio(a, b)


def auto_score(gt: str, pred: str) -> int:
    gt_s, pred_s = gt.strip(), pred.strip()
    if not gt_s and not pred_s:
        return 2
    if gt_s and not pred_s:
        return 0
    if not gt_s and pred_s:
        return 0
    sim = combined_similarity(gt_s, pred_s)
    if sim >= SIM_HIGH_MATCH:
        return 2
    if sim >= SIM_PARTIAL_MATCH:
        return 1
    return 0


@dataclass
class FieldResult:
    deck_id: str
    field: str
    gt: str
    pred: str
    gt_filled: bool
    pred_filled: bool
    fill_match: bool
    jaccard: float
    seq_ratio: float
    combined: float
    score_0_2: int
    embed_sim: float | None = None  # cosine similarity; only set when both filled (TP)


def compare(gt_rows: list[dict[str, str]], pred_by_deck: dict[str, dict[str, str]]) -> list[FieldResult]:
    results: list[FieldResult] = []
    for gt in gt_rows:
        deck_id = gt["deck_id"]
        pred = pred_by_deck.get(deck_id, {})
        for field in BMC_FIELDS:
            g = gt.get(field, "")
            p = pred.get(field, "")
            gf = bool(g.strip())
            pf = bool(p.strip())
            j = jaccard(g, p)
            s = seq_ratio(g, p)
            c = 0.5 * j + 0.5 * s
            results.append(
                FieldResult(
                    deck_id=deck_id,
                    field=field,
                    gt=g,
                    pred=p,
                    gt_filled=gf,
                    pred_filled=pf,
                    fill_match=gf == pf,
                    jaccard=round(j, 3),
                    seq_ratio=round(s, 3),
                    combined=round(c, 3),
                    score_0_2=auto_score(g, p),
                )
            )
    return results


def _lexical_score_gt_cell(r: FieldResult) -> float:
    """Lexical combined_sim for GT-filled aggregate (FN → 0)."""
    if r.gt_filled and r.pred_filled:
        return r.combined
    if r.gt_filled:
        return 0.0
    return 0.0


def _embed_score_gt_cell(r: FieldResult) -> float | None:
    """Embedding sim for GT-filled aggregate; None if embeddings not computed."""
    if not r.gt_filled:
        return None
    if r.gt_filled and r.pred_filled:
        return r.embed_sim if r.embed_sim is not None else 0.0
    return 0.0


def summarize(results: list[FieldResult]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    gt_filled = sum(1 for r in results if r.gt_filled)
    pred_filled = sum(1 for r in results if r.pred_filled)
    fill_match = sum(1 for r in results if r.fill_match)
    both_filled = sum(1 for r in results if r.gt_filled and r.pred_filled)
    both_empty = sum(1 for r in results if not r.gt_filled and not r.pred_filled)
    missed = sum(1 for r in results if r.gt_filled and not r.pred_filled)
    extra = sum(1 for r in results if not r.gt_filled and r.pred_filled)

    tp = both_filled
    tn = both_empty
    fp = extra
    fn = missed
    # Fill-status agreement only (empty vs non-empty). Not text/content correctness.
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )
    specificity = tn / (tn + fp) if (tn + fp) else None

    both = [r for r in results if r.gt_filled and r.pred_filled]
    gt_only = [r for r in results if r.gt_filled]
    content_sim_both = round(sum(r.combined for r in both) / len(both), 3) if both else None
    content_sim_gt = (
        round(sum(_lexical_score_gt_cell(r) for r in gt_only) / len(gt_only), 3)
        if gt_only
        else None
    )
    high_gt = sum(1 for r in gt_only if _lexical_score_gt_cell(r) >= SIM_HIGH_MATCH)
    partial_gt = sum(1 for r in gt_only if _lexical_score_gt_cell(r) >= SIM_PARTIAL_MATCH)

    has_embed = any(r.embed_sim is not None for r in results)
    embed_sim_both = None
    embed_sim_gt = None
    embed_high_rate = None
    embed_model = None
    if has_embed:
        both_embed = [r.embed_sim for r in both if r.embed_sim is not None]
        embed_sim_both = (
            round(sum(both_embed) / len(both_embed), 3) if both_embed else None
        )
        embed_vals = [_embed_score_gt_cell(r) for r in gt_only]
        embed_vals = [v for v in embed_vals if v is not None]
        embed_sim_gt = round(sum(embed_vals) / len(embed_vals), 3) if embed_vals else None
        embed_high_rate = (
            round(
                sum(
                    1
                    for r in gt_only
                    if r.gt_filled
                    and r.pred_filled
                    and (r.embed_sim or 0) >= SIM_HIGH_MATCH
                )
                / len(gt_only),
                3,
            )
            if gt_only
            else None
        )
    content_high_match_rate = (
        round(high_gt / len(gt_only), 3) if gt_only else None
    )
    content_partial_match_rate = (
        round(partial_gt / len(gt_only), 3) if gt_only else None
    )
    scores = [r.score_0_2 for r in results]

    by_field: dict[str, dict] = {}
    for field in BMC_FIELDS:
        fr = [r for r in results if r.field == field]
        bf = [r for r in fr if r.gt_filled and r.pred_filled]
        by_field[field] = {
            "fill_match_rate": round(sum(r.fill_match for r in fr) / len(fr), 3),
            "gt_filled": sum(1 for r in fr if r.gt_filled),
            "pred_filled": sum(1 for r in fr if r.pred_filled),
            "mean_combined_sim_when_both_filled": (
                round(sum(r.combined for r in bf) / len(bf), 3) if bf else None
            ),
            "mean_score_0_2": round(sum(r.score_0_2 for r in fr) / len(fr), 3),
        }

    by_deck: dict[str, dict] = {}
    for gt_deck in {r.deck_id for r in results}:
        dr = [r for r in results if r.deck_id == gt_deck]
        bf = [r for r in dr if r.gt_filled and r.pred_filled]
        by_deck[gt_deck] = {
            "fill_match_rate": round(sum(r.fill_match for r in dr) / len(dr), 3),
            "gt_filled": sum(1 for r in dr if r.gt_filled),
            "pred_filled": sum(1 for r in dr if r.pred_filled),
            "mean_combined_sim_when_both_filled": (
                round(sum(r.combined for r in bf) / len(bf), 3) if bf else None
            ),
            "mean_score_0_2": round(sum(r.score_0_2 for r in dr) / len(dr), 3),
        }

    return {
        "n_decks": len(by_deck),
        "n_bmc_fields": len(BMC_FIELDS),
        "n_comparisons": n,
        "fill_agreement": {
            "fill_match_rate": round(fill_match / n, 3),
            "both_filled": both_filled,
            "both_empty": both_empty,
            "missed_gt_filled_pred_empty": missed,
            "extra_gt_empty_pred_filled": extra,
        },
        "confusion_matrix_fill": {
            "positive_class": "field_filled",
            "note": "Rates are fractions 0-1. Metrics describe fill status only, not text match.",
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
            "f1": round(f1, 3) if f1 is not None else None,
            "specificity": round(specificity, 3) if specificity is not None else None,
        },
        "content_score_0_1": round(sum(scores) / n / 2, 3),
        "gt_fill_rate": round(gt_filled / n, 3),
        "pred_fill_rate": round(pred_filled / n, 3),
        "content_metrics": {
            "note": (
                "Lexical: 0.5×Jaccard + 0.5×sequence ratio. "
                f"High ≥{SIM_HIGH_MATCH}, partial ≥{SIM_PARTIAL_MATCH}. "
                "embed_sim is cosine similarity (sentence-transformers), "
                "computed only when both GT and pred have text (TP)."
            ),
            "mean_lexical_sim_gt_filled": content_sim_gt,
            "mean_lexical_sim_both_filled": content_sim_both,
            "high_match_rate_gt_filled": content_high_match_rate,
            "partial_match_rate_gt_filled": content_partial_match_rate,
            "mean_embedding_sim_gt_filled": embed_sim_gt,
            "mean_embedding_sim_both_filled": embed_sim_both,
            "embedding_high_match_rate_gt_filled": embed_high_rate,
            "embeddings_computed": has_embed,
            "mean_similarity_all_cells": round(sum(r.combined for r in results) / n, 3),
        },
        "embedding_similarity_when_both_filled": embed_sim_both,
        "embedding_similarity_when_gt_filled": embed_sim_gt,
        "embedding_high_match_rate_gt_filled": embed_high_rate,
        "content_similarity_when_both_filled": content_sim_both,
        "content_similarity_when_gt_filled": content_sim_gt,
        "content_high_match_rate_gt_filled": content_high_match_rate,
        "content_partial_match_rate_gt_filled": content_partial_match_rate,
        "mean_combined_similarity_all_cells": round(sum(r.combined for r in results) / n, 3),
        "mean_score_0_2": round(sum(scores) / n, 3),
        "pct_score_2": round(100 * sum(1 for s in scores if s == 2) / n, 1),
        "pct_score_1": round(100 * sum(1 for s in scores if s == 1) / n, 1),
        "pct_score_0": round(100 * sum(1 for s in scores if s == 0) / n, 1),
        "by_field": by_field,
        "by_deck": by_deck,
        "max_bmc_cells": n,
        "gt_cells_filled": gt_filled,
        "pred_cells_filled": pred_filled,
        "fill_match_cells": fill_match,
    }


FILL_METRICS_COLUMNS = [
    "module",
    "gt_path",
    "pred_path",
    "max_bmc_cells",
    "gt_cells_filled",
    "pred_cells_filled",
    "pred_cells_filled_of_max",
    "fill_match_cells",
    "fill_match_of_max",
    "TP",
    "TN",
    "FP",
    "FN",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "content_sim_gt_filled",
    "content_sim_both_filled",
    "content_high_match_rate",
    "content_partial_match_rate",
    "embed_sim_gt_filled",
    "embed_sim_both_filled",
    "embed_high_match_rate",
    "mean_score_0_1",
]

# Written as strings (e.g. "0.489") so Excel does not drop the leading "0."
RATE_METRIC_COLUMNS = frozenset(
    {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "content_sim_gt_filled",
        "content_sim_both_filled",
        "content_high_match_rate",
        "content_partial_match_rate",
        "embed_sim_gt_filled",
        "embed_sim_both_filled",
        "embed_high_match_rate",
        "mean_score_0_1",
    }
)


def _fmt_rate_0_1(value: float | int | str | None) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.3f}"


def _fill_metrics_row(
    module: str,
    gt_path: Path,
    pred_path: Path,
    summary: dict,
) -> dict[str, str | int | float | None]:
    n = summary["max_bmc_cells"]
    cm = summary["confusion_matrix_fill"]
    pred_n = summary["pred_cells_filled"]
    gt_n = summary["gt_cells_filled"]
    match_n = summary["fill_match_cells"]
    return {
        "module": module,
        "gt_path": str(gt_path),
        "pred_path": str(pred_path),
        "max_bmc_cells": n,
        "gt_cells_filled": gt_n,
        "pred_cells_filled": pred_n,
        "pred_cells_filled_of_max": f"{pred_n}/{n}",
        "fill_match_cells": match_n,
        "fill_match_of_max": f"{match_n}/{n}",
        "TP": cm["TP"],
        "TN": cm["TN"],
        "FP": cm["FP"],
        "FN": cm["FN"],
        "accuracy": _fmt_rate_0_1(cm["accuracy"]),
        "precision": _fmt_rate_0_1(cm["precision"]),
        "recall": _fmt_rate_0_1(cm["recall"]),
        "f1": _fmt_rate_0_1(cm["f1"]),
        "content_sim_gt_filled": _fmt_rate_0_1(
            summary.get("content_similarity_when_gt_filled")
        ),
        "content_sim_both_filled": _fmt_rate_0_1(
            summary.get("content_similarity_when_both_filled")
        ),
        "content_high_match_rate": _fmt_rate_0_1(
            summary.get("content_high_match_rate_gt_filled")
        ),
        "content_partial_match_rate": _fmt_rate_0_1(
            summary.get("content_partial_match_rate_gt_filled")
        ),
        "embed_sim_gt_filled": _fmt_rate_0_1(
            summary.get("embedding_similarity_when_gt_filled")
        ),
        "embed_sim_both_filled": _fmt_rate_0_1(
            summary.get("embedding_similarity_when_both_filled")
        ),
        "embed_high_match_rate": _fmt_rate_0_1(
            summary.get("embedding_high_match_rate_gt_filled")
        ),
        "mean_score_0_1": _fmt_rate_0_1(summary["content_score_0_1"]),
    }


def write_fill_metrics_csv(
    path: Path,
    module: str,
    gt_path: Path,
    pred_path: Path,
    summary: dict,
) -> None:
    """Upsert one module row into eval/bmc_fill_metrics.csv (comparison table)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = _fill_metrics_row(module, gt_path, pred_path, summary)
    existing: list[dict[str, str | int | float | None]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("module") != module:
                    existing.append(row)
    rows = existing + [new_row]
    rows.sort(key=lambda r: str(r.get("module", "")))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FILL_METRICS_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = dict(row)
            for col in RATE_METRIC_COLUMNS:
                if col in out and out[col] not in (None, ""):
                    out[col] = _fmt_rate_0_1(out[col])
            w.writerow(out)


def run_evaluation(
    gt_path: Path,
    pred_path: Path,
    out_dir: Path,
    *,
    module: str,
    comparison_name: str = "bmc_field_comparison.csv",
    summary_name: str = "bmc_summary.json",
    fill_metrics_name: str = "bmc_fill_metrics.csv",
    module_label: str = "Module 02 deck-only BMC",
    use_embeddings: bool = True,
    embed_model: str = "all-MiniLM-L6-v2",
) -> int:
    if not gt_path.exists():
        print(f"[eval] GT not found: {gt_path}", file=sys.stderr)
        return 1
    if not pred_path.exists():
        print(f"[eval] Predictions not found: {pred_path}", file=sys.stderr)
        return 1

    gt_rows = load_gt_bmc_rows(gt_path)

    pred_rows = load_screening_bmc_rows(pred_path)
    pred_by_deck = {r["deck_id"]: r for r in pred_rows}

    results = compare(gt_rows, pred_by_deck)
    embed_model_used: str | None = None
    if use_embeddings:
        try:
            from eval.embedding_similarity import apply_embedding_similarity

            embed_model_used = apply_embedding_similarity(results, model_name=embed_model)
            print(
                f"[eval] Embeddings: {embed_model_used} "
                f"(cosine sim on TP cells only — both GT and AI have text)"
            )
        except ImportError as e:
            print(f"[eval] WARNING: skipping embeddings — {e}", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = out_dir / comparison_name
    with comparison_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "deck_id",
                "field",
                "gt_filled",
                "pred_filled",
                "fill_match",
                "gt",
                "pred",
                "jaccard",
                "seq_ratio",
                "combined_sim",
                "embed_sim",
                "score_0_2",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.deck_id,
                    r.field,
                    r.gt_filled,
                    r.pred_filled,
                    r.fill_match,
                    r.gt,
                    r.pred,
                    r.jaccard,
                    r.seq_ratio,
                    r.combined,
                    "" if r.embed_sim is None else r.embed_sim,
                    r.score_0_2,
                ]
            )

    summary = summarize(results)
    if embed_model_used:
        summary["embed_model"] = embed_model_used
    summary_path = out_dir / summary_name
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    fill_metrics_path = (
        DEFAULT_EVAL_FILL_METRICS
        if fill_metrics_name == "bmc_fill_metrics.csv"
        else out_dir / fill_metrics_name
    )
    write_fill_metrics_csv(fill_metrics_path, module, gt_path, pred_path, summary)

    fa = summary["fill_agreement"]
    cm = summary["confusion_matrix_fill"]
    n = summary["max_bmc_cells"]
    print(f"[eval] {module_label}")
    print(f"[eval] GT:         {gt_path}")
    print(f"[eval] Pred:       {pred_path}")
    print(f"[eval] Comparison: {comparison_path}")
    print(f"[eval] Summary:    {summary_path}")
    print(f"[eval] Fill metrics: {fill_metrics_path}")
    print()
    print(
        f"[eval] Cells filled: {summary['pred_cells_filled']}/{n}  "
        f"(GT: {summary['gt_cells_filled']}/{n}, fill match: {summary['fill_match_cells']}/{n})"
    )
    print()
    print(f"[eval] LAYER 1 — Fill status only (filled vs empty, {n} cells; rates 0–1)")
    print(f"       TP (both filled):     {cm['TP']}")
    print(f"       TN (both empty):      {cm['TN']}")
    print(f"       FP (AI filled, GT empty): {cm['FP']}")
    print(f"       FN (GT filled, AI empty): {cm['FN']}")
    print(
        f"       Accuracy (fill): {cm['accuracy']:.3f}  "
        f"(= {cm['accuracy']:.1%}, same as fill_match_cells/{n})"
    )
    if cm["precision"] is not None:
        print(
            f"       Precision:    {cm['precision']:.3f}  "
            f"(when AI fills, GT also filled; no extra fills if FP=0)"
        )
    if cm["recall"] is not None:
        print(
            f"       Recall:       {cm['recall']:.3f}  "
            f"(= TP / GT filled = {cm['TP']}/{summary['gt_cells_filled']})"
        )
    if cm["f1"] is not None:
        print(f"       F1 (fill):    {cm['f1']:.3f}")
    print()
    print("[eval] LAYER 1 — Fill agreement (same fields filled?)")
    print(f"       Fill match rate:     {fa['fill_match_rate']:.1%}")
    print(f"       Both filled:         {fa['both_filled']}")
    print(f"       Both empty:          {fa['both_empty']}")
    print(f"       Missed (GT yes, AI no): {fa['missed_gt_filled_pred_empty']}")
    print(f"       Extra  (GT no, AI yes): {fa['extra_gt_empty_pred_filled']}")
    print()
    print("[eval] LAYER 2 — Text similarity (0–1; lexical on all GT-filled, embed on TP only)")
    cgt = summary.get("content_similarity_when_gt_filled")
    cboth = summary.get("content_similarity_when_both_filled")
    chigh = summary.get("content_high_match_rate_gt_filled")
    cpart = summary.get("content_partial_match_rate_gt_filled")
    print("       Lexical (word/char overlap):")
    print(
        f"         Mean (GT filled):  {cgt if cgt is not None else 'n/a'}  "
        f"(FN/missed AI → 0)"
    )
    print(
        f"         Mean (TP only):    {cboth if cboth is not None else 'n/a'}  "
        f"(both have text)"
    )
    print(
        f"         High/partial rate: {chigh if chigh is not None else 'n/a'} / "
        f"{cpart if cpart is not None else 'n/a'}"
    )
    egt = summary.get("embedding_similarity_when_gt_filled")
    eboth = summary.get("embedding_similarity_when_both_filled")
    ehigh = summary.get("embedding_high_match_rate_gt_filled")
    if eboth is not None or egt is not None:
        print("       Embedding (semantic, sentence-transformers):")
        print(
            f"         Mean (GT filled):  {egt if egt is not None else 'n/a'}  "
            f"(FN → 0; sim computed only on TP)"
        )
        print(f"         Mean (TP only):    {eboth if eboth is not None else 'n/a'}")
        print(f"         High match rate:   {ehigh if ehigh is not None else 'n/a'}")
    print(f"       Mean score (0–1):      {summary['content_score_0_1']:.3f}")
    print(
        f"       Per-cell detail:       {comparison_path.name} "
        f"(combined_sim, embed_sim, score_0_2)"
    )
    print(
        f"       Score mix (90 cells):  2={summary['pct_score_2']:.0f}% | "
        f"1={summary['pct_score_1']:.0f}% | 0={summary['pct_score_0']:.0f}%"
    )
    gt_decks = {g["deck_id"] for g in gt_rows}
    missing = sorted(gt_decks - set(pred_by_deck))
    if missing:
        print(f"[eval] WARNING: no row in predictions for: {', '.join(missing)}", file=sys.stderr)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Module 02 deck-only BMC vs pitch-deck GT (gt_bmc_pd.csv)."
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=DEFAULT_GT_BMC_PD,
        help="Pitch-deck BMC ground truth (default: data/gt/gt_pd_bmc_50.csv).",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="AI BMC CSV (default: output/module_02/screening_bmc.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EVAL_MODULE_02,
        help="Eval output directory (default: eval/module_02).",
    )
    parser.add_argument(
        "--comparison-out",
        type=str,
        default="bmc_field_comparison.csv",
        help="Comparison CSV filename inside --out.",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="bmc_summary.json",
        help="Summary JSON filename inside --out.",
    )
    parser.add_argument(
        "--fill-metrics-out",
        type=str,
        default="bmc_fill_metrics.csv",
        help="Combined fill-metrics CSV (one row per module, upserted).",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip sentence-transformer semantic similarity.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="sentence-transformers model name (default: all-MiniLM-L6-v2).",
    )
    args = parser.parse_args(argv)
    pred_path = args.pred or resolve_screening_bmc()

    return run_evaluation(
        args.gt,
        pred_path,
        args.out,
        module="module_02",
        comparison_name=args.comparison_out,
        summary_name=args.summary_out,
        fill_metrics_name=args.fill_metrics_out,
        module_label="Module 02 deck-only BMC vs pitch-deck GT",
        use_embeddings=not args.no_embeddings,
        embed_model=args.embed_model,
    )


if __name__ == "__main__":
    raise SystemExit(main())
