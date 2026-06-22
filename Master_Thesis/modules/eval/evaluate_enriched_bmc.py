"""
Evaluate Module 03 enriched BMC — web-fill completeness.

Module 03 is meant to fill gaps using public website data. In principle the web
can support all nine BMC fields, so this eval treats a *full* BMC (9/9 fields
non-empty per deck) as the target — not gt_full_bmc.csv (legacy 10-deck reference).

Compares:
  - output/module_03/enriched_bmc.csv   (Module 03 output)
  - output/module_02/screening_bmc.csv  (deck-only baseline, for web lift)

Usage:
    python modules/eval/evaluate_enriched_bmc.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from support.csv_bmc import bmc_by_deck_from_csv, load_screening_bmc_rows  # noqa: E402
from support.paths import (  # noqa: E402
    DEFAULT_EVAL_MODULE_03,
    DEFAULT_SCREENING_BMC,
    resolve_enriched_bmc,
    resolve_screening_bmc,
)
from support.schema import BMC_FIELDS  # noqa: E402

COMPLETENESS_NOTE = (
    "Target: all 9 BMC fields filled per deck from deck + web (full BMC). "
    "No per-field human GT — completeness = share of cells filled."
)


def _filled(row: dict[str, str], field: str) -> bool:
    return bool((row.get(field) or "").strip())


def _count_filled(row: dict[str, str]) -> int:
    return sum(1 for f in BMC_FIELDS if _filled(row, f))


def _empty_fields(row: dict[str, str]) -> list[str]:
    return [f for f in BMC_FIELDS if not _filled(row, f)]


def evaluate_completeness(
    enriched_rows: list[dict[str, str]],
    screening_by_deck: dict[str, dict[str, str]],
) -> tuple[list[dict], dict]:
    """Per-deck completeness + aggregate summary."""
    per_deck: list[dict] = []
    n_fields = len(BMC_FIELDS)

    total_cells = 0
    enriched_filled = 0
    screening_filled = 0
    web_lift_cells = 0
    screening_empty_cells = 0
    deck_preserved = 0
    deck_overwritten = 0
    full_bmc_decks = 0

    field_fill_m3: dict[str, int] = {f: 0 for f in BMC_FIELDS}
    field_lift: dict[str, int] = {f: 0 for f in BMC_FIELDS}

    for erow in enriched_rows:
        deck_id = (erow.get("deck_id") or "").strip()
        if not deck_id:
            continue
        srow = screening_by_deck.get(deck_id, {f: "" for f in BMC_FIELDS})

        m2_n = _count_filled(srow)
        m3_n = _count_filled(erow)
        lift = 0
        preserved = 0
        overwritten = 0

        for field in BMC_FIELDS:
            total_cells += 1
            s_f = _filled(srow, field)
            e_f = _filled(erow, field)
            if e_f:
                enriched_filled += 1
                field_fill_m3[field] += 1
            if s_f:
                screening_filled += 1
            if not s_f:
                screening_empty_cells += 1
                if e_f:
                    web_lift_cells += 1
                    field_lift[field] += 1
                    lift += 1
            elif e_f:
                preserved += 1
                if (srow.get(field) or "").strip() != (erow.get(field) or "").strip():
                    overwritten += 1
            elif s_f and not e_f:
                overwritten += 1

        deck_preserved += preserved
        deck_overwritten += overwritten
        if m3_n == n_fields:
            full_bmc_decks += 1

        per_deck.append(
            {
                "deck_id": deck_id,
                "m2_fields_filled": m2_n,
                "m3_fields_filled": m3_n,
                "web_lift_fields": lift,
                "m3_empty_fields": ",".join(_empty_fields(erow)),
                "completeness_rate": round(m3_n / n_fields, 3),
                "full_bmc": m3_n == n_fields,
            }
        )

    n_decks = len(per_deck)
    completeness_rate = enriched_filled / total_cells if total_cells else 0.0
    web_lift_rate = (
        web_lift_cells / screening_empty_cells if screening_empty_cells else None
    )
    m2_fill_rate = screening_filled / total_cells if total_cells else 0.0

    summary = {
        "eval_type": "web_fill_completeness",
        "note": COMPLETENESS_NOTE,
        "n_decks": n_decks,
        "n_fields": n_fields,
        "max_bmc_cells": total_cells,
        "expected_filled_cells": total_cells,
        "enriched_cells_filled": enriched_filled,
        "screening_cells_filled": screening_filled,
        "completeness_rate": round(completeness_rate, 3),
        "m2_fill_rate": round(m2_fill_rate, 3),
        "lift_from_m2": enriched_filled - screening_filled,
        "web_lift_cells": web_lift_cells,
        "web_lift_rate_of_m2_gaps": (
            round(web_lift_rate, 3) if web_lift_rate is not None else None
        ),
        "screening_empty_cells": screening_empty_cells,
        "full_bmc_decks": full_bmc_decks,
        "full_bmc_deck_rate": round(full_bmc_decks / n_decks, 3) if n_decks else 0.0,
        "deck_fields_preserved": deck_preserved,
        "deck_field_overwrites": deck_overwritten,
        "by_field_fill_rate_m3": {
            f: round(field_fill_m3[f] / n_decks, 3) if n_decks else 0.0
            for f in BMC_FIELDS
        },
        "by_field_web_lift": field_lift,
        "mean_fields_filled_m3": round(enriched_filled / n_decks, 2) if n_decks else 0.0,
        "mean_fields_filled_m2": round(screening_filled / n_decks, 2) if n_decks else 0.0,
    }
    return per_deck, summary


def _write_completeness_metrics_csv(path: Path, summary: dict, pred_path: Path) -> None:
    """Upsert module_03 row into eval/bmc_completeness_metrics.csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "module": "module_03",
        "pred_path": str(pred_path),
        "max_bmc_cells": summary["max_bmc_cells"],
        "enriched_cells_filled": summary["enriched_cells_filled"],
        "completeness_rate": f"{summary['completeness_rate']:.3f}",
        "full_bmc_decks": summary["full_bmc_decks"],
        "full_bmc_deck_rate": f"{summary['full_bmc_deck_rate']:.3f}",
        "m2_fill_rate": f"{summary['m2_fill_rate']:.3f}",
        "web_lift_cells": summary["web_lift_cells"],
        "web_lift_rate_of_m2_gaps": (
            f"{summary['web_lift_rate_of_m2_gaps']:.3f}"
            if summary["web_lift_rate_of_m2_gaps"] is not None
            else ""
        ),
        "mean_fields_filled_m3": summary["mean_fields_filled_m3"],
    }
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    existing = [r for r in existing if r.get("module") != "module_03"]
    existing.append({k: str(v) for k, v in row.items()})
    fieldnames = list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)


def run_completeness_eval(
    enriched_path: Path,
    screening_path: Path,
    out_dir: Path,
) -> int:
    enriched_rows = load_screening_bmc_rows(enriched_path)
    screening_by = bmc_by_deck_from_csv(screening_path)

    if not enriched_rows:
        print(f"[eval] No rows in {enriched_path}", file=sys.stderr)
        return 1

    per_deck, summary = evaluate_completeness(enriched_rows, screening_by)
    out_dir.mkdir(parents=True, exist_ok=True)

    deck_csv = out_dir / "enriched_bmc_completeness_by_deck.csv"
    deck_fields = [
        "deck_id",
        "m2_fields_filled",
        "m3_fields_filled",
        "web_lift_fields",
        "completeness_rate",
        "full_bmc",
        "m3_empty_fields",
    ]
    with deck_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=deck_fields)
        writer.writeheader()
        for row in per_deck:
            writer.writerow({k: row.get(k, "") for k in deck_fields})

    field_csv = out_dir / "enriched_bmc_field_fill_rates.csv"
    with field_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["field", "fill_rate_m3", "web_lift_count"]
        )
        writer.writeheader()
        for field in BMC_FIELDS:
            writer.writerow(
                {
                    "field": field,
                    "fill_rate_m3": summary["by_field_fill_rate_m3"][field],
                    "web_lift_count": summary["by_field_web_lift"][field],
                }
            )

    summary_path = out_dir / "enriched_bmc_completeness_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    metrics_path = Path("eval") / "bmc_completeness_metrics.csv"
    _write_completeness_metrics_csv(metrics_path, summary, enriched_path)

    n = summary["max_bmc_cells"]
    print("[eval] Module 03 — web-fill completeness (target: full 9-field BMC per deck)")
    print(f"[eval] Enriched:  {enriched_path}")
    print(f"[eval] Baseline:  {screening_path} (Module 02 deck-only)")
    print(f"[eval] By deck:   {deck_csv}")
    print(f"[eval] By field:  {field_csv}")
    print(f"[eval] Summary:   {summary_path}")
    print(f"[eval] Metrics:   {metrics_path}")
    print()
    print(f"[eval] Decks: {summary['n_decks']}  |  Cells: {summary['enriched_cells_filled']}/{n}")
    print(
        f"[eval] Completeness: {summary['completeness_rate']:.1%}  "
        f"(mean {summary['mean_fields_filled_m3']:.1f}/9 fields per deck)"
    )
    print(
        f"[eval] Full BMC (9/9): {summary['full_bmc_decks']}/{summary['n_decks']} decks "
        f"({summary['full_bmc_deck_rate']:.1%})"
    )
    print(
        f"[eval] Web lift from M2: +{summary['web_lift_cells']} cells "
        f"({summary['lift_from_m2']:+d} net vs M02)"
    )
    if summary["web_lift_rate_of_m2_gaps"] is not None:
        print(
            f"[eval] Filled {summary['web_lift_rate_of_m2_gaps']:.1%} of cells "
            f"that were empty after Module 02"
        )
    print(
        f"[eval] Deck preservation: {summary['deck_fields_preserved']} deck fields kept; "
        f"{summary['deck_field_overwrites']} overwrites/losses"
    )
    print()
    print("[eval] Hardest fields (lowest M3 fill rate):")
    ranked = sorted(
        summary["by_field_fill_rate_m3"].items(), key=lambda x: x[1]
    )
    for field, rate in ranked[:4]:
        print(f"       {field}: {rate:.1%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Module 03 enriched BMC — web-fill completeness."
    )
    parser.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="Enriched BMC CSV (default: output/module_03/enriched_bmc.csv).",
    )
    parser.add_argument(
        "--screening",
        type=Path,
        default=None,
        help="Module 02 baseline CSV (default: output/module_02/screening_bmc.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EVAL_MODULE_03,
        help="Eval output directory (default: eval/module_03).",
    )
    args = parser.parse_args(argv)
    enriched_path = args.pred or resolve_enriched_bmc()
    screening_path = args.screening or resolve_screening_bmc()

    if not enriched_path.exists():
        print(f"[eval] Missing {enriched_path}. Run Module 03 first.", file=sys.stderr)
        return 1
    if not screening_path.exists():
        print(f"[eval] Missing {screening_path}. Run Module 02 first.", file=sys.stderr)
        return 1

    return run_completeness_eval(enriched_path, screening_path, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
