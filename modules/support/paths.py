"""Shared output paths for the screening pipeline (organized by module)."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_GT_DIR = Path("data/gt")
DEFAULT_GT_BMC_PD = DEFAULT_GT_DIR / "gt_pd_bmc_50.csv"
DEFAULT_GT_FULL_BMC = DEFAULT_GT_DIR / "gt_full_bmc.csv"
LEGACY_GT_FULL_BMC = DEFAULT_GT_DIR / "gt_full_bmc_1.csv"
DEFAULT_PDF_DIR = Path("data/pitch_decks")

def _resolve_output_dir() -> Path:
    """Output root: OUTPUT_DIR env, then ./output, then legacy data/output."""
    env = os.getenv("OUTPUT_DIR", "").strip()
    if env:
        return Path(env)
    if Path("output").exists():
        return Path("output")
    return Path("data/output")

DEFAULT_OUTPUT_DIR = _resolve_output_dir()

DEFAULT_MODULE_01_DIR = DEFAULT_OUTPUT_DIR / "module_01"
DEFAULT_MODULE_01_SLIDES_CSV = DEFAULT_MODULE_01_DIR / "slides.csv"

DEFAULT_MODULE_01_JSON = DEFAULT_MODULE_01_DIR / "json"

DEFAULT_MODULE_02_DIR = DEFAULT_OUTPUT_DIR / "module_02"
DEFAULT_SCREENING_BMC = DEFAULT_MODULE_02_DIR / "screening_bmc.csv"

DEFAULT_MODULE_03_DIR = DEFAULT_OUTPUT_DIR / "module_03"
DEFAULT_ENRICHED_BMC = DEFAULT_MODULE_03_DIR / "enriched_bmc.csv"
DEFAULT_WEBSITES_CSV = DEFAULT_MODULE_03_DIR / "websites.csv"

DEFAULT_MODULE_04_DIR = DEFAULT_OUTPUT_DIR / "module_04"
DEFAULT_PEERS_RANKED_CSV = DEFAULT_MODULE_04_DIR / "peers_ranked.csv"
DEFAULT_SIMILAR_TOP5_CSV = DEFAULT_MODULE_04_DIR / "similar_top5_all_decks.csv"
DEFAULT_VC_DILIGENCE_CSV = DEFAULT_MODULE_04_DIR / "vc_diligence_summary.csv"
DEFAULT_SEARCH_QUERIES_CSV = DEFAULT_MODULE_04_DIR / "search_queries.csv"
DEFAULT_PEER_BMC_CACHE_CSV = DEFAULT_MODULE_04_DIR / "peer_bmc_cache.csv"
DEFAULT_SIMILAR_DIR = DEFAULT_MODULE_04_DIR

LEGACY_JSON_DIR = DEFAULT_OUTPUT_DIR / "json"
LEGACY_SIMILAR_DIR = DEFAULT_OUTPUT_DIR / "similar"
LEGACY_MODULE_02_JSON = DEFAULT_MODULE_02_DIR / "json"
LEGACY_MODULE_03_JSON = DEFAULT_MODULE_03_DIR / "json"

def _prefer_existing(primary: Path, *fallbacks: Path) -> Path:
    if primary.exists():
        return primary
    for path in fallbacks:
        if path.exists():
            return path
    return primary

def resolve_screening_bmc() -> Path:
    return _prefer_existing(
        DEFAULT_SCREENING_BMC,
        DEFAULT_OUTPUT_DIR / "screening_bmc.csv",
    )

def resolve_enriched_bmc() -> Path:
    return _prefer_existing(
        DEFAULT_ENRICHED_BMC,
        DEFAULT_OUTPUT_DIR / "enriched_bmc.csv",
    )

def resolve_gt_full_bmc() -> Path:
    return _prefer_existing(
        DEFAULT_GT_FULL_BMC,
        LEGACY_GT_FULL_BMC,
    )

def resolve_websites_csv() -> Path:
    return _prefer_existing(
        DEFAULT_WEBSITES_CSV,
        DEFAULT_OUTPUT_DIR / "websites.csv",
    )

def resolve_module_01_slides() -> Path:
    return _prefer_existing(DEFAULT_MODULE_01_SLIDES_CSV)

def resolve_similar_dir() -> Path:
    return _prefer_existing(DEFAULT_MODULE_04_DIR, LEGACY_SIMILAR_DIR)

def _resolve_eval_dir() -> Path:
    env = os.getenv("EVAL_DIR", "").strip()
    if env:
        return Path(env)
    if Path("eval").exists():
        return Path("eval")
    return Path("data/eval")

DEFAULT_EVAL_DIR = _resolve_eval_dir()
DEFAULT_EVAL_MODULE_02 = DEFAULT_EVAL_DIR / "module_02"
DEFAULT_EVAL_MODULE_03 = DEFAULT_EVAL_DIR / "module_03"
DEFAULT_EVAL_MODULE_04 = DEFAULT_EVAL_DIR / "module_04"
DEFAULT_EVAL_FILL_METRICS = DEFAULT_EVAL_DIR / "bmc_fill_metrics.csv"
