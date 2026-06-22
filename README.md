# Master Thesis — BMC Pipeline from Pitch Decks

**Thesis:** *Structuring Early-Stage Deal Flow: An Evidence-Grounded BMC Pipeline from Pitch Decks*  
**Author:** Constantin von Krogh · BITM, University of Amsterdam

This repository is the **reproducible research artifact** for the thesis: a local-LLM pipeline that turns pitch-deck PDFs into structured Business Model Canvas (BMC) profiles, enriches missing fields from public websites, and finds similar startups. Everything runs on your machine via **Ollama** — no cloud LLM API required.

The repo ships with the **50-deck evaluation corpus**, **deck-only ground truth**, **frozen pipeline outputs**, **evaluation summaries**, and **inter-rater agreement data** (Fleiss' κ = 0.78 on the ten-deck pilot).

---

## Repository layout

```
├── README.md
├── requirements.txt
├── .gitignore
│
├── modules/
│   ├── pipeline/           # Modules 01–04 + model_selection.py
│   ├── support/            # Shared helpers (LLM, CSV, web fetch, paths)
│   └── eval/               # Evaluation scripts
│
├── data/
│   ├── pitch_decks/        # 50 pitch-deck PDFs (input corpus)
│   └── gt/
│       └── gt_pd_bmc_50.csv   # Deck-only ground truth (Module 02 eval)
│
├── docs/
│   └── kappa/              # Inter-rater agreement (10-deck pilot)
│       ├── gt_bmc_k.csv        # Primary annotator
│       ├── bmc_jonas_k.csv     # Second annotator
│       ├── bmc_max_k.csv       # Third annotator
│       └── kappa.csv           # Fleiss' κ computation (κ ≈ 0.78)
│
├── output/                 # Frozen thesis run (Modules 01–04)
│   ├── module_01/          # slides.csv
│   ├── module_02/          # screening_bmc.csv
│   ├── module_03/          # websites.csv, enriched_bmc.csv
│   └── module_04/          # peer rankings, search queries, peer BMC cache
│
└── eval/                   # Metric summaries from the thesis run
    ├── module_02/
    ├── module_03/
    └── module_04/
```

---

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/)** — local LLM server (default model: `llama3.1:8b`)
- **Optional:** `tesseract` for OCR fallback in Module 01 (`brew install tesseract` on macOS)

---

## Setup

Clone the repo and create a virtual environment:

```bash
git clone https://github.com/cvonkrogh/Master_Thesis.git
cd Master_Thesis

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start Ollama and pull the default model:

```bash
ollama serve
ollama pull llama3.1:8b
```

Optional environment variables (defaults shown):

```bash
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_TIMEOUT=3600    # seconds per LLM request
```

---

## Pipeline overview

| Module | Script | Input → output | What it does |
|--------|--------|----------------|--------------|
| **01** | `01_pdf_to_slides.py` | PDF → `output/module_01/slides.csv` | Extract slide text (native + OCR fallback) |
| **02** | `02_bmc_extract.py` | slides → `output/module_02/screening_bmc.csv` | Deck-only BMC extraction (9 fields) |
| **03** | `03_enrich_bmc.py` | screening BMC → `output/module_03/enriched_bmc.csv` | Fill **empty** BMC cells from validated websites |
| **04** | `04_find_similar_startups.py` | enriched BMC → peer rankings | Web search + peer BMC extraction + embedding similarity |

Run from the repo root:

```bash
python modules/pipeline/01_pdf_to_slides.py
python modules/pipeline/02_bmc_extract.py
python modules/pipeline/03_enrich_bmc.py
python modules/pipeline/04_find_similar_startups.py --all
```

### Useful flags

**Module 02** — run on a subset of decks:

```bash
python modules/pipeline/02_bmc_extract.py --decks Aura,Macro
```

**Module 03** — same subset syntax:

```bash
python modules/pipeline/03_enrich_bmc.py --decks Palta,Sable
```

**Module 04** — single deck or full corpus; `--force` re-runs cached peers:

```bash
python modules/pipeline/04_find_similar_startups.py --deck Aura
python modules/pipeline/04_find_similar_startups.py --all --force
```

Additional Module 04 options: `--max-candidates 15`, `--max-llm-peers 5`.

**Naming note:** ground truth uses `Vision` for the Connectly deck (`Vision.pdf`; web search uses brand *Connectly*).

---

## Evaluation

After running the pipeline (or using the bundled frozen `output/`), evaluate each module:

```bash
# Module 02 — deck-only BMC vs ground truth
python modules/eval/evaluate_bmc.py

# Module 03 — web enrichment completeness and web lift
python modules/eval/evaluate_enriched_bmc.py

# Module 04 — build manual peer-relevance rubric template
python modules/eval/build_m04_rubric_template.py
```

| Module | What is measured | Ground truth / rubric |
|--------|------------------|------------------------|
| **02** | Fill precision/recall + lexical/embedding content similarity | `data/gt/gt_pd_bmc_50.csv` |
| **03** | 9/9 field completeness, web lift, deck-overwrite check | No per-field GT (completeness target) |
| **04** | Qualitative peer relevance (C/A/W/U rubric) | `eval/module_04/peer_relevance_rubric.csv` |

Evaluation outputs are written to `eval/module_02/`, `eval/module_03/`, and `eval/module_04/`. The repo also includes **frozen summaries** from the thesis run for direct inspection.

**Model benchmark** (5 pilot decks, optional):

```bash
python modules/pipeline/model_selection.py
```

---

## Inter-rater agreement (`docs/kappa/`)

Ground-truth annotation was validated on the **first ten decks** of the corpus (Palta, Aura, Bespoken_spirits, Jobox, Macro, Sable, Sharpist, Vision, morty, multus). Three independent coders applied the same explicit-only fill/empty rules:

| File | Description |
|------|-------------|
| `gt_bmc_k.csv` | Primary annotator (full BMC text per field) |
| `bmc_jonas_k.csv` | Second annotator |
| `bmc_max_k.csv` | Third annotator |
| `kappa.csv` | Binary fill/empty labels + Fleiss' κ math (κ ≈ 0.78) |

High κ confirms consistent **fill presence** under the protocol; it does not measure wording similarity (that is evaluated separately in Module 02).

---

## Frozen outputs vs re-running

The bundled `output/` and `eval/` folders contain the **thesis run** used in the written results. You can inspect these immediately without re-running anything.

Re-running Modules **03** and **04** may produce **slightly different results** because they depend on live web search and page content. Module **04** caches peer BMC profiles in `output/module_04/peer_bmc_cache.csv` to speed up subsequent runs.

---

## Citation

If you use this artifact, please cite the thesis:

> von Krogh, C. (2025). *Structuring Early-Stage Deal Flow: An Evidence-Grounded BMC Pipeline from Pitch Decks*. Master's thesis, University of Amsterdam.

---

## License

Academic research artifact. Pitch decks remain the property of their respective companies; included here solely for reproducibility under fair academic use.
