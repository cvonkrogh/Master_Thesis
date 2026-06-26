# Master Thesis — BMC Pipeline from Pitch Decks

**Thesis:** *From Pitch Decks to Structured Profiles: An AI-Enabled Pipeline for Startup Screening*  
Constantin von Krogh · BITM, University of Amsterdam

This repository is the **reproducible research artifact** for the thesis. It contains a local-LLM pipeline that turns pitch-deck PDFs into structured Business Model Canvas (BMC) profiles, enriches missing fields from public websites, and finds similar startups. Everything runs via **Ollama** on your machine — no cloud LLM API.

The repo ships with the full **50-deck corpus**, **deck-only ground truth**, **frozen pipeline outputs**, **evaluation summaries**, and **inter-rater agreement data** from the annotation pilot (Fleiss' κ ≈ 0.78).

---

## What's in this repo

```
├── modules/
│   ├── pipeline/           # Modules 01–04 + model_selection.py
│   ├── support/            # LLM client, CSV helpers, web fetch, paths
│   └── eval/               # evaluate_bmc, evaluate_enriched_bmc, M04 rubric
│
├── data/
│   ├── pitch_decks/        # 50 pitch-deck PDFs
│   └── gt/
│       └── gt_pd_bmc_50.csv   # Deck-only ground truth (Module 02)
│
├── docs/kappa/             # Inter-rater agreement (10-deck pilot)
│   ├── gt_bmc_k.csv            Primary annotator
│   ├── bmc_jonas_k.csv         Second annotator
│   ├── bmc_max_k.csv           Third annotator
│   └── kappa.csv               Fleiss' κ computation
│
├── output/module_01–04/    # Frozen thesis run
└── eval/module_02–04/      # Metric summaries
```

**Module 01** extracts slide text from PDFs (native text + OCR fallback) → `output/module_01/slides.csv`.

**Module 02** extracts a deck-only BMC (nine fields) via local LLM → `output/module_02/screening_bmc.csv`. Evaluated against `data/gt/gt_pd_bmc_50.csv`.

**Module 03** discovers validated company websites and fills **empty** BMC cells only → `output/module_03/enriched_bmc.csv`.

**Module 04** searches for peer startups, extracts their BMC from the web, and ranks by embedding similarity → `output/module_04/` (peer rankings, search queries, peer BMC cache).

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com/). Optional: `brew install tesseract` for OCR fallback in Module 01.

```bash
git clone https://github.com/cvonkrogh/Master_Thesis.git
cd Master_Thesis

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

ollama serve
ollama pull llama3.1:8b            # default; override with OLLAMA_MODEL
```

Optional env vars: `OLLAMA_HOST` (default `http://127.0.0.1:11434`), `OLLAMA_MODEL`, `OLLAMA_TIMEOUT` (default 1800 s).

---

## Run the pipeline

Run from the repo root, in order:

```bash
python modules/pipeline/01_pdf_to_slides.py
python modules/pipeline/02_bmc_extract.py
python modules/pipeline/03_enrich_bmc.py
python modules/pipeline/04_find_similar_startups.py --all
```

| Module | Input → output | Notes |
|--------|----------------|-------|
| **01** | PDF → `slides.csv` | Native text + OCR fallback |
| **02** | slides → `screening_bmc.csv` | Deck-only BMC; `--with-pdf` optional |
| **03** | screening BMC → `enriched_bmc.csv` | Fills **empty** cells only from validated websites |
| **04** | enriched BMC → ranked peers | Two-stage scoring + peer BMC cache |

**Subset runs:** `--decks Aura,Macro` on Modules 02–03. Module 04: `--deck Aura` or `--all`; `--force` to redo cached peers. Other flags: `--max-candidates 15`, `--max-llm-peers 5`.

**Naming:** ground truth uses `Vision` for the Connectly deck (`Vision.pdf`; web search uses brand *Connectly*).

---

## Evaluate

```bash
python modules/eval/evaluate_bmc.py              # M02 vs gt_pd_bmc_50.csv
python modules/eval/evaluate_enriched_bmc.py     # M03 completeness + web lift
```

- **Module 02:** fill precision/recall + lexical/embedding content similarity vs deck-only ground truth.
- **Module 03:** 9/9 field completeness, web lift, deck-overwrite check — no per-field GT.
- **Module 04:** automated diligence labels in `output/module_04/vc_diligence_summary.csv`; manual C/A/W/U coding reported in the thesis (Section 4.4.3).

Model benchmark (5 pilot decks): `python modules/pipeline/model_selection.py`

Results land in `eval/module_02/`, `eval/module_03/`, and `output/module_04/`. The bundled folders contain **frozen summaries** from the thesis run.

---

## Inter-rater agreement (`docs/kappa/`)

Before completing the full 50-deck ground truth, annotation rules were validated on the **first ten corpus decks** (Palta, Aura, Bespoken_spirits, Jobox, Macro, Sable, Sharpist, Vision, morty, multus). Three coders independently annotated the same decks under explicit-only rules:

- `gt_bmc_k.csv`, `bmc_jonas_k.csv`, `bmc_max_k.csv` — full BMC text per annotator
- `kappa.csv` — binary fill/empty labels and Fleiss' κ math (**κ ≈ 0.78**, substantial agreement)

κ measures whether coders agree a field should be **filled**, not whether they wrote similar text. Content similarity is evaluated separately in Module 02.

---

## Frozen outputs vs re-running

You can inspect `output/` and `eval/` immediately without re-running anything — these are the results reported in the thesis.

Re-running Modules **03** and **04** may differ slightly (live web search and page content). Module 04 reuses peer profiles via `output/module_04/peer_bmc_cache.csv`.

---

## Citation

> von Krogh, C. (2026). *From Pitch Decks to Structured Profiles: An AI-Enabled Pipeline for Startup Screening*. Master's thesis, University of Amsterdam.

Pitch decks remain the property of their respective companies; included here for academic reproducibility.
