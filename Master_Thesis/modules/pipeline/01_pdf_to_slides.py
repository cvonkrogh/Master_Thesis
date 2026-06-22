
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import fitz  


MIN_NATIVE_CHARS = 30
MIN_OCR_ALPHA_RATIO = 0.5  # OCR result must be ≥50% letters/digits, else treat as noise
OCR_DPI = 300
OCR_ENGINES = ("auto", "rapidocr", "tesseract")
DEFAULT_PDF_DIR = Path("data/pitch_decks")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from support.csv_bmc import canonical_deck_id  # noqa: E402
from support.paths import DEFAULT_MODULE_01_SLIDES_CSV  # noqa: E402
from support.slides_store import slide_row, write_slides_csv  # noqa: E402

# Markers that mean PyMuPDF returned glyphs it couldn't map to real characters.
# Most commonly seen with subset CFF/Type1 fonts that ship without a ToUnicode
# table, or when the producer munged the encoding.
_GARBLE_PATTERNS = (
    re.compile(r"\(cid:\d+\)"),
    re.compile(r"\ufffd"),
)


@dataclass
class Slide:
    page: int
    title: str
    body: str
    source: str
    char_count: int


def _clean_text(text: str) -> str:
    """Normalize whitespace while preserving line breaks."""
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _split_title_body(text: str) -> tuple[str, str]:
    """Use the first non-empty line as a rough title, the rest as body."""
    if not text:
        return "", ""
    parts = text.split("\n", 1)
    title = parts[0].strip()
    body = parts[1].strip() if len(parts) > 1 else ""
    return title, body


def _looks_garbled(text: str) -> bool:
    """Detect native text that's structurally junk (bad font encoding, etc.)."""
    if not text:
        return False
    for pat in _GARBLE_PATTERNS:
        if pat.search(text):
            return True
    # Heuristic: very long strings of letters with almost no spaces are usually
    # the result of a missing ToUnicode CMap (glyphs concatenated as one blob).
    if len(text) > 80:
        space_ratio = (text.count(" ") + text.count("\n")) / len(text)
        if space_ratio < 0.02:
            return True
    return False


def _alpha_ratio(text: str) -> float:
    """Fraction of characters that are letters or digits — a crude noise filter."""
    if not text:
        return 0.0
    keep = sum(1 for c in text if c.isalnum())
    return keep / len(text)


def _rasterize_page(page: "fitz.Page", dpi: int = OCR_DPI) -> bytes:
    """Render a PDF page to a PNG byte string."""
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return pix.tobytes("png")


@lru_cache(maxsize=1)
def _load_rapidocr():
    """Lazy-load the RapidOCR engine (heavy import + model load)."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:
        raise RuntimeError(
            "RapidOCR not installed. Install with: pip install rapidocr-onnxruntime"
        ) from e
    return RapidOCR()


def _ocr_page_rapidocr(page: "fitz.Page", dpi: int = OCR_DPI) -> str:
    """OCR a page with the pure-Python RapidOCR (ONNX) engine."""
    import numpy as np
    from PIL import Image

    engine = _load_rapidocr()
    img = Image.open(io.BytesIO(_rasterize_page(page, dpi))).convert("RGB")
    result, _ = engine(np.array(img))
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def _ocr_page_tesseract(page: "fitz.Page", dpi: int = OCR_DPI) -> str:
    """OCR a page with Tesseract (requires the `tesseract` binary)."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "Tesseract OCR deps missing. Install with: pip install pytesseract Pillow"
        ) from e

    img = Image.open(io.BytesIO(_rasterize_page(page, dpi)))
    try:
        return pytesseract.image_to_string(img)
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract binary not found. Install it with: brew install tesseract"
        ) from e


def _resolve_ocr_engine(preferred: str) -> str:
    """Pick an OCR engine, defaulting to RapidOCR when 'auto'."""
    if preferred == "auto":
        try:
            import rapidocr_onnxruntime 
            return "rapidocr"
        except ImportError:
            return "tesseract"
    if preferred not in OCR_ENGINES:
        raise ValueError(f"Unknown OCR engine: {preferred!r}")
    return preferred


def _ocr_page(page: "fitz.Page", engine: str, dpi: int = OCR_DPI) -> str:
    if engine == "rapidocr":
        return _ocr_page_rapidocr(page, dpi)
    if engine == "tesseract":
        return _ocr_page_tesseract(page, dpi)
    raise ValueError(f"Unknown OCR engine: {engine!r}")


def _open_pdf(pdf_path: Path) -> "fitz.Document":
    """Open a PDF, surfacing a helpful error for encrypted/corrupt files."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Could not open {pdf_path}: {e}") from e
    if doc.is_encrypted:

        if not doc.authenticate(""):
            doc.close()
            raise RuntimeError(
                f"{pdf_path} is password-protected. Decrypt it first "
                f"(e.g. `qpdf --decrypt in.pdf out.pdf`) and re-run."
            )
    return doc


def extract_slides(
    pdf_path: Path,
    use_ocr: bool = True,
    ocr_engine: str = "auto",
    verbose: bool = True,
) -> list[Slide]:
    """Extract a structured list of slides from a pitch-deck PDF.

    Per page we try, in order:
      1. Native PyMuPDF text in proper reading order (`sort=True`).
      2. If that text is missing, too short, or garbled (bad font encoding),
         rasterize the page and run OCR.
      3. OCR is also discarded if it looks like noise (very low alpha ratio).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    engine = _resolve_ocr_engine(ocr_engine) if use_ocr else None

    slides: list[Slide] = []
    with _open_pdf(pdf_path) as doc:
        n_pages = doc.page_count
        if verbose:
            print(f"  [01] {n_pages} pages, OCR engine: {engine or 'disabled'}", flush=True)
        for i, page in enumerate(doc, start=1):
            native = _clean_text(page.get_text("text", sort=True))
            garbled = _looks_garbled(native)
            too_short = len(native) < MIN_NATIVE_CHARS

            text = "" if garbled else native
            source = "text"
            reason = ""

            needs_ocr = too_short or garbled
            if needs_ocr and use_ocr and engine is not None:
                reason = "garbled" if garbled else f"only {len(native)} native chars"
                if verbose:
                    print(f"  [01] page {i}/{n_pages}: OCR ({engine}, {reason}) ...", flush=True)
                try:
                    ocr_clean = _clean_text(_ocr_page(page, engine))
                    if ocr_clean and _alpha_ratio(ocr_clean) >= MIN_OCR_ALPHA_RATIO:
                        if len(ocr_clean) >= len(text):
                            text = ocr_clean
                            source = f"ocr:{engine}"
                    elif verbose and ocr_clean:
                        print(
                            f"  [01] page {i}/{n_pages}: OCR result looked like noise, dropped",
                            flush=True,
                        )
                except RuntimeError as e:
                    print(f"  [01] page {i}/{n_pages}: OCR skipped ({e})", file=sys.stderr)
            elif verbose:
                print(f"  [01] page {i}/{n_pages}: native text ({len(native)} chars)", flush=True)

            if not text:
                source = "empty"

            title, body = _split_title_body(text)
            slides.append(
                Slide(
                    page=i,
                    title=title,
                    body=body,
                    source=source,
                    char_count=len(text),
                )
            )
    return slides


def save_slides(slides: list[Slide], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(s) for s in slides]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _summarize(slides: list[Slide]) -> str:
    by_source: dict[str, int] = {}
    for s in slides:
        by_source[s.source] = by_source.get(s.source, 0) + 1
    parts = [f"{k}={v}" for k, v in sorted(by_source.items())]
    total_chars = sum(s.char_count for s in slides)
    return f"{len(slides)} slides ({', '.join(parts)}), {total_chars} chars total"


def _resolve_pdf_inputs(pdfs: list[Path]) -> list[Path]:
    """Expand the CLI's pdf args into a concrete list of PDF files.

    - No args -> every *.pdf inside DEFAULT_PDF_DIR.
    - A directory -> every *.pdf directly inside it.
    - A file -> used as-is.
    """
    if not pdfs:
        if not DEFAULT_PDF_DIR.exists():
            raise FileNotFoundError(
                f"No PDF given and default directory {DEFAULT_PDF_DIR} does not exist."
            )
        found = sorted(DEFAULT_PDF_DIR.glob("*.pdf"))
        if not found:
            raise FileNotFoundError(
                f"No PDF given and no *.pdf files found in {DEFAULT_PDF_DIR}."
            )
        return found

    resolved: list[Path] = []
    for p in pdfs:
        if p.is_dir():
            resolved.extend(sorted(p.glob("*.pdf")))
        else:
            resolved.append(p)
    if not resolved:
        raise FileNotFoundError("No PDF files matched the given inputs.")
    return resolved


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Module 01 — extract slide text from a pitch-deck PDF.",
    )
    parser.add_argument(
        "pdf",
        type=Path,
        nargs="*",
        help=(
            "Path(s) to pitch-deck PDF(s) or a directory containing them. "
            f"Defaults to every *.pdf in {DEFAULT_PDF_DIR}/."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_MODULE_01_SLIDES_CSV,
        help="Combined slides CSV (default: output/module_01/slides.csv).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable OCR fallback for image-heavy slides.",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=OCR_ENGINES,
        default="auto",
        help="OCR engine to use (default: auto -> rapidocr if available, else tesseract).",
    )
    args = parser.parse_args(argv)

    pdf_paths = _resolve_pdf_inputs(args.pdf)
    all_rows: list[dict] = []

    for pdf_path in pdf_paths:
        deck_id = canonical_deck_id(pdf_path.stem)
        print(f"[01] Reading {pdf_path} ...")
        slides = extract_slides(
            pdf_path,
            use_ocr=not args.no_ocr,
            ocr_engine=args.ocr_engine,
        )
        for s in slides:
            all_rows.append(slide_row(deck_id, asdict(s)))
        print(f"[01] {deck_id}: {_summarize(slides)}")

    write_slides_csv(args.out, all_rows)
    decks = len({r["deck_id"] for r in all_rows})
    print(f"[01] Wrote {len(all_rows)} slide rows ({decks} decks) -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
