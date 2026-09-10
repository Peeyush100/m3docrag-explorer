# M3DocRAG Results Explorer

A Streamlit app for browsing [M3DocVQA](https://m3docrag.github.io/) eval
results from an [M3DocRAG](https://github.com/bloomberg/m3docrag) (ColPali +
MLM) run: which questions were answered correctly vs. incorrectly, filtered
by hop type (single/multi) and answer modality (text/table/image), with a
per-question view of the gold answer vs. the model's answer and the pages
the model *should* have retrieved next to what ColPali *actually* retrieved.

Prediction files and gold Q&A live in `data/` (small, checked into git).
The ~1.6GB of source PDFs do **not** live in this repo — they're fetched
lazily, one page at a time, from a Hugging Face Datasets repo and cached
locally in `pdf_cache/` (gitignored).

## Run locally

```bash
pip install -r requirements.txt
# poppler is needed by pdf2image; on Debian/Ubuntu: apt-get install poppler-utils
# on a conda env where poppler isn't on PATH, point at it explicitly:
export LOCAL_POPPLER_BIN=/path/to/conda/env/bin

# Either point at PDFs already on disk...
export LOCAL_PDF_DIR=/path/to/m3docvqa/pdfs_dev
# ...or fetch them from the published Hugging Face dataset:
export HF_PDF_DATASET_REPO=your-username/m3docvqa-pdfs

streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Publish the PDFs once: `python scripts/upload_pdfs_to_hf.py --repo-id <you>/m3docvqa-pdfs --source-dir <path-to-pdfs_dev>` (see that script's docstring).
3. On [share.streamlit.io](https://share.streamlit.io), "New app" → pick this repo → main file `app.py`.
4. Under the app's *Advanced settings → Secrets*, add:
   ```toml
   HF_PDF_DATASET_REPO = "your-username/m3docvqa-pdfs"
   ```
5. Deploy. `packages.txt` (poppler-utils) and `requirements.txt` are picked up automatically.

## Files

- `app.py` — Streamlit UI (tabs, filters, question table, detail view).
- `data.py` — loads gold Q&A + predictions, scores them, resolves PDF paths (local or HF).
- `scoring.py` — vendored M3DocVQA list-EM / list-F1 scoring (no torch dependency).
- `pdf_render.py` — renders a single PDF page to an image via pdf2image.
- `scripts/upload_pdfs_to_hf.py` — one-time script to publish the PDF corpus to Hugging Face.
