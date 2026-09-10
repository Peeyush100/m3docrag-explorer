"""Render single pages from the M3DocVQA source PDFs as images.

Retrieval page indices produced by the M3DocRAG pipeline are 0-based
(see m3docrag's examples/run_indexing_m3docvqa.py: `page_id = range(len(doc_emb))`,
built directly from `pdf2image.convert_from_path` page order). pdf2image's
`first_page`/`last_page` args are 1-based, so we add 1 when rendering.
"""

import os
import shutil

import streamlit as st
from pdf2image import convert_from_path, pdfinfo_from_path

# On Streamlit Community Cloud, poppler-utils is installed via packages.txt
# and lands on PATH already. For local development outside that env (e.g. a
# conda env where poppler was installed into the env's own bin/), point
# LOCAL_POPPLER_BIN at it via an env var instead of hardcoding a path.
_LOCAL_POPPLER_BIN = os.environ.get("LOCAL_POPPLER_BIN", "")
if _LOCAL_POPPLER_BIN and _LOCAL_POPPLER_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _LOCAL_POPPLER_BIN + os.pathsep + os.environ.get("PATH", "")

POPPLER_PATH = _LOCAL_POPPLER_BIN if shutil.which("pdftoppm", path=_LOCAL_POPPLER_BIN or None) else None


@st.cache_data(show_spinner=False)
def get_page_count(pdf_path_str: str) -> int:
    try:
        info = pdfinfo_from_path(pdf_path_str, poppler_path=POPPLER_PATH)
        return info.get("Pages", 0)
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def render_page(pdf_path_str: str, page_index_0based: int, dpi: int = 120):
    """Returns a PIL.Image for the given 0-based page index, or None."""
    page_1based = page_index_0based + 1
    try:
        images = convert_from_path(
            pdf_path_str,
            dpi=dpi,
            first_page=page_1based,
            last_page=page_1based,
            poppler_path=POPPLER_PATH,
        )
    except Exception:
        return None
    return images[0] if images else None
