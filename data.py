"""Data loading / scoring helpers for the M3DocRAG results explorer.

Joins gold M3DocVQA questions with a model prediction file (retrieval + QA
answer) and produces one row per question with everything the Streamlit app
needs: gold answer, predicted answer, correctness, modality, hop type,
question type, gold supporting docs and the pages actually retrieved.

Source PDFs (~1.6GB for the full dev set) are NOT stored in this git repo.
They're fetched on demand, one file at a time, from a public Hugging Face
Datasets repo (see scripts/upload_pdfs_to_hf.py) and cached under
`./pdf_cache/`. For local development on a machine/cluster that already has
the M3DocVQA PDFs on disk, set LOCAL_PDF_DIR to skip the download entirely.
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from scoring import list_em, list_f1, MULTI_HOP_QUESTION_TYPES

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

GOLD_FILES = {
    "Full dev (2441 Qs)": DATA_DIR / "multimodalqa" / "MMQA_dev.jsonl",
    "Dev subset (300 docs)": DATA_DIR / "multimodalqa" / "MMQA_devsubset.jsonl",
}

OUTPUT_DIR = DATA_DIR / "output"

# All splits' PDFs are drawn from the same underlying pool of ~3,366 M3DocVQA
# dev documents, so a single doc_id -> pdf lookup covers every gold_key.
HF_PDF_DATASET_REPO = os.environ.get("HF_PDF_DATASET_REPO", "")  # e.g. "your-username/m3docvqa-pdfs"
LOCAL_PDF_DIR = os.environ.get("LOCAL_PDF_DIR", "")  # optional: skip HF entirely if PDFs are already on disk
PDF_CACHE_DIR = REPO_ROOT / "pdf_cache"


def _prettify_run_label(folder_name: str, file_name: str) -> str:
    is_subset = "subset" in folder_name
    scope = "dev subset" if is_subset else "full dev"
    base = folder_name.replace("_subset", "")
    names = {
        "colpali_qwen2vl": "ColPali retrieval + Qwen2-VL answer",
        "colpali_qwen3vl": "ColPali retrieval + Qwen3-VL answer",
        "colpali_qwen38": "ColPali retrieval + Qwen3.8-27B answer (thinking on, 128 tok — mostly truncated, diagnostic only)",
        "colpali_qwen38_nothink": "ColPali retrieval + Qwen3.8-27B answer (no-think, matched to Qwen2-VL)",
        "oracle_qwen2vl": "Oracle (gold pages) + Qwen2-VL answer",
        "oracle_qwen3vl": "Oracle (gold pages) + Qwen3-VL answer",
    }
    pretty = names.get(base, base)
    ts = file_name.replace(".json", "")
    ts_suffix = ts.split("_")[-2] + "_" + ts.split("_")[-1] if "_" in ts else ts
    return f"{pretty} · {scope} · {ts_suffix}"


def discover_prediction_runs():
    """Find prediction json files that actually contain QA answers (i.e. not
    the *_eval_results.json summaries and not retrieval-only runs, which
    store an empty pred_answer for every question)."""
    runs = {}
    if not OUTPUT_DIR.exists():
        return runs
    for sub in sorted(OUTPUT_DIR.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.glob("*.json")):
            if "eval_results" in f.name:
                continue
            try:
                with open(f, encoding="utf-8") as fh:
                    sample = json.load(fh)
            except Exception:
                continue
            if not isinstance(sample, dict) or not sample:
                continue
            probe_vals = list(sample.values())[:5]
            if not all(isinstance(v, dict) and "pred_answer" in v for v in probe_vals):
                continue
            if not any(v["pred_answer"].strip() for v in probe_vals):
                continue  # retrieval-only run: pred_answer is always empty
            label = _prettify_run_label(sub.name, f.name)
            is_subset = "subset" in sub.name
            gold_key = "Dev subset (300 docs)" if is_subset else "Full dev (2441 Qs)"
            runs[label] = {"path": f, "gold_key": gold_key}
    return runs


@st.cache_data(show_spinner=False)
def load_gold(gold_key: str):
    path = GOLD_FILES[gold_key]
    examples = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples[ex["qid"]] = ex
    return examples


@st.cache_data(show_spinner=False)
def load_predictions(pred_path: str):
    with open(pred_path, encoding="utf-8") as f:
        return json.load(f)


def _gold_answer_list(example):
    return [str(item["answer"]) for item in example["answers"]]


def _answer_modality(example):
    modalities = {item["modality"] for item in example["answers"]}
    return sorted(modalities)[0] if modalities else "unknown"


def _hop_type(example):
    qtype = example["metadata"]["type"]
    return "Multi-hop" if qtype in MULTI_HOP_QUESTION_TYPES else "Single-hop"


def run_has_judge_scores(pred_path: str) -> bool:
    """True if this run's prediction file was annotated by
    merge_llm_judge_qwen38.py (has an llm_judge_score on at least one entry)."""
    preds = load_predictions(pred_path)
    return any("llm_judge_score" in v for v in list(preds.values())[:5])


@st.cache_data(show_spinner="Scoring predictions against gold answers...")
def build_results_table(gold_key: str, pred_path: str, metric: str = "em", threshold: float = 1.0):
    gold = load_gold(gold_key)
    preds = load_predictions(pred_path)

    rows = []
    for qid, example in gold.items():
        gold_answers = _gold_answer_list(example)
        pred_entry = preds.get(qid)
        pred_answer = pred_entry["pred_answer"].strip() if pred_entry else ""
        retrieval_results = pred_entry["page_retrieval_results"] if pred_entry else []

        em = list_em(pred_answer, gold_answers) if pred_entry else 0.0
        f1 = list_f1(pred_answer, gold_answers) if pred_entry else 0.0
        llm_judge_score = pred_entry.get("llm_judge_score") if pred_entry else None
        llm_judge_raw = pred_entry.get("llm_judge_raw", "") if pred_entry else ""

        gold_doc_ids = sorted({ctx["doc_id"] for ctx in example["supporting_context"]})
        gold_doc_parts = {ctx["doc_id"]: ctx["doc_part"] for ctx in example["supporting_context"]}
        retrieved_doc_ids = [r[0] for r in retrieval_results]
        retrieved_hit = any(d in gold_doc_ids for d in retrieved_doc_ids)

        if metric == "judge":
            correct = llm_judge_score == 1
        else:
            correct = (em if metric == "em" else f1) >= threshold

        rows.append(
            {
                "qid": qid,
                "question": example["question"],
                "gold_answers": gold_answers,
                "pred_answer": pred_answer,
                "has_prediction": pred_entry is not None,
                "em": em,
                "f1": f1,
                "llm_judge_score": llm_judge_score,
                "llm_judge_raw": llm_judge_raw,
                "correct": correct,
                "modality": _answer_modality(example),
                "hop_type": _hop_type(example),
                "q_type": example["metadata"]["type"],
                "gold_doc_ids": gold_doc_ids,
                "gold_doc_parts": gold_doc_parts,
                "retrieval_results": retrieval_results,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieval_hit": retrieved_hit,
            }
        )

    return pd.DataFrame(rows)


def pdf_path_for_doc(gold_key: str, doc_id: str):
    """Resolve a doc_id to a local PDF path, trying (in order): an explicit
    local corpus dir, a previously-downloaded cache entry, then a lazy
    single-file download from the Hugging Face dataset repo."""
    if LOCAL_PDF_DIR:
        p = Path(LOCAL_PDF_DIR) / f"{doc_id}.pdf"
        if p.exists():
            return p

    cached = PDF_CACHE_DIR / f"{doc_id}.pdf"
    if cached.exists():
        return cached

    if not HF_PDF_DATASET_REPO:
        return None

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=HF_PDF_DATASET_REPO,
            repo_type="dataset",
            filename=f"pdfs/{doc_id}.pdf",
            local_dir=str(PDF_CACHE_DIR),
        )
        return Path(downloaded)
    except Exception:
        return None
