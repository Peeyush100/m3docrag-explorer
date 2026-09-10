"""M3DocRAG Results Explorer

A Streamlit app to browse M3DocVQA questions, split by whether the model
answered correctly or incorrectly, filter by hop type / answer modality,
and inspect - per question - the gold answer vs. the model's answer plus
the pages the model was *supposed* to retrieve (gold supporting docs) next
to the pages ColPali *actually* retrieved.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import streamlit as st

from data import (
    build_results_table,
    discover_prediction_runs,
    pdf_path_for_doc,
)
from pdf_render import get_page_count, render_page

st.set_page_config(
    page_title="M3DocRAG Results Explorer",
    page_icon="\U0001f4c4",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODALITY_COLORS = {"text": "#2563eb", "table": "#16a34a", "image": "#ea580c"}
MODALITY_ICONS = {"text": "\U0001f4dd", "table": "\U0001f4ca", "image": "\U0001f5bc️"}

CUSTOM_CSS = """
<style>
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    color: white;
    margin-right: 6px;
}
.qcard {
    padding: 0.9rem 1.1rem;
    border-radius: 10px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 0.6rem;
}
.answer-box {
    padding: 0.7rem 1rem;
    border-radius: 8px;
    font-size: 0.95rem;
    min-height: 3rem;
}
.gold-box { background: rgba(22,163,74,0.12); border: 1px solid rgba(22,163,74,0.4); }
.pred-correct { background: rgba(22,163,74,0.12); border: 1px solid rgba(22,163,74,0.4); }
.pred-wrong { background: rgba(220,38,38,0.10); border: 1px solid rgba(220,38,38,0.4); }
.small-mono { font-family: monospace; font-size: 0.8rem; opacity: 0.75; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def badge(text, color):
    return f'<span class="badge" style="background:{color}">{text}</span>'


def modality_badge(modality):
    color = MODALITY_COLORS.get(modality, "#6b7280")
    icon = MODALITY_ICONS.get(modality, "❓")
    return badge(f"{icon} {modality}", color)


def short_id(doc_id):
    return doc_id[:10] + "…"


# --------------------------------------------------------------------------
# Sidebar: choose run + dataset + correctness definition
# --------------------------------------------------------------------------
st.sidebar.title("\U0001f4c4 M3DocRAG Explorer")
st.sidebar.caption("Browse M3DocVQA predictions from the ColPali + MLM RAG pipeline.")

runs = discover_prediction_runs()
if not runs:
    st.error(
        "No prediction runs found under `job/output/`. Run the RAG pipeline "
        "(examples/run_rag_m3docvqa.py) first."
    )
    st.stop()

run_labels = list(runs.keys())
default_idx = next(
    (i for i, l in enumerate(run_labels) if "full dev" in l and "Qwen2-VL" in l),
    0,
)
selected_run_label = st.sidebar.selectbox("Prediction run", run_labels, index=default_idx)
run_info = runs[selected_run_label]
pred_path = str(run_info["path"])
default_gold_key = run_info["gold_key"]

gold_key = st.sidebar.selectbox(
    "Gold / PDF corpus",
    ["Full dev (2441 Qs)", "Dev subset (300 docs)"],
    index=0 if default_gold_key.startswith("Full") else 1,
    help="Automatically matched to the selected run; override only if you know the run used a different split.",
)

st.sidebar.markdown("---")
metric_choice = st.sidebar.radio(
    "Correctness definition",
    ["Exact match (EM = 1.0)", "Partial credit (F1 ≥ 0.5)"],
    index=0,
)
metric = "em" if metric_choice.startswith("Exact") else "f1"
threshold = 1.0 if metric == "em" else 0.5

with st.spinner("Loading & scoring..."):
    df = build_results_table(gold_key, pred_path, metric=metric, threshold=threshold)

st.sidebar.markdown("---")
search_query = st.sidebar.text_input("\U0001f50d Search question / qid", "")
st.sidebar.caption(f"{len(df)} questions loaded from `{Path(pred_path).parent.name}/{Path(pred_path).name}`")

if search_query.strip():
    q = search_query.strip().lower()
    df = df[df["question"].str.lower().str.contains(q) | df["qid"].str.lower().str.contains(q)]

n_total = len(df)
n_correct = int(df["correct"].sum())
n_incorrect = n_total - n_correct
acc = (n_correct / n_total * 100) if n_total else 0.0

# --------------------------------------------------------------------------
# Top metrics
# --------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Questions", f"{n_total:,}")
m2.metric("✅ Correct", f"{n_correct:,}")
m3.metric("❌ Incorrect", f"{n_incorrect:,}")
m4.metric("Accuracy", f"{acc:.1f}%")

st.markdown("")


def render_filters(sub_df, key_prefix):
    """Hop-type + modality filter widgets. Returns the filtered dataframe."""
    c1, c2, c3 = st.columns([1.2, 1.6, 1.6])
    with c1:
        hop_choice = st.segmented_control(
            "Hop type",
            options=["All", "Single-hop", "Multi-hop"],
            default="All",
            key=f"{key_prefix}_hop",
        )
    with c2:
        modality_choice = st.segmented_control(
            "Answer modality",
            options=["All", "text", "table", "image"],
            default="All",
            key=f"{key_prefix}_modality",
        )
    with c3:
        qtypes = sorted(sub_df["q_type"].unique().tolist())
        qtype_choice = st.multiselect(
            "Question type (optional)", qtypes, default=[], key=f"{key_prefix}_qtype"
        )

    out = sub_df
    if hop_choice and hop_choice != "All":
        out = out[out["hop_type"] == hop_choice]
    if modality_choice and modality_choice != "All":
        out = out[out["modality"] == modality_choice]
    if qtype_choice:
        out = out[out["q_type"].isin(qtype_choice)]
    return out


def render_question_list(sub_df, key_prefix):
    if sub_df.empty:
        st.info("No questions match the current filters.")
        return None

    display_df = sub_df.copy().reset_index(drop=True)
    display_df["Gold"] = display_df["gold_answers"].apply(lambda a: ", ".join(a)[:60])
    display_df["Predicted"] = display_df["pred_answer"].apply(lambda a: (a[:60] if a else "(no prediction)"))
    display_df["Hit"] = display_df["retrieval_hit"].apply(lambda h: "✅" if h else "❌")
    display_df["EM"] = display_df["em"].round(2)
    display_df["F1"] = display_df["f1"].round(2)

    show_cols = [
        "question",
        "Gold",
        "Predicted",
        "modality",
        "hop_type",
        "EM",
        "F1",
        "Hit",
    ]
    rename = {
        "question": "Question",
        "modality": "Modality",
        "hop_type": "Hop",
        "Hit": "Retrieved gold doc?",
    }

    st.caption(f"{len(display_df)} question(s) — click a row to inspect it below.")
    event = st.dataframe(
        display_df[show_cols].rename(columns=rename),
        width="stretch",
        hide_index=True,
        height=360,
        on_select="rerun",
        selection_mode="single-row",
        key=f"{key_prefix}_table",
    )

    rows = event.selection.rows if event and event.selection else []
    if not rows:
        return None
    return display_df.loc[rows[0], "qid"]


def render_pages_for_doc(doc_id, gold_key, key_prefix, highlight_page=None, page_scores=None):
    pdf_path = pdf_path_for_doc(gold_key, doc_id)
    if pdf_path is None:
        st.warning(f"PDF not found locally for doc `{short_id(doc_id)}`.")
        return
    n_pages = get_page_count(str(pdf_path))
    if n_pages == 0:
        st.warning("Could not read this PDF.")
        return

    default_page = highlight_page if highlight_page is not None else 0
    default_page = max(0, min(default_page, n_pages - 1))

    page_0based = st.slider(
        "Page",
        min_value=0,
        max_value=n_pages - 1,
        value=default_page,
        key=f"{key_prefix}_slider",
        format="Page %d",
        help=f"0-indexed to match ColPali's page ids · document has {n_pages} pages",
    )
    img = render_page(str(pdf_path), page_0based)
    caption = f"doc `{short_id(doc_id)}` · page {page_0based}"
    if page_scores and page_0based in page_scores:
        caption += f" · score {page_scores[page_0based]:.2f}"
    if img is not None:
        st.image(img, caption=caption, width="stretch")
    else:
        st.warning("Could not render this page.")


def render_detail(row, gold_key):
    st.markdown("---")
    correct = row["correct"]
    status_badge = badge("✅ CORRECT", "#16a34a") if correct else badge("❌ INCORRECT", "#dc2626")
    st.markdown(
        f"### {row['question']}",
    )
    st.markdown(
        f"{status_badge} {modality_badge(row['modality'])} "
        f"{badge(row['hop_type'], '#7c3aed')} {badge(row['q_type'], '#6b7280')} "
        f"&nbsp;&nbsp;<span class='small-mono'>qid: {row['qid']} &middot; EM {row['em']:.2f} &middot; F1 {row['f1']:.2f}</span>",
        unsafe_allow_html=True,
    )

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Gold (ground-truth) answer**")
        st.markdown(
            f"<div class='answer-box gold-box'>{' / '.join(row['gold_answers'])}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("**Model's predicted answer**")
        pred_class = "pred-correct" if correct else "pred-wrong"
        pred_text = row["pred_answer"] if row["pred_answer"] else "(no prediction)"
        st.markdown(
            f"<div class='answer-box {pred_class}'>{pred_text}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown("#### \U0001f4c4 Required vs. retrieved pages")
    st.caption(
        "M3DocVQA gold annotations are **document-level** (which PDF holds the answer), "
        "not page-level — use the slider to browse the required document. ColPali's retrieved "
        "pages are exact; a match against a gold document is marked ✅ below."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("##### \U0001f3af Required (gold) document(s)")
        if not row["gold_doc_ids"]:
            st.info("No supporting documents listed for this question.")
        for i, doc_id in enumerate(row["gold_doc_ids"]):
            part = row["gold_doc_parts"].get(doc_id, "?")
            was_retrieved = doc_id in row["retrieved_doc_ids"]
            hit_badge = badge("retrieved by ColPali", "#16a34a") if was_retrieved else badge("missed by ColPali", "#dc2626")
            st.markdown(f"{modality_badge(part)} {hit_badge}", unsafe_allow_html=True)
            with st.expander(f"Document {i+1}: `{short_id(doc_id)}`", expanded=(i == 0)):
                render_pages_for_doc(doc_id, gold_key, key_prefix=f"gold_{row['qid']}_{i}")

    with right:
        st.markdown("##### \U0001f50e Actually retrieved by ColPali RAG")
        if not row["retrieval_results"]:
            st.info("No retrieval results recorded for this question.")
        for rank, (doc_id, page, score) in enumerate(row["retrieval_results"], start=1):
            is_gold = doc_id in row["gold_doc_ids"]
            hit_badge = badge("✅ gold doc", "#16a34a") if is_gold else badge("not a gold doc", "#dc2626")
            st.markdown(
                f"{badge(f'rank #{rank}', '#2563eb')} {hit_badge} "
                f"<span class='small-mono'>score {score:.3f}</span>",
                unsafe_allow_html=True,
            )
            with st.expander(f"Retrieved: `{short_id(doc_id)}` (page {page})", expanded=(rank == 1)):
                render_pages_for_doc(
                    doc_id,
                    gold_key,
                    key_prefix=f"ret_{row['qid']}_{rank}",
                    highlight_page=page,
                    page_scores={page: score},
                )


def get_row_by_qid(source_df, qid):
    match = source_df[source_df["qid"] == qid]
    if match.empty:
        return None
    return match.iloc[0]


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_correct, tab_incorrect = st.tabs(
    [
        "\U0001f4ca Overview",
        f"✅ Correct ({n_correct})",
        f"❌ Incorrect ({n_incorrect})",
    ]
)

with tab_overview:
    st.markdown("#### Accuracy breakdown")
    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown("**By hop type**")
        hop_acc = df.groupby("hop_type")["correct"].mean().mul(100).round(1)
        st.bar_chart(hop_acc)
    with oc2:
        st.markdown("**By answer modality**")
        mod_acc = df.groupby("modality")["correct"].mean().mul(100).round(1)
        st.bar_chart(mod_acc)

    st.markdown("**By question type**")
    qtype_acc = df.groupby("q_type")["correct"].mean().mul(100).round(1).sort_values()
    st.bar_chart(qtype_acc)

    st.markdown("**Retrieval hit rate (gold doc appears in retrieved pages)**")
    hit_rate = df["retrieval_hit"].mean() * 100
    st.progress(min(hit_rate / 100, 1.0), text=f"{hit_rate:.1f}% of questions had a gold document among the retrieved pages")

with tab_correct:
    correct_df = df[df["correct"]]
    filtered = render_filters(correct_df, "correct")
    selected_qid = render_question_list(filtered, "correct")
    if selected_qid:
        row = get_row_by_qid(filtered, selected_qid)
        if row is not None:
            render_detail(row, gold_key)

with tab_incorrect:
    incorrect_df = df[~df["correct"]]
    filtered = render_filters(incorrect_df, "incorrect")
    selected_qid = render_question_list(filtered, "incorrect")
    if selected_qid:
        row = get_row_by_qid(filtered, selected_qid)
        if row is not None:
            render_detail(row, gold_key)
