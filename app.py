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

from data import build_results_table, discover_prediction_runs, pdf_path_for_doc
from pdf_render import get_page_count, render_page
from theme import (
    GREEN,
    MUTED,
    RED,
    badge,
    inject,
    masthead,
    modality_badge,
    stat_card,
)

st.set_page_config(
    page_title="M3DocRAG Explorer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject()


def short_id(doc_id):
    return doc_id[:10] + "…"


def ctl_label(text):
    st.markdown(f'<div class="nf-ctl-label">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------
runs = discover_prediction_runs()
if not runs:
    st.error(
        "No prediction runs found under `data/output/`. Add a run produced by "
        "the M3DocRAG pipeline (examples/run_rag_m3docvqa.py)."
    )
    st.stop()

run_labels = list(runs.keys())
default_idx = next((i for i, l in enumerate(run_labels) if "full dev" in l and "Qwen2-VL" in l), 0)

masthead("Multi-modal RAG results explorer &nbsp;·&nbsp; M3DocVQA")

# --------------------------------------------------------------------------
# Top control bar (replaces the sidebar)
# --------------------------------------------------------------------------
st.markdown('<div class="nf-controlbar">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([3.1, 1.7, 1.9, 2.1])

with c1:
    ctl_label("Run")
    selected_run_label = st.selectbox(
        "Run", run_labels, index=default_idx, label_visibility="collapsed"
    )
run_info = runs[selected_run_label]
pred_path = str(run_info["path"])

with c2:
    ctl_label("Corpus")
    gold_key = st.selectbox(
        "Corpus",
        ["Full dev (2441 Qs)", "Dev subset (300 docs)"],
        index=0 if run_info["gold_key"].startswith("Full") else 1,
        label_visibility="collapsed",
        help="Matched to the selected run automatically.",
    )

with c3:
    ctl_label("Scored as")
    metric_choice = st.selectbox(
        "Scored as",
        ["Exact match", "Partial credit (F1 ≥ 0.5)"],
        label_visibility="collapsed",
    )

with c4:
    ctl_label("Search")
    search_query = st.text_input(
        "Search", "", placeholder="Search questions or qid…", label_visibility="collapsed"
    )
st.markdown("</div>", unsafe_allow_html=True)

metric = "em" if metric_choice.startswith("Exact") else "f1"
threshold = 1.0 if metric == "em" else 0.5

df = build_results_table(gold_key, pred_path, metric=metric, threshold=threshold)

if search_query.strip():
    q = search_query.strip().lower()
    df = df[df["question"].str.lower().str.contains(q) | df["qid"].str.lower().str.contains(q)]

n_total = len(df)
n_correct = int(df["correct"].sum())
n_incorrect = n_total - n_correct
acc = (n_correct / n_total * 100) if n_total else 0.0
hit_rate = df["retrieval_hit"].mean() * 100 if n_total else 0.0

# --------------------------------------------------------------------------
# Stat cards
# --------------------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    stat_card("Questions", f"{n_total:,}", "#4A9EFF")
with m2:
    stat_card("Correct", f"{n_correct:,}", GREEN)
with m3:
    stat_card("Incorrect", f"{n_incorrect:,}", RED)
with m4:
    stat_card("Accuracy", f"{acc:.1f}%", "#F5B50A")
with m5:
    stat_card("Retrieval hit", f"{hit_rate:.1f}%", "#A855F7")

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------
def render_filters(sub_df, key_prefix):
    """Hop-type + modality filter widgets. Returns the filtered dataframe."""
    c1, c2, c3 = st.columns([1.2, 1.6, 1.9])
    with c1:
        ctl_label("Hops")
        hop_choice = st.segmented_control(
            "Hop type",
            options=["All", "Single-hop", "Multi-hop"],
            default="All",
            key=f"{key_prefix}_hop",
            label_visibility="collapsed",
        )
    with c2:
        ctl_label("Answer modality")
        modality_choice = st.segmented_control(
            "Answer modality",
            options=["All", "text", "table", "image"],
            default="All",
            key=f"{key_prefix}_modality",
            label_visibility="collapsed",
        )
    with c3:
        ctl_label("Question type (optional)")
        qtypes = sorted(sub_df["q_type"].unique().tolist())
        qtype_choice = st.multiselect(
            "Question type",
            qtypes,
            default=[],
            key=f"{key_prefix}_qtype",
            label_visibility="collapsed",
            placeholder="Any reasoning pattern",
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
    display_df["Predicted"] = display_df["pred_answer"].apply(
        lambda a: (a[:60] if a else "(no prediction)")
    )
    display_df["Hit"] = display_df["retrieval_hit"].apply(lambda h: "✅" if h else "❌")
    display_df["EM"] = display_df["em"].round(2)
    display_df["F1"] = display_df["f1"].round(2)

    show_cols = ["question", "Gold", "Predicted", "modality", "hop_type", "EM", "F1", "Hit"]
    rename = {
        "question": "Question",
        "modality": "Modality",
        "hop_type": "Hop",
        "Hit": "Got gold doc?",
    }

    st.markdown(
        f"<div class='nf-note'>{len(display_df):,} question(s) — "
        f"<b style='color:#fff'>click any row</b> to open it below.</div>",
        unsafe_allow_html=True,
    )
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
        st.warning(f"PDF unavailable for `{short_id(doc_id)}`.")
        return
    n_pages = get_page_count(str(pdf_path))
    if n_pages == 0:
        st.warning("Could not read this PDF.")
        return

    default_page = highlight_page if highlight_page is not None else 0
    default_page = max(0, min(default_page, n_pages - 1))

    if n_pages > 1:
        page_0based = st.slider(
            "Page",
            min_value=0,
            max_value=n_pages - 1,
            value=default_page,
            key=f"{key_prefix}_slider",
            format="Page %d",
            help=f"0-indexed to match ColPali's page ids · {n_pages} pages total",
        )
    else:
        page_0based = 0

    img = render_page(str(pdf_path), page_0based)
    caption = f"{short_id(doc_id)} · page {page_0based}"
    if page_scores and page_0based in page_scores:
        caption += f" · score {page_scores[page_0based]:.2f}"
    if img is not None:
        st.image(img, caption=caption, width="stretch")
    else:
        st.warning("Could not render this page.")


def render_detail(row, gold_key):
    correct = row["correct"]
    status = badge("✓ Correct", GREEN) if correct else badge("✗ Incorrect", RED)

    st.markdown('<div class="nf-detail">', unsafe_allow_html=True)
    st.markdown(f'<div class="nf-question">{row["question"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f"{status} {modality_badge(row['modality'])} "
        f"{badge(row['hop_type'], '#7c3aed')} {badge(row['q_type'], '#444')} "
        f"<div class='small-mono' style='margin-top:6px'>qid {row['qid']} &nbsp;·&nbsp; "
        f"EM {row['em']:.2f} &nbsp;·&nbsp; F1 {row['f1']:.2f}</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="nf-answer-label">Ground truth</div>', unsafe_allow_html=True)
        st.markdown(
            f"<div class='answer-box gold-box'>{' / '.join(row['gold_answers'])}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown('<div class="nf-answer-label">Model answered</div>', unsafe_allow_html=True)
        pred_class = "pred-correct" if correct else "pred-wrong"
        pred_text = row["pred_answer"] or "(no prediction)"
        st.markdown(
            f"<div class='answer-box {pred_class}'>{pred_text}</div>", unsafe_allow_html=True
        )

    st.markdown('<div class="nf-section">Required vs. retrieved pages</div>', unsafe_allow_html=True)
    st.markdown(
        "<div class='nf-note'>M3DocVQA marks evidence at <b>document</b> level, not page level — "
        "use the slider to browse the required document. ColPali's retrieved pages are exact.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="nf-section">🎯 Should have retrieved</div>', unsafe_allow_html=True)
        if not row["gold_doc_ids"]:
            st.info("No supporting documents listed.")
        for i, doc_id in enumerate(row["gold_doc_ids"]):
            part = row["gold_doc_parts"].get(doc_id, "?")
            was_retrieved = doc_id in row["retrieved_doc_ids"]
            cls = "hit" if was_retrieved else "miss"
            tag = (
                badge("found it", GREEN) if was_retrieved else badge("missed", RED)
            )
            st.markdown(
                f"<div class='nf-pagecard {cls}'>{modality_badge(part)} {tag}"
                f"<div class='small-mono'>{short_id(doc_id)}</div></div>",
                unsafe_allow_html=True,
            )
            with st.expander(f"View document {i + 1}", expanded=(i == 0)):
                render_pages_for_doc(doc_id, gold_key, key_prefix=f"gold_{row['qid']}_{i}")

    with right:
        st.markdown('<div class="nf-section">🔎 ColPali retrieved</div>', unsafe_allow_html=True)
        if not row["retrieval_results"]:
            st.info("No retrieval results recorded.")
        for rank, (doc_id, page, score) in enumerate(row["retrieval_results"], start=1):
            is_gold = doc_id in row["gold_doc_ids"]
            cls = "hit" if is_gold else "miss"
            tag = badge("gold doc", GREEN) if is_gold else badge("wrong doc", RED)
            st.markdown(
                f"<div class='nf-pagecard {cls}'>{badge(f'#{rank}', '#4A9EFF')} {tag}"
                f"<div class='small-mono'>{short_id(doc_id)} · page {page} · score {score:.3f}</div></div>",
                unsafe_allow_html=True,
            )
            with st.expander(f"View retrieved page (rank {rank})", expanded=(rank == 1)):
                render_pages_for_doc(
                    doc_id,
                    gold_key,
                    key_prefix=f"ret_{row['qid']}_{rank}",
                    highlight_page=page,
                    page_scores={page: score},
                )

    st.markdown("</div>", unsafe_allow_html=True)


def get_row_by_qid(source_df, qid):
    match = source_df[source_df["qid"] == qid]
    return None if match.empty else match.iloc[0]


def results_tab(sub_df, key_prefix):
    filtered = render_filters(sub_df, key_prefix)
    selected_qid = render_question_list(filtered, key_prefix)
    if selected_qid:
        row = get_row_by_qid(filtered, selected_qid)
        if row is not None:
            render_detail(row, gold_key)


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_correct, tab_incorrect = st.tabs(
    ["📊  OVERVIEW", f"✓  CORRECT ({n_correct:,})", f"✗  INCORRECT ({n_incorrect:,})"]
)

with tab_overview:
    oc1, oc2 = st.columns(2)
    with oc1:
        st.markdown('<div class="nf-section">Accuracy by hop type</div>', unsafe_allow_html=True)
        st.bar_chart(
            df.groupby("hop_type")["correct"].mean().mul(100).round(1), color=RED, height=260
        )
    with oc2:
        st.markdown('<div class="nf-section">Accuracy by answer modality</div>', unsafe_allow_html=True)
        st.bar_chart(
            df.groupby("modality")["correct"].mean().mul(100).round(1), color=RED, height=260
        )

    st.markdown('<div class="nf-section">Accuracy by reasoning pattern</div>', unsafe_allow_html=True)
    st.bar_chart(
        df.groupby("q_type")["correct"].mean().mul(100).round(1).sort_values(),
        color=RED,
        height=330,
    )

    st.markdown('<div class="nf-section">Retrieval hit rate</div>', unsafe_allow_html=True)
    st.progress(
        min(hit_rate / 100, 1.0),
        text=f"{hit_rate:.1f}% of questions had a gold document among the retrieved pages",
    )

with tab_correct:
    results_tab(df[df["correct"]], "correct")

with tab_incorrect:
    results_tab(df[~df["correct"]], "incorrect")
