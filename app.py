"""
app.py
------
Streamlit web application for AI-Based Resume Screening and Ranking.

Run with:
    streamlit run app.py
"""

import io
import os
import sys
import logging
import streamlit as st

# Ensure local modules are importable when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    setup_logging,
    validate_file,
    validate_job_description,
    create_score_chart,
    get_matched_keywords,
    truncate_text,
    TFIDF_VS_SBERT,
)
from parser import parse_resume
from similarity import rank_candidates, get_top_n, score_label
from preprocessing import extract_keywords
from vectorizer import is_sbert_available

setup_logging(logging.WARNING)  # suppress verbose logs in UI

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeRank AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, professional theme
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0f1e;
    color: #e2e8f0;
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}

/* ── Headings ── */
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 14px 18px;
}

/* ── Text area ── */
textarea {
    background: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

/* ── Rank badge ── */
.rank-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 50%;
    width: 32px; height: 32px;
    text-align: center;
    line-height: 32px;
    font-weight: 700;
    font-size: 14px;
}
.rank-top { background: linear-gradient(135deg, #059669, #10b981) !important; }
.rank-good { background: linear-gradient(135deg, #d97706, #f59e0b) !important; }

/* ── Score bar ── */
.score-bar-outer {
    background: #1e293b;
    border-radius: 999px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    margin-top: 4px;
}
.score-bar-inner {
    height: 10px;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── Card ── */
.resume-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.resume-card:hover { border-color: #6366f1; }
.resume-card.top-card { border-color: #10b981; }

/* ── Keyword pill ── */
.kw-pill {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    border: 1px solid #2563eb;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 12px;
    margin: 2px;
}
.kw-missing {
    background: #3f1515;
    color: #f87171;
    border-color: #dc2626;
}

/* ── Info box ── */
.info-box {
    background: #0f2744;
    border: 1px solid #1d4ed8;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 14px;
    line-height: 1.7;
}

/* ── Table ── */
table { width: 100%; border-collapse: collapse; }
th { background: #1e293b; color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; padding: 10px 14px; border-bottom: 1px solid #334155; }
td { padding: 10px 14px; border-bottom: 1px solid #1e293b; font-size: 14px; vertical-align: middle; }
tr:hover td { background: #1e293b; }

/* ── Divider ── */
hr { border-color: #1e293b !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("## 🎯")
with col_title:
    st.markdown("# ResumeRank AI")
    st.caption("AI-powered resume screening · NLP-based similarity ranking")

st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    method_options = ["TF-IDF (Baseline)"]
    if is_sbert_available():
        method_options.append("Sentence-BERT (Semantic)")
    method_choice = st.selectbox("Vectorization Method", method_options)
    method = "tfidf" if method_choice.startswith("TF-IDF") else "sbert"

    top_n = st.slider("Show Top N Candidates", min_value=1, max_value=20, value=5)
    show_keywords = st.toggle("Show Keyword Analysis", value=True)
    show_chart = st.toggle("Show Score Chart", value=True)
    show_snippet = st.toggle("Show Resume Snippet", value=True)

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown(
        """
        **ResumeRank AI** extracts text from uploaded resumes,
        computes NLP-based similarity scores against the job
        description, and ranks candidates by relevance.

        **Supported formats:** PDF, TXT  
        **Max file size:** 10 MB each
        """
    )

    if not is_sbert_available():
        st.info(
            "💡 Install `sentence-transformers` to unlock Sentence-BERT "
            "semantic matching:\n```\npip install sentence-transformers\n```"
        )

# ---------------------------------------------------------------------------
# Main layout: two columns
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1], gap="large")

# ── Job Description input ──────────────────────────────────────────────────
with left_col:
    st.markdown("### 📋 Job Description")
    jd_text = st.text_area(
        label="Paste the job description here",
        height=280,
        placeholder=(
            "e.g.\n\nWe are looking for a Senior Python Developer with 5+ years "
            "of experience in machine learning, NLP, and REST API development. "
            "Strong knowledge of scikit-learn, PyTorch, and AWS is required..."
        ),
        label_visibility="collapsed",
    )

# ── Resume upload ──────────────────────────────────────────────────────────
with right_col:
    st.markdown("### 📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT resumes",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.markdown(f"&nbsp;&nbsp;• `{f.name}` — {size_kb:.1f} KB")

# ---------------------------------------------------------------------------
# Rank button
# ---------------------------------------------------------------------------
st.markdown("---")
run_col, _, info_col = st.columns([2, 1, 3])
with run_col:
    run_btn = st.button(
        "🚀 Rank Candidates",
        type="primary",
        use_container_width=True,
        disabled=(not jd_text.strip() or not uploaded_files),
    )
with info_col:
    if not jd_text.strip():
        st.warning("⬅️ Please enter a job description.")
    elif not uploaded_files:
        st.warning("⬅️ Please upload at least one resume.")

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
if run_btn:
    # ── Input validation ─────────────────────────────────────────────────
    jd_valid, jd_err = validate_job_description(jd_text)
    if not jd_valid:
        st.error(f"❌ {jd_err}")
        st.stop()

    resume_texts, resume_names, parse_errors = [], [], []

    for f in uploaded_files:
        raw_bytes = f.getvalue()
        is_valid, err_msg = validate_file(f.name, raw_bytes)
        if not is_valid:
            parse_errors.append(f"**{f.name}**: {err_msg}")
            continue
        try:
            text = parse_resume(raw_bytes, filename=f.name)
            if not text.strip():
                parse_errors.append(f"**{f.name}**: No extractable text found.")
                continue
            resume_texts.append(text)
            resume_names.append(f.name)
        except Exception as e:
            parse_errors.append(f"**{f.name}**: {e}")

    if parse_errors:
        st.warning("⚠️ Some files could not be parsed:")
        for err in parse_errors:
            st.markdown(f"&nbsp;&nbsp;• {err}")

    if not resume_texts:
        st.error("❌ No valid resumes could be processed. Please check your files.")
        st.stop()

    # ── Run ranking ───────────────────────────────────────────────────────
    progress = st.progress(0, text="Initialising…")

    try:
        progress.progress(10, text="Preprocessing text…")
        progress.progress(30, text=f"Vectorizing with {method.upper()}…")

        ranked = rank_candidates(
            job_description_text=jd_text,
            resume_texts=resume_texts,
            resume_names=resume_names,
            method=method,
            preprocess=True,
        )
        progress.progress(80, text="Computing rankings…")

    except Exception as e:
        progress.empty()
        st.error(f"❌ Ranking failed: {e}")
        st.stop()

    progress.progress(100, text="Done!")
    progress.empty()

    # ── Extract JD keywords (for highlighting) ────────────────────────────
    jd_keywords = extract_keywords(jd_text, top_n=25)

    # ── Summary metrics ───────────────────────────────────────────────────
    st.markdown("## 📊 Results")
    top_results = get_top_n(ranked, top_n)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Resumes Analysed", len(ranked))
    m2.metric("Top Score", f"{ranked[0]['score'] * 100:.1f}%")
    m3.metric("Method Used", method.upper())
    m4.metric("JD Keywords Found", len(jd_keywords))

    st.markdown("---")

    # ── Score chart ───────────────────────────────────────────────────────
    if show_chart:
        st.markdown("### 📈 Score Distribution")
        try:
            fig = create_score_chart(ranked, top_n=min(top_n, len(ranked)))
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Chart generation failed: {e}")
        st.markdown("---")

    # ── Results table (compact) ───────────────────────────────────────────
    st.markdown("### 🏆 Candidate Rankings")

    table_html = """
    <table>
      <thead><tr>
        <th>Rank</th><th>Resume</th><th>Score</th><th>Match Level</th><th>Progress</th>
      </tr></thead><tbody>
    """
    for r in get_top_n(ranked, top_n):
        pct = r["score"] * 100
        label = score_label(r["score"])
        badge_cls = "rank-top" if r["rank"] <= 3 else ("rank-good" if r["rank"] <= 5 else "")
        bar_color = (
            "#10b981" if r["rank"] <= 3 else
            "#f59e0b" if r["rank"] <= 5 else "#6b7280"
        )
        table_html += f"""
        <tr>
          <td><span class="rank-badge {badge_cls}">{r['rank']}</span></td>
          <td><strong>{r['name']}</strong></td>
          <td><strong>{pct:.1f}%</strong></td>
          <td>{label}</td>
          <td>
            <div class="score-bar-outer">
              <div class="score-bar-inner" style="width:{pct:.1f}%;background:{bar_color}"></div>
            </div>
          </td>
        </tr>"""
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("---")

    # ── Per-candidate detail cards ────────────────────────────────────────
    st.markdown("### 🗂️ Candidate Detail")

    for r in get_top_n(ranked, top_n):
        is_top = r["rank"] <= 3
        card_cls = "resume-card top-card" if is_top else "resume-card"
        medal = "🥇" if r["rank"] == 1 else "🥈" if r["rank"] == 2 else "🥉" if r["rank"] == 3 else f"#{r['rank']}"

        with st.expander(
            f"{medal}  **{r['name']}**  —  {r['score_pct']}  ({score_label(r['score'])})",
            expanded=is_top,
        ):
            detail_left, detail_right = st.columns([1, 1])

            with detail_left:
                st.markdown("**Score**")
                st.progress(r["score"], text=r["score_pct"])
                st.metric("Rank", f"#{r['rank']}", delta=None)
                st.metric("Match Level", score_label(r["score"]))

            if show_keywords:
                matched, missing = get_matched_keywords(jd_keywords, r["raw_text"])
                with detail_right:
                    st.markdown("**Matched Keywords**")
                    if matched:
                        pills = " ".join(f'<span class="kw-pill">{k}</span>' for k in matched[:15])
                        st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.caption("No keyword matches found.")

                    if missing:
                        st.markdown("**Missing Keywords**")
                        pills_m = " ".join(f'<span class="kw-pill kw-missing">{k}</span>' for k in missing[:12])
                        st.markdown(pills_m, unsafe_allow_html=True)

            if show_snippet:
                st.markdown("**Resume Snippet**")
                snippet = truncate_text(r["raw_text"], 600)
                st.markdown(
                    f'<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;'
                    f'padding:12px;font-size:13px;font-family:monospace;white-space:pre-wrap;'
                    f'color:#94a3b8;">{snippet}</div>',
                    unsafe_allow_html=True,
                )

    # ── Methodology explanation ───────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 How It Works — Methodology & Theory", expanded=False):
        st.markdown(
            """
### TF-IDF (Term Frequency–Inverse Document Frequency)

**TF(t, d)** = (occurrences of term *t* in document *d*) / (total terms in *d*)

**IDF(t)** = log( (1 + N) / (1 + df(t)) ) + 1  &nbsp;*(sklearn smooth variant)*

**TF-IDF(t, d)** = TF(t, d) × IDF(t)

Each resume and the job description are converted into a sparse vector of TF-IDF
weights, then L2-normalised. Common words (high df) get low IDF weights; rare,
discriminating terms (e.g. *"PyTorch"*, *"transformer"*) get high weights.

---

### Cosine Similarity

$$cos(\\theta) = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\cdot \\|\\mathbf{B}\\|}$$

- **A** = job description vector  
- **B** = resume vector  
- Result ∈ [0, 1] — **1** means identical content, **0** means no overlap  

**Why cosine over Euclidean distance?**  
Cosine is length-invariant. A 200-word resume that perfectly matches the JD
scores the same as a 2000-word resume with the same matched terms. Length
should not penalise concise, relevant resumes.

---

### Why TF-IDF Reflects Relevance

Resumes that use domain-specific keywords present in the job description
receive high TF-IDF weights for those terms. Because IDF down-weights ubiquitous
words (e.g. *"the"*, *"is"*), the similarity score is driven by rare, meaningful
domain terms — which is exactly what recruiters look for.

---

### TF-IDF vs Sentence-BERT
"""
        )
        headers = TFIDF_VS_SBERT["headers"]
        rows = TFIDF_VS_SBERT["rows"]
        tbl = "| " + " | ".join(headers) + " |\n"
        tbl += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            tbl += "| " + " | ".join(row) + " |\n"
        st.markdown(tbl)

        st.markdown(
            """
**Recommendation:** Use TF-IDF for speed and interpretability. Switch to
Sentence-BERT when resumes use varied vocabulary and you want semantic
matching (e.g., *"developed predictive models"* matching *"machine learning
engineer"*).
"""
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#475569;font-size:12px;">'
    "ResumeRank AI · Built with Streamlit · NLP-powered candidate screening"
    "</div>",
    unsafe_allow_html=True,
)
