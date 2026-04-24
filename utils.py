"""
utils.py
--------
Shared helper utilities used across the project:
- Logging setup
- Keyword highlighting
- Score chart generation (matplotlib)
- File validation
- Session state helpers for Streamlit
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple


# ===========================================================================
# Logging
# ===========================================================================

def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ===========================================================================
# File validation
# ===========================================================================

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_MB = 10


def validate_file(filename: str, file_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate an uploaded file by name and size.

    Args:
        filename:   Original filename.
        file_bytes: Raw bytes of the file.

    Returns:
        (is_valid: bool, error_message: str)
        error_message is "" if valid.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, (
            f"'{filename}' has unsupported format '{ext}'. "
            f"Accepted: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, (
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{MAX_FILE_SIZE_MB} MB limit."
        )

    if len(file_bytes) == 0:
        return False, f"'{filename}' is empty."

    return True, ""


def validate_job_description(text: str) -> Tuple[bool, str]:
    """
    Validate that a job description has sufficient content.

    Returns:
        (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Job description cannot be empty."
    word_count = len(text.split())
    if word_count < 10:
        return False, (
            f"Job description is too short ({word_count} words). "
            "Please provide a more detailed description."
        )
    return True, ""


# ===========================================================================
# Keyword highlighting
# ===========================================================================

def highlight_keywords(
    text: str,
    keywords: List[str],
    highlight_tag: str = "**",
) -> str:
    """
    Wrap occurrences of keywords in the text with a highlight marker.

    Args:
        text:          Raw text to search in.
        keywords:      List of keywords to highlight.
        highlight_tag: Markdown bold marker or HTML tag prefix.

    Returns:
        Text with keywords wrapped in highlight markers.
    """
    if not keywords or not text:
        return text

    # Build pattern: match whole words, case-insensitive
    escaped = [re.escape(kw) for kw in sorted(keywords, key=len, reverse=True)]
    pattern = r"\b(" + "|".join(escaped) + r")\b"

    def replacer(m):
        return f"{highlight_tag}{m.group(0)}{highlight_tag}"

    return re.sub(pattern, replacer, text, flags=re.IGNORECASE)


def get_matched_keywords(
    jd_keywords: List[str], resume_text: str
) -> Tuple[List[str], List[str]]:
    """
    Find which job-description keywords appear in the resume text.

    Args:
        jd_keywords:  Keywords extracted from the job description.
        resume_text:  Raw or cleaned resume text.

    Returns:
        (matched_keywords, missing_keywords)
    """
    resume_lower = resume_text.lower()
    matched = []
    missing = []
    for kw in jd_keywords:
        # Check whole-word presence
        pattern = r"\b" + re.escape(kw.lower()) + r"\b"
        if re.search(pattern, resume_lower):
            matched.append(kw)
        else:
            missing.append(kw)
    return matched, missing


# ===========================================================================
# Score chart (matplotlib)
# ===========================================================================

def create_score_chart(
    ranked_results: List[Dict[str, Any]],
    top_n: int = 10,
) -> "matplotlib.figure.Figure":
    """
    Create a horizontal bar chart of candidate scores.

    Args:
        ranked_results: Output of similarity.rank_candidates().
        top_n:          How many candidates to show.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    data = ranked_results[:top_n]
    names  = [d["name"] for d in reversed(data)]
    scores = [d["score"] * 100 for d in reversed(data)]
    ranks  = [d["rank"] for d in reversed(data)]

    # Colour: green for top 3, amber for next 2, grey for rest
    colors = []
    for r in reversed(ranks):
        if r <= 3:
            colors.append("#22c55e")   # green
        elif r <= 5:
            colors.append("#f59e0b")   # amber
        else:
            colors.append("#6b7280")   # grey

    fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.55)))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    bars = ax.barh(names, scores, color=colors, height=0.6, edgecolor="#334155")

    # Score labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}%",
            va="center",
            ha="left",
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Similarity Score (%)", color="#94a3b8", fontsize=10)
    ax.set_title("Candidate Ranking by Resume Relevance", color="white", fontsize=13, pad=14)
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")

    # Legend
    legend_items = [
        mpatches.Patch(color="#22c55e", label="Top 3 (Excellent)"),
        mpatches.Patch(color="#f59e0b", label="Rank 4-5 (Strong)"),
        mpatches.Patch(color="#6b7280", label="Other candidates"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        facecolor="#0f172a",
        edgecolor="#334155",
        labelcolor="white",
        fontsize=8,
    )

    fig.tight_layout()
    return fig


# ===========================================================================
# Text truncation helper
# ===========================================================================

def truncate_text(text: str, max_chars: int = 500) -> str:
    """Return text truncated to max_chars with ellipsis if needed."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


# ===========================================================================
# Comparison table: TF-IDF vs SBERT
# ===========================================================================

TFIDF_VS_SBERT = {
    "headers": ["Aspect", "TF-IDF", "Sentence-BERT"],
    "rows": [
        ["Representation",    "Sparse, high-dim",            "Dense, 384-768 dim"],
        ["Semantics",         "None (bag-of-words)",          "Rich contextual understanding"],
        ["Speed",             "Very fast (ms)",               "Slower (seconds per batch)"],
        ["GPU required",      "No",                           "Optional (CPU works)"],
        ["OOV handling",      "Ignores unknown words",        "Handles via subword/context"],
        ["Synonym matching",  "No",                           "Yes"],
        ["Interpretability",  "High (feature = word/bigram)", "Low (latent embedding)"],
        ["Setup complexity",  "None (built into sklearn)",    "Requires model download (~80 MB)"],
        ["Best for",          "Keyword-dense job specs",      "Semantically varied resumes"],
    ],
}
