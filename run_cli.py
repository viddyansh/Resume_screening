"""
run_cli.py
----------
Command-line interface for ResumeRank AI.
Useful for batch processing or testing without the Streamlit UI.

Usage:
    python run_cli.py --jd sample_data/job_description.txt \
                      --resumes sample_data/ \
                      --method tfidf \
                      --top 3
"""

import argparse
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(__file__))

from utils import setup_logging, get_matched_keywords
from parser import parse_resume
from similarity import rank_candidates, get_top_n, score_label
from preprocessing import extract_keywords

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

SEPARATOR = "─" * 72


def load_resumes_from_dir(directory: str):
    """Load all .pdf and .txt files from a directory."""
    texts, names = [], []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith((".pdf", ".txt")) and fname != "job_description.txt":
            fpath = os.path.join(directory, fname)
            try:
                text = parse_resume(fpath)
                texts.append(text)
                names.append(fname)
                logger.info(f"  Parsed: {fname} ({len(text)} chars)")
            except Exception as e:
                logger.error(f"  Failed to parse {fname}: {e}")
    return texts, names


def print_results(ranked, jd_keywords, top_n):
    print(f"\n{'='*72}")
    print("  RESUMERANK AI — CANDIDATE RANKING RESULTS")
    print(f"{'='*72}\n")

    for r in get_top_n(ranked, top_n):
        medal = "🥇" if r["rank"] == 1 else "🥈" if r["rank"] == 2 else "🥉" if r["rank"] == 3 else f"#{r['rank']}"
        print(f"  {medal}  RANK {r['rank']}  —  {r['name']}")
        print(f"     Score:  {r['score_pct']}  ({score_label(r['score'])})")

        matched, missing = get_matched_keywords(jd_keywords, r["raw_text"])
        if matched:
            print(f"     ✅ Matched keywords: {', '.join(matched[:10])}")
        if missing:
            print(f"     ❌ Missing keywords: {', '.join(missing[:8])}")
        print(SEPARATOR)

    print(f"\n  Total candidates analysed: {len(ranked)}")
    print(f"  Showing top {min(top_n, len(ranked))} results\n")


def main():
    parser = argparse.ArgumentParser(
        description="ResumeRank AI — NLP-based resume screening CLI"
    )
    parser.add_argument("--jd",      required=True, help="Path to job description file (.txt)")
    parser.add_argument("--resumes", required=True, help="Path to resume file or directory of resumes")
    parser.add_argument("--method",  default="tfidf", choices=["tfidf", "sbert"],
                        help="Vectorization method: tfidf (default) or sbert")
    parser.add_argument("--top",     type=int, default=5, help="Number of top candidates to display")
    args = parser.parse_args()

    # ── Load job description ─────────────────────────────────────────────
    logger.info(f"Loading job description: {args.jd}")
    try:
        jd_text = parse_resume(args.jd)
    except Exception as e:
        print(f"ERROR: Could not load job description: {e}")
        sys.exit(1)

    # ── Load resumes ─────────────────────────────────────────────────────
    resume_texts, resume_names = [], []

    if os.path.isdir(args.resumes):
        logger.info(f"Loading resumes from directory: {args.resumes}")
        resume_texts, resume_names = load_resumes_from_dir(args.resumes)
    elif os.path.isfile(args.resumes):
        try:
            text = parse_resume(args.resumes)
            resume_texts = [text]
            resume_names = [os.path.basename(args.resumes)]
        except Exception as e:
            print(f"ERROR: Could not load resume: {e}")
            sys.exit(1)
    else:
        print(f"ERROR: '{args.resumes}' is not a valid file or directory.")
        sys.exit(1)

    if not resume_texts:
        print("ERROR: No valid resumes found.")
        sys.exit(1)

    print(f"\n  Loaded {len(resume_texts)} resume(s). Method: {args.method.upper()}")
    print(SEPARATOR)

    # ── Rank ─────────────────────────────────────────────────────────────
    try:
        ranked = rank_candidates(
            job_description_text=jd_text,
            resume_texts=resume_texts,
            resume_names=resume_names,
            method=args.method,
            preprocess=True,
        )
    except Exception as e:
        print(f"ERROR: Ranking failed: {e}")
        sys.exit(1)

    jd_keywords = extract_keywords(jd_text, top_n=20)
    print_results(ranked, jd_keywords, args.top)


if __name__ == "__main__":
    main()
