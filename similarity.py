"""
similarity.py
-------------
Cosine similarity computation and candidate ranking.

Cosine Similarity Formula
--------------------------
    cos(θ) = (A · B) / (||A|| × ||B||)

Where:
    A, B    = document vectors (TF-IDF weights or SBERT embeddings)
    A · B   = dot product (element-wise multiply, then sum)
    ||A||   = Euclidean (L2) norm of vector A

Range: [-1, 1] for raw vectors; [0, 1] when vectors are non-negative (TF-IDF)
       or L2-normalised (SBERT with normalize_embeddings=True).

Why cosine over Euclidean distance?
    Cosine is length-invariant — a short résumé that perfectly matches the
    job description scores as high as a long one that repeats the same terms.
    This is critical for fair resume ranking.
"""

import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def cosine_similarity_matrix(
    query_vector: np.ndarray, document_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between one query vector and N document vectors.

    Both inputs should already be L2-normalised (TFIDFVectorizer and
    SBERTVectorizer both normalise by default), so this reduces to a
    simple dot product:  cos(θ) = q · d   (since ||q|| = ||d|| = 1).

    Args:
        query_vector:    1D array of shape (features,)
        document_matrix: 2D array of shape (n_docs, features)

    Returns:
        1D array of shape (n_docs,) with similarity scores ∈ [0, 1].
    """
    query = query_vector.flatten()

    # Normalise defensively in case caller skipped it
    q_norm = np.linalg.norm(query)
    if q_norm == 0:
        logger.warning("Query vector has zero norm; returning zeros.")
        return np.zeros(document_matrix.shape[0])
    query = query / q_norm

    d_norms = np.linalg.norm(document_matrix, axis=1, keepdims=True)
    # Avoid division by zero for empty documents
    d_norms = np.where(d_norms == 0, 1e-10, d_norms)
    doc_matrix_normed = document_matrix / d_norms

    scores = doc_matrix_normed @ query  # shape (n_docs,)
    # Clip to [0, 1] — small negative values arise from floating-point noise
    return np.clip(scores, 0.0, 1.0)


def rank_candidates(
    job_description_text: str,
    resume_texts: List[str],
    resume_names: List[str],
    method: str = "tfidf",
    preprocess: bool = True,
) -> List[Dict[str, Any]]:
    """
    End-to-end pipeline: preprocess → vectorize → score → rank.

    Args:
        job_description_text: Raw text of the job description.
        resume_texts:         List of raw resume texts.
        resume_names:         Corresponding filenames/labels.
        method:               "tfidf" or "sbert".
        preprocess:           Whether to clean text before vectorizing.

    Returns:
        List of dicts sorted by score descending:
        [
          {
            "rank":       1,
            "name":       "alice_resume.pdf",
            "score":      0.87,
            "score_pct":  "87.3%",
            "raw_text":   "...",
          },
          ...
        ]
    """
    from preprocessing import preprocess_corpus
    from vectorizer import get_vectorizer

    if not resume_texts:
        raise ValueError("No resume texts provided.")
    if len(resume_texts) != len(resume_names):
        raise ValueError("resume_texts and resume_names must have the same length.")

    # ------------------------------------------------------------------
    # 1. Preprocessing
    # ------------------------------------------------------------------
    if preprocess:
        logger.info("Preprocessing documents...")
        all_raw = [job_description_text] + resume_texts
        all_clean = preprocess_corpus(all_raw)
        jd_clean = all_clean[0]
        resumes_clean = all_clean[1:]
    else:
        jd_clean = job_description_text
        resumes_clean = resume_texts

    # Warn if any resume ended up blank after cleaning
    empty_indices = [i for i, t in enumerate(resumes_clean) if not t.strip()]
    if empty_indices:
        logger.warning(
            f"Resumes at indices {empty_indices} are empty after preprocessing."
        )

    # ------------------------------------------------------------------
    # 2. Vectorization
    # ------------------------------------------------------------------
    logger.info(f"Vectorizing with method='{method}'...")
    vectorizer = get_vectorizer(method)

    corpus = [jd_clean] + resumes_clean
    try:
        vectors = vectorizer.fit_transform(corpus)
    except Exception as e:
        raise RuntimeError(f"Vectorization failed: {e}")

    jd_vector = vectors[0]          # shape (features,)
    resume_vectors = vectors[1:]    # shape (n_resumes, features)

    # ------------------------------------------------------------------
    # 3. Similarity scoring
    # ------------------------------------------------------------------
    logger.info("Computing cosine similarities...")
    scores = cosine_similarity_matrix(jd_vector, resume_vectors)

    # ------------------------------------------------------------------
    # 4. Rank
    # ------------------------------------------------------------------
    ranked_indices = np.argsort(scores)[::-1]  # descending

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        results.append(
            {
                "rank":       rank,
                "name":       resume_names[idx],
                "score":      float(scores[idx]),
                "score_pct":  f"{scores[idx] * 100:.1f}%",
                "raw_text":   resume_texts[idx],
                "clean_text": resumes_clean[idx],
            }
        )

    logger.info(f"Ranking complete. Top candidate: {results[0]['name']} ({results[0]['score_pct']})")
    return results


def get_top_n(
    ranked_results: List[Dict[str, Any]], n: int = 5
) -> List[Dict[str, Any]]:
    """
    Return the top-N candidates from a pre-ranked list.

    Args:
        ranked_results: Output of rank_candidates().
        n:              Number of top candidates to return.

    Returns:
        Sliced list of top-N candidates.
    """
    return ranked_results[:max(1, n)]


def score_label(score: float) -> str:
    """
    Human-readable relevance label based on score thresholds.

    Args:
        score: Float ∈ [0, 1]

    Returns:
        One of: "Excellent", "Strong", "Good", "Fair", "Weak"
    """
    if score >= 0.75:
        return "Excellent"
    elif score >= 0.55:
        return "Strong"
    elif score >= 0.35:
        return "Good"
    elif score >= 0.18:
        return "Fair"
    else:
        return "Weak"
