"""
vectorizer.py
-------------
Feature extraction module implementing two approaches:

(A) TF-IDF  — sklearn TfidfVectorizer (baseline, always available)
(B) SBERT   — Sentence-BERT via sentence-transformers (optional, richer semantics)

Both expose the same interface:
    vectorize(corpus: List[str]) -> np.ndarray   shape (n_docs, n_features)
"""

import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ===========================================================================
# (A) TF-IDF Vectorizer
# ===========================================================================

class TFIDFVectorizer:
    """
    Wraps sklearn's TfidfVectorizer.

    Mathematical background
    -----------------------
    TF(t, d)  = (count of term t in doc d) / (total terms in doc d)
    IDF(t)    = log((1 + N) / (1 + df(t))) + 1    [sklearn 'smooth' variant]
    TF-IDF(t, d) = TF(t, d) × IDF(t)

    Each document is represented as a sparse vector of TF-IDF weights,
    then L2-normalised so cosine similarity reduces to a dot product.

    Advantages:
    - Fast, interpretable, no GPU required
    - Works well on keyword-rich domain text (resumes, JDs)

    Limitations:
    - Ignores word order and semantics ("Python expert" ≠ "expert in Python")
    - High-dimensional sparse space
    - Out-of-vocabulary problem
    """

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,   # unigrams + bigrams for richer matching
            sublinear_tf=sublinear_tf, # log(1+tf) dampens high-freq terms
            norm="l2",                 # cosine-ready output
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
        )
        self._fitted = False

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        """
        Fit on corpus and return dense matrix (n_docs × n_features).

        Args:
            corpus: List of preprocessed text documents.

        Returns:
            Dense numpy array of TF-IDF vectors.
        """
        if not corpus or all(not t.strip() for t in corpus):
            raise ValueError("Corpus is empty or all documents are blank.")
        matrix = self._vectorizer.fit_transform(corpus)
        self._fitted = True
        return matrix.toarray()

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts using already-fitted vectorizer."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        return self._vectorizer.transform(texts).toarray()

    @property
    def feature_names(self) -> List[str]:
        """Return vocabulary feature names."""
        if not self._fitted:
            return []
        return self._vectorizer.get_feature_names_out().tolist()


# ===========================================================================
# (B) Sentence-BERT Vectorizer
# ===========================================================================

class SBERTVectorizer:
    """
    Produces dense semantic embeddings using a pre-trained Sentence-BERT model.

    Mathematical background
    -----------------------
    Sentence-BERT fine-tunes BERT with a siamese/triplet network so that
    semantically similar sentences have high cosine similarity in the
    embedding space (typically 384- or 768-dim).

    Advantages over TF-IDF:
    - Captures semantics: "machine learning engineer" ≈ "ML developer"
    - Dense, low-dimensional, continuous space
    - Context-aware (transformer attention)

    Limitations:
    - Requires ~80–500 MB model download
    - Slower inference (GPU helps)
    - May hallucinate similarity for very domain-specific jargon
    """

    # Lightweight, fast model — good default for CPU environments
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"SBERT model '{self.model_name}' loaded.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        """
        Encode all documents.  (No 'fitting' needed for SBERT.)

        Args:
            corpus: List of text documents (raw or preprocessed).

        Returns:
            Dense numpy array of shape (n_docs, embedding_dim).
        """
        if self._model is None:
            self._load_model()
        if not corpus:
            raise ValueError("Corpus is empty.")
        embeddings = self._model.encode(
            corpus,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-norm → cosine = dot product
        )
        return embeddings

    def transform(self, texts: List[str]) -> np.ndarray:
        """Encode new texts (same as fit_transform for SBERT)."""
        return self.fit_transform(texts)


# ===========================================================================
# Factory function
# ===========================================================================

def get_vectorizer(method: str = "tfidf", **kwargs):
    """
    Return a vectorizer instance based on the chosen method.

    Args:
        method: "tfidf" or "sbert"
        **kwargs: Passed to the chosen vectorizer constructor.

    Returns:
        TFIDFVectorizer or SBERTVectorizer instance.
    """
    method = method.lower().strip()
    if method == "tfidf":
        return TFIDFVectorizer(**kwargs)
    elif method in ("sbert", "bert", "sentence-bert", "sentence_bert"):
        return SBERTVectorizer(**kwargs)
    else:
        raise ValueError(f"Unknown vectorizer method: '{method}'. Use 'tfidf' or 'sbert'.")


def is_sbert_available() -> bool:
    """Check whether sentence-transformers is installed."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False
