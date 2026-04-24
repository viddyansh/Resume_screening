"""
preprocessing.py
----------------
Text cleaning and NLP preprocessing pipeline.
Steps: lowercase → strip HTML/URLs → remove special chars →
       tokenise → remove stopwords → lemmatize → rejoin.
"""

import re
import string
import logging
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK bootstrap (download required corpora once at import time)
# ---------------------------------------------------------------------------
import nltk

def _ensure_nltk_data():
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    }
    for path, pkg in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:
                logger.warning(f"Could not download NLTK resource '{pkg}': {e}")

_ensure_nltk_data()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_STOPWORDS = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Individual cleaning helpers
# ---------------------------------------------------------------------------

def remove_html_tags(text: str) -> str:
    """Strip HTML/XML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove http/https URLs and bare www.* links."""
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)
    return text


def remove_emails(text: str) -> str:
    """Remove email addresses."""
    return re.sub(r"\S+@\S+\.\S+", " ", text)


def remove_phone_numbers(text: str) -> str:
    """Remove common phone number patterns."""
    return re.sub(r"[\+\(]?[1-9][0-9 .\-\(\)]{7,}[0-9]", " ", text)


def fix_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_special_characters(text: str) -> str:
    """
    Remove punctuation and non-alphanumeric characters,
    but preserve hyphens inside words (e.g. 'self-motivated').
    """
    # Keep letters, digits, spaces, and intra-word hyphens
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    # Remove standalone hyphens that are not inside words
    text = re.sub(r"(?<!\w)-|-(?!\w)", " ", text)
    return text


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter out English stopwords from a token list."""
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatize each token using WordNet."""
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_text(
    text: str,
    do_lemmatize: bool = True,
    remove_stops: bool = True,
) -> str:
    """
    Full preprocessing pipeline for a single document.

    Pipeline:
        1. Lowercase
        2. Remove HTML tags
        3. Remove URLs / emails / phone numbers
        4. Remove special characters
        5. Fix whitespace
        6. Tokenize (NLTK word_tokenize)
        7. Remove stopwords  (optional)
        8. Lemmatize          (optional)
        9. Rejoin to string

    Args:
        text:          Raw input text.
        do_lemmatize:  Apply WordNet lemmatization (default True).
        remove_stops:  Remove English stopwords (default True).

    Returns:
        Cleaned, preprocessed text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2-4. Remove noise
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_phone_numbers(text)
    text = remove_special_characters(text)

    # 5. Fix whitespace
    text = fix_whitespace(text)

    # 6. Tokenize
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Keep only alphabetic tokens (filter out stray numbers/symbols)
    tokens = [t for t in tokens if t.isalpha()]

    # 7. Remove stopwords
    if remove_stops:
        tokens = remove_stopwords(tokens)

    # 8. Lemmatize
    if do_lemmatize:
        tokens = lemmatize(tokens)

    # 9. Rejoin
    return " ".join(tokens)


def preprocess_corpus(
    texts: List[str],
    do_lemmatize: bool = True,
    remove_stops: bool = True,
) -> List[str]:
    """
    Apply preprocess_text to a list of documents.

    Args:
        texts:        List of raw text strings.
        do_lemmatize: Apply lemmatization.
        remove_stops: Remove stopwords.

    Returns:
        List of cleaned text strings (same length as input).
    """
    processed = []
    for i, text in enumerate(texts):
        try:
            clean = preprocess_text(text, do_lemmatize=do_lemmatize, remove_stops=remove_stops)
            processed.append(clean)
        except Exception as e:
            logger.error(f"Error preprocessing document {i}: {e}")
            processed.append("")
    return processed


def extract_keywords(text: str, top_n: int = 20) -> List[str]:
    """
    Extract the most informative keywords from a text using
    simple frequency analysis on preprocessed tokens.
    Used for keyword highlighting in the UI.

    Args:
        text:  Raw or preprocessed text.
        top_n: Number of top keywords to return.

    Returns:
        List of top keywords (strings).
    """
    from collections import Counter
    cleaned = preprocess_text(text, do_lemmatize=True, remove_stops=True)
    tokens = cleaned.split()
    # Filter very short tokens
    tokens = [t for t in tokens if len(t) >= 3]
    counter = Counter(tokens)
    return [word for word, _ in counter.most_common(top_n)]
