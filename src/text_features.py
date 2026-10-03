from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


TASKS = {
    "sentiment": {
        "text_column": "combined_review",
        "target_column": "sentiment_label",
        "labels": ["negative", "neutral", "positive"],
    },
    "effectiveness": {
        "text_column": "effectiveness_text",
        "target_column": "effectiveness",
        "labels": [
            "Ineffective",
            "Marginally Effective",
            "Moderately Effective",
            "Considerably Effective",
            "Highly Effective",
        ],
    },
    "severity": {
        "text_column": "severity_text",
        "target_column": "side_effects",
        "labels": [
            "No Side Effects",
            "Mild Side Effects",
            "Moderate Side Effects",
            "Severe Side Effects",
            "Extremely Severe Side Effects",
        ],
    },
}


def build_text_pipeline(classifier):
    """Combine text feature extraction and a classifier."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_features=20000,
                sublinear_tf=True,
                # Preserve negation words, including "no" and "not".
                stop_words=None,
            ),
        ),
        ("classifier", classifier),
    ])