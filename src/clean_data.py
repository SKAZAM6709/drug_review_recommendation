import html
import json
import re
import unicodedata

import pandas as pd

from src.config import (
    METRICS_DIR,
    PROCESSED_DIR,
    TEXT_COLUMNS,
    create_output_folders,
)
from src.data_loader import load_raw_data


COLUMN_NAMES = {
    "Unnamed: 0": "review_id",
    "urlDrugName": "drug_name",
    "sideEffects": "side_effects",
    "benefitsReview": "benefits_review",
    "sideEffectsReview": "side_effects_review",
    "commentsReview": "comments_review",
}

REVIEW_COLUMNS = [
    "benefits_review",
    "side_effects_review",
    "comments_review",
]

EFFECTIVENESS_LABELS = {
    "Ineffective",
    "Marginally Effective",
    "Moderately Effective",
    "Considerably Effective",
    "Highly Effective",
}

SIDE_EFFECT_LABELS = {
    "No Side Effects",
    "Mild Side Effects",
    "Moderate Side Effects",
    "Severe Side Effects",
    "Extremely Severe Side Effects",
}


def clean_text(value):
    """Clean formatting while preserving words and punctuation."""
    if pd.isna(value):
        return ""

    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML tags but retain their surrounding text.
    text = re.sub(r"<[^>]*>", " ", text)

    # Replace repeated whitespace with a single space.
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(value):
    """Normalize case and spacing without merging medical concepts."""
    return clean_text(value).casefold()


def make_review_key(dataframe):
    """
    Identify records with the same drug, condition, and review text.

    Ratings and category labels are intentionally excluded so identical
    text with different labels cannot remain across the two splits.
    """
    key_columns = [
        "drug_name_normalized",
        "condition_normalized",
        *REVIEW_COLUMNS,
    ]

    normalized = dataframe[key_columns].copy()

    for column in REVIEW_COLUMNS:
        normalized[column] = normalized[column].str.casefold()

    # Tuples avoid ambiguous text concatenation.
    return pd.Series(
        list(normalized.itertuples(index=False, name=None)),
        index=dataframe.index,
    )


def prepare_dataset(raw_data, split_name):
    """Prepare one split without learning anything from the other."""
    dataframe = raw_data.rename(columns=COLUMN_NAMES).copy()
    dataframe["source_split"] = split_name

    # Preserve source names for traceability.
    dataframe["drug_name_original"] = dataframe["drug_name"]
    dataframe["condition_original"] = dataframe["condition"]

    dataframe["drug_name"] = dataframe["drug_name"].map(clean_text)
    dataframe["condition"] = dataframe["condition"].map(clean_text)

    dataframe["drug_name_normalized"] = (
        dataframe["drug_name"].map(normalize_name)
    )
    dataframe["condition_normalized"] = (
        dataframe["condition"].map(normalize_name)
    )

    dataframe["condition_missing"] = (
        dataframe["condition_normalized"].eq("")
    )

    # Validate ratings instead of replacing invalid values.
    ratings = pd.to_numeric(dataframe["rating"], errors="coerce")

    invalid_ratings = (
        ratings.isna()
        | ~ratings.between(1, 10)
        | ratings.mod(1).ne(0)
    )

    if invalid_ratings.any():
        raise ValueError(
            f"{split_name}: {int(invalid_ratings.sum())} invalid ratings."
        )

    dataframe["rating"] = ratings.astype(int)

    # Normalize category formatting and validate known labels.
    for column, allowed_labels in [
        ("effectiveness", EFFECTIVENESS_LABELS),
        ("side_effects", SIDE_EFFECT_LABELS),
    ]:
        dataframe[column] = dataframe[column].map(clean_text)

        unexpected = set(dataframe[column]) - allowed_labels

        if unexpected:
            raise ValueError(
                f"{split_name}: unexpected {column} labels: "
                f"{sorted(unexpected)}"
            )

    # Missing descriptions are not equivalent to "No Side Effects".
    for column in REVIEW_COLUMNS:
        dataframe[column] = dataframe[column].map(clean_text)
        dataframe[f"{column}_missing"] = dataframe[column].eq("")

    # Separate inputs for the three future modeling tasks.
    dataframe["effectiveness_text"] = (
        dataframe["benefits_review"]
        + " "
        + dataframe["comments_review"]
    ).str.strip()

    dataframe["severity_text"] = (
        dataframe["side_effects_review"]
        + " "
        + dataframe["comments_review"]
    ).str.strip()

    dataframe["combined_review"] = (
        dataframe["benefits_review"]
        + " "
        + dataframe["side_effects_review"]
        + " "
        + dataframe["comments_review"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    # These labels are derived from ratings, not human annotation.
    dataframe["sentiment_label"] = dataframe["rating"].map(
        lambda rating: (
            "negative"
            if rating <= 4
            else "neutral"
            if rating <= 6
            else "positive"
        )
    )

    return dataframe


def main():
    create_output_folders()
    raw_train, raw_test = load_raw_data()

    train = prepare_dataset(raw_train, "train")
    test = prepare_dataset(raw_test, "test")

    original_train_count = len(train)

    # Keep the supplied test split intact.
    test_keys = set(make_review_key(test))
    train_keys = make_review_key(train)

    overlap_mask = train_keys.isin(test_keys)
    removed_overlap = int(overlap_mask.sum())

    train = train.loc[~overlap_mask].copy()

    # Remove repeated training review content.
    duplicate_mask = make_review_key(train).duplicated(keep="first")
    removed_duplicates = int(duplicate_mask.sum())

    train = train.loc[~duplicate_mask].copy()

    # Exclude entirely empty training reviews from text modeling.
    empty_mask = train["combined_review"].eq("")
    removed_empty = int(empty_mask.sum())

    train = train.loc[~empty_mask].reset_index(drop=True)
    test = test.reset_index(drop=True)

    # Verify that no matching review keys remain across splits.
    remaining_overlap = set(make_review_key(train)).intersection(
        set(make_review_key(test))
    )

    if remaining_overlap:
        raise RuntimeError("Matching review content remains across splits.")

    train_path = PROCESSED_DIR / "train_clean.csv"
    test_path = PROCESSED_DIR / "test_clean.csv"

    train.to_csv(train_path, index=False, encoding="utf-8-sig")
    test.to_csv(test_path, index=False, encoding="utf-8-sig")

    # Training-only inventory for later manual condition mapping.
    condition_inventory = (
        train.loc[~train["condition_missing"]]
        .groupby("condition_normalized")
        .agg(
            review_count=("review_id", "size"),
            unique_drugs=("drug_name_normalized", "nunique"),
        )
        .reset_index()
        .sort_values(
            ["review_count", "condition_normalized"],
            ascending=[False, True],
        )
    )

    inventory_path = PROCESSED_DIR / "condition_inventory.csv"
    condition_inventory.to_csv(
        inventory_path,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "original_train_rows": int(original_train_count),
        "original_test_rows": int(len(raw_test)),
        "training_rows_removed_for_test_overlap": removed_overlap,
        "training_duplicates_removed": removed_duplicates,
        "empty_training_reviews_removed": removed_empty,
        "final_train_rows": int(len(train)),
        "final_test_rows": int(len(test)),
        "remaining_cross_split_review_matches": len(remaining_overlap),
        "duplicate_test_review_keys": int(
            make_review_key(test).duplicated().sum()
        ),
        "empty_test_reviews": int(test["combined_review"].eq("").sum()),
        "missing_train_conditions": int(train["condition_missing"].sum()),
        "missing_test_conditions": int(test["condition_missing"].sum()),
        "train_sentiment_distribution": {
            label: int(count)
            for label, count in
            train["sentiment_label"].value_counts().items()
        },
        "notes": [
            "Test records were retained.",
            "Name normalization does not merge condition synonyms.",
            "Missing conditions are retained for text classification.",
            "Missing conditions must be excluded from condition rankings.",
            "Overlap checks detect normalized exact matches, "
            "not near-duplicate reviews.",
            "Sentiment labels are rating-derived proxies.",
        ],
    }

    report_path = METRICS_DIR / "cleaning_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("DATA CLEANING COMPLETE")
    print(f"Original training rows: {original_train_count}")
    print(f"Training rows matching test reviews: {removed_overlap}")
    print(f"Other training duplicates removed: {removed_duplicates}")
    print(f"Empty training reviews removed: {removed_empty}")
    print(f"Final training rows: {len(train)}")
    print(f"Final test rows: {len(test)}")

    print("\nTraining sentiment labels:")
    print(train["sentiment_label"].value_counts().to_string())

    print("\nFiles saved:")
    for path in [
        train_path,
        test_path,
        inventory_path,
        report_path,
    ]:
        print(path)


if __name__ == "__main__":
    main()