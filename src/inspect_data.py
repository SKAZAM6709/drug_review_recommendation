import json

import pandas as pd

from src.config import (
    METRICS_DIR,
    REQUIRED_COLUMNS,
    TEXT_COLUMNS,
    create_output_folders,
)
from src.data_loader import load_raw_data


def inspect_dataset(dataframe, name):
    """Calculate and print basic data-quality statistics."""
    ratings = pd.to_numeric(dataframe["rating"], errors="coerce")

    valid_ratings = (
        ratings.notna()
        & ratings.between(1, 10)
        & ratings.mod(1).eq(0)
    )

    text_missing = {
        column: int(
            dataframe[column]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )
        for column in TEXT_COLUMNS
    }

    # Exclude the identifier when checking duplicate content.
    content_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column != "Unnamed: 0"
    ]

    summary = {
        "dataset": name,
        "rows": int(len(dataframe)),
        "columns": list(dataframe.columns),
        "unique_drugs": int(dataframe["urlDrugName"].nunique()),
        "unique_raw_conditions": int(dataframe["condition"].nunique()),
        "missing_values": {
            column: int(count)
            for column, count in dataframe.isna().sum().items()
        },
        "missing_or_blank_text": text_missing,
        "duplicate_ids": int(
            dataframe["Unnamed: 0"].duplicated().sum()
        ),
        "duplicate_content_rows": int(
            dataframe.duplicated(subset=content_columns).sum()
        ),
        "invalid_rating_rows": int((~valid_ratings).sum()),
        "rating_distribution": {
            str(value): int(count)
            for value, count in ratings.value_counts().sort_index().items()
        },
        "effectiveness_distribution": {
            str(value): int(count)
            for value, count in
            dataframe["effectiveness"].value_counts().items()
        },
        "side_effect_distribution": {
            str(value): int(count)
            for value, count in
            dataframe["sideEffects"].value_counts().items()
        },
    }

    print(f"\n{name.upper()} DATASET")
    print(f"Rows: {summary['rows']}")
    print(f"Columns: {len(summary['columns'])}")
    print(f"Unique drugs: {summary['unique_drugs']}")
    print(
        "Unique condition strings:",
        summary["unique_raw_conditions"],
    )
    print(f"Invalid ratings: {summary['invalid_rating_rows']}")
    print(
        "Duplicate content rows:",
        summary["duplicate_content_rows"],
    )

    print("\nMissing values:")
    print(dataframe.isna().sum().to_string())

    print("\nMissing or blank review text:")
    for column, count in text_missing.items():
        print(f"{column}: {count}")

    print("\nRating distribution:")
    print(ratings.value_counts().sort_index().to_string())

    print("\nEffectiveness distribution:")
    print(dataframe["effectiveness"].value_counts().to_string())

    print("\nSide-effect severity distribution:")
    print(dataframe["sideEffects"].value_counts().to_string())

    return summary


def check_split_overlap(train_data, test_data):
    """Check shared identifiers and exact content across the splits."""
    content_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column != "Unnamed: 0"
    ]

    shared_ids = set(train_data["Unnamed: 0"]).intersection(
        set(test_data["Unnamed: 0"])
    )

    # Match content exactly, including missing values.
    # Deduplicate test keys so each training row is counted once.
    test_keys = test_data[content_columns].drop_duplicates()

    matched_train = train_data[content_columns].merge(
        test_keys,
        on=content_columns,
        how="inner",
        validate="many_to_one",
    )

    overlap = {
        "shared_record_ids": int(len(shared_ids)),
        "training_rows_matching_test_content": int(len(matched_train)),
    }

    print("\nTRAIN / TEST OVERLAP")
    print(f"Shared record IDs: {overlap['shared_record_ids']}")
    print(
        "Training rows matching test content:",
        overlap["training_rows_matching_test_content"],
    )

    return overlap


def main():
    create_output_folders()
    train_data, test_data = load_raw_data()

    train_summary = inspect_dataset(train_data, "Training")
    test_summary = inspect_dataset(test_data, "Test")
    overlap = check_split_overlap(train_data, test_data)

    report = {
        "training": train_summary,
        "test": test_summary,
        "split_overlap": overlap,
    }

    report_path = METRICS_DIR / "data_inspection.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nTotal reviews: {len(train_data) + len(test_data)}")
    print(f"Inspection report saved to: {report_path}")
    print("\nInspection complete. Original files were not modified.")


if __name__ == "__main__":
    main()