import argparse
import json
import re

import pandas as pd

from src.config import (
    METRICS_DIR,
    MIN_REVIEWS,
    PROCESSED_DIR,
    RANKING_PRIOR_STRENGTH,
    create_output_folders,
)
from src.summaries import build_drug_summary, format_summary


def load_ranking_data():
    """Load training reviews for descriptive rankings."""
    tsv_path = PROCESSED_DIR / "train_clean.tsv"
    csv_path = PROCESSED_DIR / "train_clean.csv"

    if tsv_path.exists():
        path = tsv_path
        separator = "\t"
    elif csv_path.exists():
        path = csv_path
        separator = ","
    else:
        raise FileNotFoundError(
            "Cleaned training data not found. "
            "Run python -m src.clean_data first."
        )

    dataframe = pd.read_csv(
        path,
        sep=separator,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    required = [
        "review_id",
        "drug_name",
        "drug_name_normalized",
        "condition_normalized",
        "rating",
        "effectiveness",
        "side_effects",
        "sentiment_label",
        "benefits_review",
        "side_effects_review",
    ]

    missing = [
        column for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    dataframe["rating"] = pd.to_numeric(
        dataframe["rating"], errors="raise"
    )

    # Missing names cannot support condition-based rankings.
    valid = (
        dataframe["condition_normalized"].str.strip().ne("")
        & dataframe["drug_name_normalized"].str.strip().ne("")
    )

    return dataframe.loc[valid].copy()


def list_conditions(dataframe):
    return (
        dataframe["condition_normalized"]
        .value_counts()
        .rename_axis("condition")
        .reset_index(name="review_count")
    )


def rank_drugs(
    dataframe,
    condition,
    min_reviews=MIN_REVIEWS,
    prior_strength=RANKING_PRIOR_STRENGTH,
):
    """Return condition-specific, adjusted patient-rating rankings."""
    if min_reviews < 1:
        raise ValueError("min_reviews must be at least 1.")

    if prior_strength <= 0:
        raise ValueError("prior_strength must be greater than 0.")

    normalized_condition = re.sub(
        r"\s+", " ", condition
    ).strip().casefold()

    condition_reviews = dataframe.loc[
        dataframe["condition_normalized"].eq(normalized_condition)
    ].copy()

    if condition_reviews.empty:
        raise ValueError(
            f"No reviews found for condition '{condition}'. "
            "Use --list-conditions to inspect available names."
        )

    condition_mean = float(condition_reviews["rating"].mean())

    rankings = (
        condition_reviews.groupby("drug_name_normalized")
        .agg(
            drug_name=("drug_name", "first"),
            review_count=("rating", "size"),
            average_rating=("rating", "mean"),
        )
        .reset_index()
    )

    rankings["eligible"] = rankings["review_count"].ge(min_reviews)

    rankings["adjusted_score"] = (
        rankings["review_count"] * rankings["average_rating"]
        + prior_strength * condition_mean
    ) / (
        rankings["review_count"] + prior_strength
    )

    # Exclude sparse groups from the displayed ranking.
    eligible = rankings.loc[rankings["eligible"]].copy()

    eligible = eligible.sort_values(
        ["adjusted_score", "review_count", "drug_name_normalized"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    eligible["rank"] = range(1, len(eligible) + 1)
    eligible["condition"] = normalized_condition
    eligible["condition_average_rating"] = condition_mean
    eligible["eligible_drugs"] = len(eligible)

    # Keep the coverage information even when no drug qualifies.
    coverage = {
        "condition": normalized_condition,
        "condition_review_count": int(len(condition_reviews)),
        "condition_average_rating": condition_mean,
        "total_drugs": int(len(rankings)),
        "eligible_drugs": int(len(eligible)),
        "excluded_drugs": int((~rankings["eligible"]).sum()),
        "minimum_reviews": int(min_reviews),
        "prior_strength": float(prior_strength),
    }

    return eligible, condition_reviews, coverage


def main():
    parser = argparse.ArgumentParser(
        description="Rank drugs using condition-specific patient ratings."
    )

    parser.add_argument("--condition", type=str)
    parser.add_argument("--list-conditions", action="store_true")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--min-reviews", type=int, default=MIN_REVIEWS)
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=RANKING_PRIOR_STRENGTH,
    )

    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1.")

    create_output_folders()
    dataframe = load_ranking_data()

    if args.list_conditions:
        conditions = list_conditions(dataframe)

        print(conditions.head(30).to_string(index=False))

        conditions.to_csv(
            METRICS_DIR / "available_conditions.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("\nComplete list saved to available_conditions.csv")
        return

    if not args.condition:
        parser.error("Provide --condition or --list-conditions.")

    rankings, reviews, coverage = rank_drugs(
        dataframe,
        args.condition,
        min_reviews=args.min_reviews,
        prior_strength=args.prior_strength,
    )

    print("\nRANKING COVERAGE")
    for key, value in coverage.items():
        print(f"{key}: {value}")

    # Use a safe condition name and separate folder for each result.
    safe_name = re.sub(
        r"[^a-z0-9]+", "_", coverage["condition"]
    ).strip("_") or "condition"

    output_folder = METRICS_DIR / "rankings" / safe_name
    output_folder.mkdir(parents=True, exist_ok=True)

    rankings.to_csv(
        output_folder / "ranked_drugs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summaries = []
    text_sections = []

    if rankings.empty:
        message = (
            "Insufficient reviews for ranking at the selected "
            "minimum review count."
        )
        print("\n" + message)
        text_sections.append(message)
    else:
        print("\nRANKED DRUGS")

        display_columns = [
            "rank",
            "drug_name",
            "review_count",
            "average_rating",
            "adjusted_score",
        ]

        print(
            rankings[display_columns]
            .head(args.top)
            .round(3)
            .to_string(index=False)
        )

        for _, row in rankings.head(args.top).iterrows():
            drug_reviews = reviews.loc[
                reviews["drug_name_normalized"].eq(
                    row["drug_name_normalized"]
                )
            ]

            summary = build_drug_summary(drug_reviews, row)
            summaries.append(summary)

            description = format_summary(summary)
            text_sections.append(description)
            print("\n" + description)

    with (output_folder / "ranking_details.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {"coverage": coverage, "summaries": summaries},
            file,
            indent=2,
            ensure_ascii=False,
        )

    (output_folder / "written_summaries.txt").write_text(
        "\n\n".join(text_sections),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_folder}")
    print(
        "These results rank patient feedback; "
        "they are not personalized treatment recommendations."
    )


if __name__ == "__main__":
    main()