import json

import matplotlib

# Save charts without opening separate windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    GRAPHS_DIR,
    METRICS_DIR,
    PROCESSED_DIR,
    create_output_folders,
)


EFFECTIVENESS_ORDER = [
    "Ineffective",
    "Marginally Effective",
    "Moderately Effective",
    "Considerably Effective",
    "Highly Effective",
]

SEVERITY_ORDER = [
    "No Side Effects",
    "Mild Side Effects",
    "Moderate Side Effects",
    "Severe Side Effects",
    "Extremely Severe Side Effects",
]


def load_training_data():
    """Load cleaned training data; never load test data for EDA."""
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
        "drug_name_normalized",
        "condition_normalized",
        "rating",
        "effectiveness",
        "side_effects",
        "sentiment_label",
        "combined_review",
        "benefits_review",
        "side_effects_review",
        "comments_review",
    ]

    missing = [
        column for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(f"Missing cleaned columns: {missing}")

    if dataframe.empty:
        raise ValueError("Cleaned training data is empty.")

    dataframe["rating"] = pd.to_numeric(
        dataframe["rating"], errors="raise"
    )

    print(f"Loaded: {path}")
    return dataframe


def save_chart(filename):
    """Save and close the current chart."""
    path = GRAPHS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {path.name}")


def plot_categories(counts, title, filename):
    """Create a horizontal category-count chart."""
    plt.figure(figsize=(10, 5))
    positions = list(range(len(counts)))

    plt.barh(positions, counts.values, color="#2675AD")
    plt.yticks(positions, counts.index)
    plt.gca().invert_yaxis()

    plt.title(title)
    plt.xlabel("Number of training reviews")
    plt.ylabel("")

    save_chart(filename)


def create_charts(dataframe):
    # 1. Rating distribution.
    ratings = (
        dataframe["rating"]
        .value_counts()
        .reindex(range(1, 11), fill_value=0)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(ratings.index, ratings.values, color="#2675AD")
    plt.xticks(range(1, 11))
    plt.xlabel("Patient rating")
    plt.ylabel("Number of training reviews")
    plt.title("Patient Rating Distribution")
    save_chart("01_rating_distribution.png")

    # 2. Rating-derived sentiment distribution.
    sentiment = (
        dataframe["sentiment_label"]
        .value_counts()
        .reindex(["negative", "neutral", "positive"], fill_value=0)
    )

    plot_categories(
        sentiment,
        "Rating-Derived Sentiment Labels",
        "02_sentiment_distribution.png",
    )

    # 3. Patient-reported effectiveness.
    effectiveness = (
        dataframe["effectiveness"]
        .value_counts()
        .reindex(EFFECTIVENESS_ORDER, fill_value=0)
    )

    plot_categories(
        effectiveness,
        "Patient-Reported Effectiveness",
        "03_effectiveness_distribution.png",
    )

    # 4. Patient-reported side-effect severity.
    severity = (
        dataframe["side_effects"]
        .value_counts()
        .reindex(SEVERITY_ORDER, fill_value=0)
    )

    plot_categories(
        severity,
        "Patient-Reported Side-Effect Severity",
        "04_side_effect_severity.png",
    )

    # 5. Most reviewed conditions.
    valid_conditions = dataframe.loc[
        dataframe["condition_normalized"].str.strip().ne("")
    ]

    plot_categories(
        valid_conditions["condition_normalized"]
        .value_counts()
        .head(10),
        "10 Most Reviewed Conditions",
        "05_top_conditions.png",
    )

    # 6. Most reviewed drugs.
    valid_drugs = dataframe.loc[
        dataframe["drug_name_normalized"].str.strip().ne("")
    ]

    plot_categories(
        valid_drugs["drug_name_normalized"]
        .value_counts()
        .head(10),
        "10 Most Reviewed Drugs",
        "06_top_drugs.png",
    )

    # 7. Review length.
    word_counts = (
        dataframe["combined_review"].str.split().str.len()
    )

    plt.figure(figsize=(10, 5))
    plt.hist(word_counts, bins=35, color="#2675AD", edgecolor="white")
    plt.xlabel("Words in combined review")
    plt.ylabel("Number of training reviews")
    plt.title("Review Length Distribution")
    save_chart("07_review_lengths.png")

    # 8. Average ratings by reported severity.
    plt.figure(figsize=(11, 5))
    sns.boxplot(
        data=dataframe,
        x="side_effects",
        y="rating",
        order=SEVERITY_ORDER,
        color="#8EBADC",
    )
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("Reported side-effect severity")
    plt.ylabel("Patient rating")
    plt.title("Ratings by Reported Side-Effect Severity")
    save_chart("08_rating_by_severity.png")

    # 9. Number of reviews supporting each drug-condition pair.
    pairs = dataframe.loc[
        dataframe["condition_normalized"].str.strip().ne("")
        & dataframe["drug_name_normalized"].str.strip().ne("")
    ]

    pair_counts = pairs.groupby(
        ["condition_normalized", "drug_name_normalized"]
    ).size()

    # Log-spaced bins make sparse and larger groups visible.
    if not pair_counts.empty:
        import numpy as np

        bins = np.geomspace(
            0.5,
            float(pair_counts.max()) + 0.5,
            num=20,
        )

        plt.figure(figsize=(10, 5))
        plt.hist(pair_counts, bins=bins, color="#2675AD")
        plt.xscale("log")
        plt.xlabel("Reviews per drug-condition pair (log scale)")
        plt.ylabel("Number of drug-condition pairs")
        plt.title("Evidence Available for Condition-Based Rankings")
        save_chart("09_ranking_review_coverage.png")


def create_reports(dataframe):
    total = len(dataframe)

    valid_pairs = dataframe.loc[
        dataframe["condition_normalized"].str.strip().ne("")
        & dataframe["drug_name_normalized"].str.strip().ne("")
    ]

    pair_summary = (
        valid_pairs.groupby(
            ["condition_normalized", "drug_name_normalized"]
        )
        .agg(
            review_count=("rating", "size"),
            average_rating=("rating", "mean"),
        )
        .reset_index()
        .sort_values("review_count", ascending=False)
    )

    pair_summary.to_csv(
        METRICS_DIR / "drug_condition_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Coverage thresholds are diagnostics, not a final policy.
    coverage = {
        str(threshold): int(
            pair_summary["review_count"].ge(threshold).sum()
        )
        for threshold in [1, 3, 5, 10]
    }

    severity_summary = (
        dataframe.groupby("side_effects")["rating"]
        .agg(["count", "mean", "median"])
        .reindex(SEVERITY_ORDER)
    )

    severity_summary.to_csv(
        METRICS_DIR / "severity_rating_summary.csv",
        encoding="utf-8-sig",
    )

    positive_count = int(
        dataframe["sentiment_label"].eq("positive").sum()
    )

    missing_text = {
        column: int(dataframe[column].str.strip().eq("").sum())
        for column in [
            "benefits_review",
            "side_effects_review",
            "comments_review",
        ]
    }

    valid_conditions = dataframe.loc[
        dataframe["condition_normalized"].str.strip().ne("")
    ]

    condition_counts = (
        valid_conditions["condition_normalized"].value_counts()
    )

    stats = {
        "training_reviews": int(total),
        "unique_nonblank_drugs": int(
            dataframe.loc[
                dataframe["drug_name_normalized"].str.strip().ne(""),
                "drug_name_normalized",
            ].nunique()
        ),
        "unique_nonblank_condition_strings": int(
            valid_conditions["condition_normalized"].nunique()
        ),
        "average_rating": float(dataframe["rating"].mean()),
        "median_rating": float(dataframe["rating"].median()),
        "positive_proxy_label_percent": float(
            100 * positive_count / total
        ),
        "median_review_words": float(
            dataframe["combined_review"]
            .str.split()
            .str.len()
            .median()
        ),
        "missing_review_text": missing_text,
        "drug_condition_pairs": int(len(pair_summary)),
        "pairs_meeting_minimum_review_count": coverage,
    }

    with (METRICS_DIR / "eda_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(stats, file, indent=2)

    findings = [
        "TRAINING DATA: EXPLORATORY FINDINGS",
        "",
        f"Training reviews: {total:,}",
        f"Unique nonblank drug names: "
        f"{stats['unique_nonblank_drugs']:,}",
        f"Unique normalized condition strings: "
        f"{stats['unique_nonblank_condition_strings']:,}",
        f"Average patient rating: "
        f"{stats['average_rating']:.2f}/10",
        f"Median patient rating: "
        f"{stats['median_rating']:.1f}/10",
        "",
        f"Positive rating-derived labels account for "
        f"{stats['positive_proxy_label_percent']:.1f}% of reviews.",
        "These labels describe rating groups, not independently "
        "annotated sentiment.",
        "",
        f"The median combined review contains "
        f"{stats['median_review_words']:.0f} words.",
        "",
        "Missing or blank descriptions:",
    ]

    for column, count in missing_text.items():
        findings.append(f"- {column}: {count:,}")

    if not condition_counts.empty:
        findings.extend([
            "",
            f"The most reviewed normalized condition string is "
            f"'{condition_counts.index[0]}' "
            f"with {int(condition_counts.iloc[0]):,} reviews.",
        ])

    findings.extend([
        "",
        "Ranking evidence coverage:",
        f"Total drug-condition pairs: {len(pair_summary):,}",
    ])

    for threshold, count in coverage.items():
        findings.append(
            f"- Pairs with at least {threshold} reviews: {count:,}"
        )

    findings.extend([
        "",
        "Interpretation:",
        "Review counts describe dataset representation, not "
        "disease prevalence or drug popularity in the population.",
        "Effectiveness and severity are patient-reported labels.",
        "Similar condition strings have not automatically been "
        "merged into a single medical condition.",
        "Small review groups provide limited evidence for rankings.",
        "All findings use training data only.",
    ])

    findings_text = "\n".join(findings)

    with (METRICS_DIR / "eda_findings.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write(findings_text)

    print("\n" + findings_text)


def main():
    create_output_folders()
    sns.set_theme(style="whitegrid")

    dataframe = load_training_data()
    create_charts(dataframe)
    create_reports(dataframe)

    print("\nEDA complete.")
    print(f"Charts: {GRAPHS_DIR}")
    print(f"Reports: {METRICS_DIR}")


if __name__ == "__main__":
    main()