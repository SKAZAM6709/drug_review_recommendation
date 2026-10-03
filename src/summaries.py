import pandas as pd


def category_summary(series):
    """Describe the most frequent category, including ties."""
    counts = series.value_counts()

    if counts.empty:
        return "No category information is available."

    largest = int(counts.max())
    categories = counts[counts.eq(largest)].index.tolist()
    percentage = 100 * largest / len(series)

    if len(categories) == 1:
        return (
            f"The most frequent category is '{categories[0]}' "
            f"({largest}/{len(series)} reviews, {percentage:.1f}%)."
        )

    names = ", ".join(f"'{name}'" for name in categories)

    return (
        f"The most frequent categories are tied: {names}, "
        f"each appearing in {largest}/{len(series)} reviews "
        f"({percentage:.1f}%)."
    )


def review_excerpts(reviews, column, limit=3, max_characters=400):
    """
    Select distinct excerpts around the median rating.
    These are examples, not automatically verified common experiences.
    """
    if reviews.empty:
        return []

    examples = reviews.copy()

    examples[column] = examples[column].fillna("").astype(str).str.strip()
    examples = examples.loc[examples[column].ne("")]

    examples = examples.drop_duplicates(subset=[column])

    median_rating = reviews["rating"].median()
    examples["distance_from_median"] = (
        examples["rating"] - median_rating
    ).abs()

    examples = examples.sort_values(
        ["distance_from_median", "review_id"],
        kind="stable",
    ).head(limit)

    output = []

    for _, row in examples.iterrows():
        text = row[column]

        if len(text) > max_characters:
            text = text[:max_characters].rstrip() + "…"

        output.append({
            "review_id": str(row["review_id"]),
            "rating": int(row["rating"]),
            "text": text,
        })

    return output


def build_drug_summary(reviews, ranking_row):
    """Create a descriptive summary for one ranked drug."""
    count = len(reviews)

    if count == 0:
        raise ValueError("Cannot summarize an empty review group.")

    positive_count = int(
        reviews["sentiment_label"].eq("positive").sum()
    )
    positive_percentage = 100 * positive_count / count

    effectiveness = category_summary(reviews["effectiveness"])
    severity = category_summary(reviews["side_effects"])

    description = (
        f"For '{ranking_row['condition']}', "
        f"'{ranking_row['drug_name']}' has {count} training reviews. "
        f"The average patient rating is "
        f"{ranking_row['average_rating']:.2f}/10, "
        f"and its adjusted rating is "
        f"{ranking_row['adjusted_score']:.2f}/10. "
        f"It ranks {int(ranking_row['rank'])} among "
        f"{int(ranking_row['eligible_drugs'])} eligible drugs. "
        f"{positive_percentage:.1f}% of reviews have positive "
        f"rating-derived labels."
    )

    return {
        "drug_name": str(ranking_row["drug_name"]),
        "condition": str(ranking_row["condition"]),
        "description": description,
        "effectiveness_summary": effectiveness,
        "side_effect_severity_summary": severity,
        "ranking_explanation": (
            "The score combines this drug's average rating with "
            "the condition's average rating. Smaller review groups "
            "are adjusted more strongly toward the condition average. "
            "Rank is determined by adjusted score, then review count, "
            "then drug name."
        ),
        "benefits_excerpts": review_excerpts(
            reviews, "benefits_review"
        ),
        "side_effects_excerpts": review_excerpts(
            reviews, "side_effects_review"
        ),
        "notes": [
            "Excerpts are selected around the group's median rating.",
            "Excerpts do not establish how common a benefit "
            "or side effect is.",
            "Ratings and categories are patient-reported.",
            "Positive labels are derived from ratings.",
        ],
    }


def format_summary(summary):
    """Convert a structured summary into readable text."""
    lines = [
        summary["description"],
        "",
        "REPORTED EFFECTIVENESS",
        summary["effectiveness_summary"],
        "",
        "REPORTED SIDE-EFFECT SEVERITY",
        summary["side_effect_severity_summary"],
        "",
        "WHY THIS RANK?",
        summary["ranking_explanation"],
    ]

    for title, key in [
        ("BENEFITS REVIEW EXCERPTS", "benefits_excerpts"),
        ("SIDE-EFFECT REVIEW EXCERPTS", "side_effects_excerpts"),
    ]:
        lines.extend(["", title])

        excerpts = summary[key]

        if not excerpts:
            lines.append("No written descriptions are available.")

        for excerpt in excerpts:
            lines.append(
                f"- Review {excerpt['review_id']} "
                f"(rating {excerpt['rating']}/10): "
                f"{excerpt['text']}"
            )

    lines.extend(["", "NOTES"])
    lines.extend(f"- {note}" for note in summary["notes"])

    return "\n".join(lines)