import plotly.express as px
import streamlit as st

from src.ui import apply_theme
from src.recommend import load_ranking_data


st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide",
)

apply_theme()

CHART_COLORS = [
    "#176B75",
    "#168A8A",
    "#2675AD",
    "#73B7B8",
    "#E79B36",
]


@st.cache_data
def get_data():
    return load_ranking_data()


def style_chart(figure):
    """Apply consistent colors and layout to charts."""
    figure.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#172B4D"),
        title=dict(font=dict(color="#102A43")),
        colorway=CHART_COLORS,
        margin=dict(l=30, r=20, t=60, b=40),
    )

    figure.update_xaxes(
        gridcolor="#E8EEF5",
        zerolinecolor="#DCE5EF",
        automargin=True,
    )

    figure.update_yaxes(
        gridcolor="#E8EEF5",
        zerolinecolor="#DCE5EF",
        automargin=True,
    )

    return figure


def category_chart(data, column, title, order=None):
    """Plot category counts for the current filters."""
    counts = data[column].value_counts()

    if order is not None:
        counts = counts.reindex(order, fill_value=0)

    table = counts.rename_axis("category").reset_index(name="reviews")

    figure = px.bar(
        table,
        x="category",
        y="reviews",
        title=title,
        color_discrete_sequence=["#176B75"],
        labels={
            "category": "",
            "reviews": "Review count",
        },
    )

    figure.update_layout(showlegend=False)

    return style_chart(figure)


st.title("Data Explorer")

st.caption(
    "Explore cleaned training reviews. Counts describe this dataset, "
    "not population prevalence or clinical effectiveness."
)

try:
    data = get_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

if data.empty:
    st.warning(
        "No reviews with valid drug and condition names are available."
    )
    st.stop()

conditions = sorted(
    data["condition_normalized"].unique().tolist()
)

with st.sidebar:
    st.subheader("Review filters")

    selected_condition = st.selectbox(
        "Condition",
        options=[None] + conditions,
        format_func=lambda value: (
            "All conditions" if value is None else value
        ),
    )

    condition_data = data

    if selected_condition is not None:
        condition_data = data.loc[
            data["condition_normalized"].eq(selected_condition)
        ]

    drugs = sorted(
        condition_data["drug_name_normalized"].unique().tolist()
    )

    selected_drug = st.selectbox(
        "Drug",
        options=[None] + drugs,
        format_func=lambda value: (
            "All drugs" if value is None else value
        ),
    )

    rating_range = st.slider(
        "Patient rating range",
        min_value=1,
        max_value=10,
        value=(1, 10),
    )

    selected_sentiments = st.multiselect(
        "Rating-derived sentiment",
        options=["negative", "neutral", "positive"],
        default=["negative", "neutral", "positive"],
    )

    search_text = st.text_input(
        "Search within review text",
        placeholder="For example: nausea",
    )

filtered = condition_data.copy()

if selected_drug is not None:
    filtered = filtered.loc[
        filtered["drug_name_normalized"].eq(selected_drug)
    ]

filtered = filtered.loc[
    filtered["rating"].between(*rating_range)
    & filtered["sentiment_label"].isin(selected_sentiments)
]

if search_text.strip():
    filtered = filtered.loc[
        filtered["combined_review"].str.contains(
            search_text.strip(),
            case=False,
            regex=False,
            na=False,
        )
    ]

st.caption(
    "Text search matches words literally. It does not distinguish "
    "'nausea' from 'no nausea'."
)

if filtered.empty:
    st.warning(
        "No reviews match these filters. Adjust the sidebar filters."
    )
    st.stop()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Matching reviews", f"{len(filtered):,}")

col2.metric(
    "Drug names",
    f"{filtered['drug_name_normalized'].nunique():,}",
)

col3.metric(
    "Condition strings",
    f"{filtered['condition_normalized'].nunique():,}",
)

col4.metric(
    "Average rating",
    f"{filtered['rating'].mean():.2f}/10",
)

st.subheader("Written overview")

positive_percentage = (
    100 * filtered["sentiment_label"].eq("positive").mean()
)

st.write(
    f"The current selection contains **{len(filtered):,} reviews**, "
    f"with an average rating of "
    f"**{filtered['rating'].mean():.2f}/10**. "
    f"**{positive_percentage:.1f}%** have positive rating-derived labels. "
    "These findings describe the filtered reviews."
)

st.divider()

left, right = st.columns(2)

with left:
    # Explicit categories ensure one bar for each integer rating.
    rating_counts = (
        filtered["rating"]
        .value_counts()
        .reindex(range(1, 11), fill_value=0)
        .rename_axis("rating")
        .reset_index(name="reviews")
    )

    rating_figure = px.bar(
        rating_counts,
        x="rating",
        y="reviews",
        title="Patient Ratings",
        color_discrete_sequence=["#176B75"],
        labels={
            "rating": "Patient rating",
            "reviews": "Review count",
        },
    )

    rating_figure.update_xaxes(
        tickmode="linear",
        dtick=1,
        range=[0.5, 10.5],
    )

    st.plotly_chart(
        style_chart(rating_figure),
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        category_chart(
            filtered,
            "sentiment_label",
            "Rating-Derived Sentiment",
            order=["negative", "neutral", "positive"],
        ),
        use_container_width=True,
    )

left, right = st.columns(2)

with left:
    st.plotly_chart(
        category_chart(
            filtered,
            "effectiveness",
            "Patient-Reported Effectiveness",
            order=[
                "Ineffective",
                "Marginally Effective",
                "Moderately Effective",
                "Considerably Effective",
                "Highly Effective",
            ],
        ),
        use_container_width=True,
    )

with right:
    st.plotly_chart(
        category_chart(
            filtered,
            "side_effects",
            "Patient-Reported Side-Effect Severity",
            order=[
                "No Side Effects",
                "Mild Side Effects",
                "Moderate Side Effects",
                "Severe Side Effects",
                "Extremely Severe Side Effects",
            ],
        ),
        use_container_width=True,
    )

st.subheader("Most reviewed drugs in this selection")

drug_counts = (
    filtered["drug_name_normalized"]
    .value_counts()
    .head(10)
    .rename_axis("drug")
    .reset_index(name="reviews")
)

figure = px.bar(
    drug_counts.sort_values("reviews"),
    x="reviews",
    y="drug",
    orientation="h",
    color_discrete_sequence=["#168A8A"],
    labels={
        "reviews": "Review count",
        "drug": "",
    },
)

st.plotly_chart(
    style_chart(figure),
    use_container_width=True,
)

st.divider()
st.subheader("Review records")

display_columns = [
    "review_id",
    "drug_name",
    "condition",
    "rating",
    "effectiveness",
    "side_effects",
    "sentiment_label",
]

st.dataframe(
    filtered[display_columns],
    hide_index=True,
    use_container_width=True,
)

st.download_button(
    label="Download matching reviews as TSV",
    data=filtered.to_csv(
        sep="\t",
        index=False,
    ).encode("utf-8-sig"),
    file_name="filtered_training_reviews.tsv",
    mime="text/tab-separated-values",
)

st.subheader("Read an individual review")

# Select by row position, rather than assuming identifiers are unique.
review_records = filtered.reset_index(drop=True)

selected_position = st.selectbox(
    "Select a matching review",
    options=range(len(review_records)),
    format_func=lambda position: (
        f"Review {review_records.iloc[position]['review_id']} — "
        f"{review_records.iloc[position]['drug_name']} — "
        f"Rating {review_records.iloc[position]['rating']}/10"
    ),
)

review = review_records.iloc[selected_position]

st.write(
    f"**Condition:** {review['condition']}  \n"
    f"**Reported effectiveness:** {review['effectiveness']}  \n"
    f"**Reported severity:** {review['side_effects']}"
)

for title, column in [
    ("Benefits description", "benefits_review"),
    ("Side-effect description", "side_effects_review"),
    ("Additional comments", "comments_review"),
]:
    with st.expander(title, expanded=True):
        value = review[column]
        text = "" if value is None else str(value).strip()

        # Display review content as plain text.
        if text:
            st.text(text)
        else:
            st.write("No written description provided.")