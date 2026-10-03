import plotly.express as px
import streamlit as st

from src.ui import apply_theme
from src.recommend import load_ranking_data
from src.summaries import category_summary


st.set_page_config(
    page_title="Drug Comparison",
    page_icon="⚖️",
    layout="wide",
)

apply_theme()

DRUG_COLORS = ["#176B75", "#2675AD", "#E79B36"]

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

EFFECTIVENESS_COLORS = {
    "Ineffective": "#B84A48",
    "Marginally Effective": "#E79B36",
    "Moderately Effective": "#2675AD",
    "Considerably Effective": "#168A8A",
    "Highly Effective": "#10565E",
}

SEVERITY_COLORS = {
    "No Side Effects": "#10565E",
    "Mild Side Effects": "#168A8A",
    "Moderate Side Effects": "#2675AD",
    "Severe Side Effects": "#E79B36",
    "Extremely Severe Side Effects": "#B84A48",
}


@st.cache_data
def get_data():
    return load_ranking_data()


def style_chart(figure):
    figure.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#172B4D"),
        title=dict(font=dict(color="#102A43")),
        margin=dict(l=25, r=25, t=65, b=45),
        legend_title_text="",
    )

    figure.update_xaxes(
        gridcolor="#E8EEF5",
        automargin=True,
    )

    figure.update_yaxes(
        gridcolor="#E8EEF5",
        automargin=True,
    )

    return figure


st.title("Drug Comparison")

st.caption(
    "Compare patient-reported experiences within the same condition. "
    "Results use cleaned training reviews."
)

try:
    data = get_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

if data.empty:
    st.warning("No eligible data is available.")
    st.stop()

condition = st.selectbox(
    "Select a condition",
    data["condition_normalized"].value_counts().index.tolist(),
)

condition_data = data.loc[
    data["condition_normalized"].eq(condition)
].copy()

drugs = (
    condition_data["drug_name_normalized"]
    .value_counts()
    .index.tolist()
)

if len(drugs) < 2:
    st.info(
        "This condition has reviews for fewer than two drugs. "
        "Choose another condition."
    )
    st.stop()

selected = st.multiselect(
    "Select two or three drugs",
    options=drugs,
    default=drugs[:2],
)

if len(selected) < 2:
    st.info("Select at least two drugs.")
    st.stop()

if len(selected) > 3:
    st.warning("Select no more than three drugs.")
    st.stop()

comparison_data = condition_data.loc[
    condition_data["drug_name_normalized"].isin(selected)
].copy()

summary = (
    comparison_data.groupby("drug_name_normalized")
    .agg(
        review_count=("rating", "size"),
        average_rating=("rating", "mean"),
        median_rating=("rating", "median"),
        positive_label_share=(
            "sentiment_label",
            lambda values: 100 * values.eq("positive").mean(),
        ),
    )
    .reindex(selected)
    .reset_index()
)

st.subheader("Comparison Overview")

metric_columns = st.columns(len(selected))

for column, drug in zip(metric_columns, selected):
    row = summary.loc[
        summary["drug_name_normalized"].eq(drug)
    ].iloc[0]

    with column:
        st.metric(
            label=drug,
            value=f"{row['average_rating']:.2f}/10",
        )
        st.caption(f"Based on {int(row['review_count'])} reviews.")

st.dataframe(
    summary.round(2),
    hide_index=True,
    use_container_width=True,
)

st.caption(
    "Positive label share is a percentage derived from ratings. "
    "Different review counts and patient populations limit comparisons. "
    "Higher ratings do not establish a clinically superior treatment."
)

small_groups = summary.loc[
    summary["review_count"].lt(3),
    "drug_name_normalized",
].tolist()

if small_groups:
    st.warning(
        "Limited review evidence for: "
        + ", ".join(small_groups)
        + ". Interpret these comparisons cautiously."
    )

st.divider()
st.subheader("Patient Rating Distributions")

rating_chart = px.box(
    comparison_data,
    x="drug_name_normalized",
    y="rating",
    color="drug_name_normalized",
    points="all",
    color_discrete_sequence=DRUG_COLORS,
    category_orders={"drug_name_normalized": selected},
    hover_data=["review_id"],
    labels={
        "drug_name_normalized": "Drug",
        "rating": "Patient rating",
    },
)

rating_chart.update_layout(showlegend=False)
rating_chart.update_yaxes(range=[0.5, 10.5], dtick=1)

st.plotly_chart(
    style_chart(rating_chart),
    use_container_width=True,
)

effectiveness_tab, severity_tab = st.tabs(
    ["Reported effectiveness", "Reported side-effect severity"]
)

for tab, column, title, order, colors in [
    (
        effectiveness_tab,
        "effectiveness",
        "Reported Effectiveness",
        EFFECTIVENESS_ORDER,
        EFFECTIVENESS_COLORS,
    ),
    (
        severity_tab,
        "side_effects",
        "Reported Side-Effect Severity",
        SEVERITY_ORDER,
        SEVERITY_COLORS,
    ),
]:
    with tab:
        counts = (
            comparison_data.groupby(
                ["drug_name_normalized", column]
            )
            .size()
            .reset_index(name="reviews")
        )

        totals = counts.groupby("drug_name_normalized")[
            "reviews"
        ].transform("sum")

        counts["percentage"] = (
            100 * counts["reviews"] / totals
        )

        figure = px.bar(
            counts,
            x="drug_name_normalized",
            y="percentage",
            color=column,
            barmode="stack",
            color_discrete_map=colors,
            category_orders={
                column: order,
                "drug_name_normalized": selected,
            },
            hover_data={
                "reviews": True,
                "percentage": ":.1f",
            },
            labels={
                "drug_name_normalized": "Drug",
                "percentage": "Reviews (%)",
            },
            title=title,
        )

        figure.update_yaxes(range=[0, 100])

        st.plotly_chart(
            style_chart(figure),
            use_container_width=True,
        )

st.divider()
st.subheader("Written Comparison")

written_sections = []

for drug in selected:
    reviews = comparison_data.loc[
        comparison_data["drug_name_normalized"].eq(drug)
    ]

    overview = (
        f"{drug} has {len(reviews)} reviews for '{condition}', "
        f"with an average rating of "
        f"{reviews['rating'].mean():.2f}/10 "
        f"and a median rating of "
        f"{reviews['rating'].median():.1f}/10."
    )

    effectiveness_text = category_summary(
        reviews["effectiveness"]
    )

    severity_text = category_summary(
        reviews["side_effects"]
    )

    with st.expander(drug, expanded=True):
        st.write(overview)

        st.markdown("**Reported effectiveness**")
        st.write(effectiveness_text)

        st.markdown("**Reported side-effect severity**")
        st.write(severity_text)

    written_sections.append(
        "\n".join([
            overview,
            "",
            "Reported effectiveness:",
            effectiveness_text,
            "",
            "Reported side-effect severity:",
            severity_text,
        ])
    )

st.caption(
    "These descriptions summarize review categories. "
    "They do not establish clinical effectiveness or adverse-event rates."
)

left, right = st.columns(2)

with left:
    st.download_button(
        label="Download comparison as TSV",
        data=summary.to_csv(
            sep="\t",
            index=False,
        ).encode("utf-8-sig"),
        file_name="drug_comparison.tsv",
        mime="text/tab-separated-values",
    )

with right:
    st.download_button(
        label="Download written comparison",
        data="\n\n".join(written_sections).encode("utf-8"),
        file_name="drug_comparison_summary.txt",
        mime="text/plain",
    )