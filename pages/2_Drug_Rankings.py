import plotly.express as px
import streamlit as st

from src.ui import apply_theme
from src.config import MIN_REVIEWS, RANKING_PRIOR_STRENGTH
from src.recommend import load_ranking_data, rank_drugs
from src.summaries import build_drug_summary, format_summary


st.set_page_config(
    page_title="Drug Rankings",
    page_icon="📋",
    layout="wide",
)

apply_theme()


@st.cache_data
def get_data():
    return load_ranking_data()


st.title("Condition-Based Drug Rankings")

st.caption(
    "Rankings summarize patient feedback from training reviews. "
    "They do not determine which treatment is suitable for you."
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

conditions = (
    data["condition_normalized"]
    .value_counts()
    .index.tolist()
)

condition = st.selectbox(
    "Select a condition",
    conditions,
)

with st.expander("Ranking settings"):
    left, right = st.columns(2)

    with left:
        minimum = st.number_input(
            "Minimum reviews per drug",
            min_value=1,
            value=int(MIN_REVIEWS),
            step=1,
        )

    with right:
        strength = st.number_input(
            "Smoothing strength",
            min_value=0.1,
            value=float(RANKING_PRIOR_STRENGTH),
            step=0.5,
        )

    st.caption(
        "Increasing smoothing moves scores closer to the condition "
        "average. These are exploratory settings."
    )

try:
    rankings, reviews, coverage = rank_drugs(
        data,
        condition,
        min_reviews=int(minimum),
        prior_strength=float(strength),
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

a, b, c = st.columns(3)

a.metric(
    "Condition reviews",
    f"{coverage['condition_review_count']:,}",
)

b.metric(
    "Eligible drugs",
    coverage["eligible_drugs"],
)

c.metric(
    "Drugs below minimum",
    coverage["excluded_drugs"],
)

st.subheader("How the Ranking Works")

st.write(
    "The adjusted score combines each drug's average rating with "
    "the condition average. Smaller review groups are adjusted "
    "more strongly toward that average."
)

st.latex(
    r"S_d=\frac{n_d\bar{r}_d+m\mu_c}{n_d+m}"
)

with st.expander("Formula definitions"):
    st.markdown(
        """
- **n₍d₎:** review count for the drug within the selected condition.
- **r̄₍d₎:** average rating for that drug-condition pair.
- **μ₍c₎:** average rating across reviews for the condition.
- **m:** smoothing strength.
- **S₍d₎:** adjusted patient-rating score.
"""
    )

    st.write(
        f"Condition average rating: "
        f"**{coverage['condition_average_rating']:.2f}/10**."
    )

if rankings.empty:
    st.warning(
        "Insufficient reviews for ranking at this minimum. "
        "Choose another condition or adjust the minimum review count."
    )
    st.stop()

st.divider()
st.subheader("Ranked Drugs")

columns = [
    "rank",
    "drug_name",
    "review_count",
    "average_rating",
    "adjusted_score",
]

st.dataframe(
    rankings[columns].round(3),
    hide_index=True,
    use_container_width=True,
)

figure = px.bar(
    rankings.head(10).sort_values(
        "adjusted_score",
        kind="stable",
    ),
    x="adjusted_score",
    y="drug_name",
    orientation="h",
    color_discrete_sequence=["#176B75"],
    hover_data={
        "rank": True,
        "review_count": True,
        "average_rating": ":.2f",
        "adjusted_score": ":.2f",
    },
    labels={
        "adjusted_score": "Adjusted patient rating",
        "drug_name": "Drug",
        "review_count": "Review count",
        "average_rating": "Average rating",
        "rank": "Rank",
    },
    title="Top Eligible Drugs",
)

figure.update_layout(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    font=dict(color="#172B4D"),
    title=dict(font=dict(color="#102A43")),
    height=max(350, 45 * min(len(rankings), 10) + 100),
    margin=dict(l=20, r=20, t=60, b=40),
    showlegend=False,
)

figure.update_xaxes(
    range=[0, 10],
    gridcolor="#E8EEF5",
)

figure.update_yaxes(
    automargin=True,
    categoryorder="array",
    categoryarray=rankings.head(10)["drug_name"].tolist()[::-1],
)

st.plotly_chart(
    figure,
    use_container_width=True,
)

st.download_button(
    label="Download rankings as TSV",
    data=rankings.to_csv(
        sep="\t",
        index=False,
    ).encode("utf-8-sig"),
    file_name="drug_rankings.tsv",
    mime="text/tab-separated-values",
)

st.divider()
st.subheader("Written Summary and Supporting Reviews")

display_names = dict(
    zip(
        rankings["drug_name_normalized"],
        rankings["drug_name"],
    )
)

selected = st.selectbox(
    "Choose a ranked drug",
    rankings["drug_name_normalized"].tolist(),
    format_func=lambda name: display_names[name],
)

row = rankings.loc[
    rankings["drug_name_normalized"].eq(selected)
].iloc[0]

drug_reviews = reviews.loc[
    reviews["drug_name_normalized"].eq(selected)
]

summary = build_drug_summary(drug_reviews, row)

st.write(summary["description"])

left, right = st.columns(2)

with left:
    st.subheader("Reported Effectiveness")
    st.write(summary["effectiveness_summary"])

with right:
    st.subheader("Reported Side-Effect Severity")
    st.write(summary["side_effect_severity_summary"])

with st.expander("Why this rank?"):
    st.write(summary["ranking_explanation"])

benefits_tab, side_effects_tab = st.tabs(
    ["Benefits excerpts", "Side-effect excerpts"]
)

for tab, key in [
    (benefits_tab, "benefits_excerpts"),
    (side_effects_tab, "side_effects_excerpts"),
]:
    with tab:
        if not summary[key]:
            st.info("No written descriptions are available.")

        for excerpt in summary[key]:
            st.caption(
                f"Review {excerpt['review_id']} | "
                f"Rating {excerpt['rating']}/10"
            )

            # Show source review excerpts as plain text.
            st.text(excerpt["text"])
            st.divider()

st.caption(
    "Excerpts are examples selected around the median rating. "
    "They do not establish how common a reported experience is."
)

st.download_button(
    label="Download written summary",
    data=format_summary(summary).encode("utf-8"),
    file_name="drug_summary.txt",
    mime="text/plain",
)