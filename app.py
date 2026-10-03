import pandas as pd
import streamlit as st

from src.ui import apply_theme
from src.config import MODELS_DIR
from src.recommend import load_ranking_data


st.set_page_config(
    page_title="Drug Review Insights",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply the shared theme.
apply_theme()


@st.cache_data
def get_training_data():
    return load_ranking_data()


def feature_card(title, description):
    st.markdown(
        f"""
        <div class="feature-card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("## 💊 Drug Review Insights")
    st.caption("Patient feedback • NLP • Machine learning")
    st.divider()

    st.write(
        "Explore reviews, compare reported experiences, "
        "and inspect model performance."
    )

    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


st.markdown(
    """
    <div class="hero">
        <div class="hero-label">PATIENT REVIEW ANALYTICS</div>
        <h1>Understand the experiences behind the ratings.</h1>
        <p>
            Explore patient reviews, compare drugs within a condition,
            and examine predictions from NLP and machine learning models.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice">
        Rankings describe patient feedback for research purposes.
        They do not provide personalized treatment recommendations.
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    data = get_training_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.code("python -m src.clean_data")
    st.stop()

if data.empty:
    st.warning(
        "No reviews with valid drug and condition names are available."
    )
    st.stop()


col1, col2, col3, col4 = st.columns(4)

col1.metric("Training reviews", f"{len(data):,}")

col2.metric(
    "Drug names",
    f"{data['drug_name_normalized'].nunique():,}",
)

col3.metric(
    "Condition strings",
    f"{data['condition_normalized'].nunique():,}",
)

col4.metric(
    "Average rating",
    f"{data['rating'].mean():.2f}/10",
)

st.caption(
    "Counts include cleaned training reviews with nonblank drug and "
    "condition names. Test data is reserved for evaluation."
)


st.markdown("## Explore the project")

left, right = st.columns(2)

with left:
    feature_card(
        "Explore patient reviews",
        "Filter by condition, drug, and rating. View distributions "
        "and read benefits, side effects, and additional comments.",
    )

    st.page_link(
        "pages/1_Data_Explorer.py",
        label="Open Data Explorer",
        icon="📊",
    )

    st.write("")

    feature_card(
        "Compare reported experiences",
        "Compare two or three drugs for the same condition using "
        "rating distributions and patient-reported categories.",
    )

    st.page_link(
        "pages/3_Drug_Comparison.py",
        label="Open Drug Comparison",
        icon="⚖️",
    )

with right:
    feature_card(
        "Inspect condition-based rankings",
        "View adjusted patient ratings, review counts, written summaries, "
        "and the evidence supporting each ranking.",
    )

    st.page_link(
        "pages/2_Drug_Rankings.py",
        label="Open Drug Rankings",
        icon="📋",
    )

    st.write("")

    feature_card(
        "Analyze a new review",
        "Enter a benefits description, side-effect description, and "
        "comments to predict review categories.",
    )

    st.page_link(
        "pages/4_Review_Analyzer.py",
        label="Open Review Analyzer",
        icon="💬",
    )


st.divider()

left, right = st.columns([3, 2])

with left:
    st.subheader("Check the evidence")

    st.write(
        "Inspect model comparison results, held-out test metrics, "
        "confusion matrices, and incorrect predictions."
    )

    st.page_link(
        "pages/5_Model_Evaluation.py",
        label="Open Model Evaluation",
        icon="🧪",
    )

    with st.expander("How to interpret the results"):
        st.write(
            "Effectiveness and severity labels are patient-reported. "
            "Sentiment categories are derived from ratings. "
            "Condition names are normalized for case and spacing, "
            "so medical synonyms may still appear separately."
        )

        st.write(
            "Classification scores measure predictions of review labels. "
            "They do not establish clinical effectiveness or ranking quality."
        )

with right:
    st.subheader("Model availability")

    statuses = pd.DataFrame([
        {
            "Task": task.title(),
            "Status": (
                "Available"
                if (MODELS_DIR / f"{task}_pipeline.joblib").exists()
                else "Not trained"
            ),
        }
        for task in ["sentiment", "effectiveness", "severity"]
    ])

    st.dataframe(
        statuses,
        hide_index=True,
        use_container_width=True,
    )


st.markdown(
    """
    <div class="footer">
        Data-Driven Drug Recommendation and Patient Review Analysis
        Using Machine Learning and NLP<br>
        Final Year Project
    </div>
    """,
    unsafe_allow_html=True,
)