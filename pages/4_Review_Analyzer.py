import json

import joblib
import streamlit as st

from src.ui import apply_theme
from src.clean_data import clean_text
from src.config import MODELS_DIR
from src.text_features import TASKS


st.set_page_config(
    page_title="Review Analyzer",
    page_icon="💬",
    layout="wide",
)

apply_theme()

STATE_KEY = "review_analyzer_results"

TASK_TITLES = {
    "sentiment": "Review Sentiment",
    "effectiveness": "Reported Effectiveness",
    "severity": "Reported Side-Effect Severity",
}


@st.cache_resource
def load_model(path_string, modification_time):
    """Reload a cached model when its file changes."""
    return joblib.load(path_string)


def analyze_review(inputs):
    """Return a prediction or an explanation for each task."""
    outcomes = []

    for task_name in TASKS:
        text = inputs[task_name]
        path = MODELS_DIR / f"{task_name}_pipeline.joblib"

        outcome = {
            "task": task_name,
            "input_text": text,
            "predicted_label": None,
            "status": "skipped",
            "message": "",
            "short_input": len(text.split()) < 5,
        }

        if not text:
            outcome["message"] = (
                "No relevant description was provided for this task."
            )
            outcomes.append(outcome)
            continue

        if not path.exists():
            outcome["status"] = "missing_model"
            outcome["message"] = (
                f"Missing model: {path.name}. "
                "Run python -m src.train_models."
            )
            outcomes.append(outcome)
            continue

        try:
            pipeline = load_model(
                str(path),
                path.stat().st_mtime_ns,
            )

            features = pipeline.named_steps["tfidf"].transform(
                [text]
            )

            if features.nnz == 0:
                outcome["message"] = (
                    "No text features were recognized by this model. "
                    "No prediction is shown."
                )
            else:
                outcome["predicted_label"] = str(
                    pipeline.predict([text])[0]
                )
                outcome["status"] = "success"

        except Exception as error:
            outcome["status"] = "error"
            outcome["message"] = (
                f"Could not run this model: {error}"
            )

        outcomes.append(outcome)

    return outcomes


st.title("Patient Review Analyzer")

st.caption(
    "Analyze written patient experiences using the trained NLP models."
)

st.info(
    "Predictions classify review descriptions. They do not assess "
    "a medicine's suitability for an individual."
)

with st.expander("How to enter a review"):
    st.write(
        "Describe benefits in the benefits field and side effects "
        "in the side-effect field. Add any remaining context as comments. "
        "Keep phrases such as 'no nausea' or 'did not help' intact."
    )
    st.write(
        "The sentiment model uses all three fields. Effectiveness "
        "uses benefits and comments; severity uses side effects "
        "and comments."
    )

st.subheader("Enter Your Review")

with st.form("review_form"):
    left, right = st.columns(2)

    with left:
        benefits = st.text_area(
            "Benefits description",
            height=180,
            placeholder=(
                "Describe the reported benefits or lack of benefit."
            ),
        )

    with right:
        side_effects = st.text_area(
            "Side-effect description",
            height=180,
            placeholder=(
                "Describe reported side effects or their absence."
            ),
        )

    comments = st.text_area(
        "Additional comments",
        height=110,
        placeholder="Add any other details about the experience.",
    )

    submitted = st.form_submit_button(
        "Analyze Review",
        use_container_width=True,
    )

if submitted:
    benefits = clean_text(benefits)
    side_effects = clean_text(side_effects)
    comments = clean_text(comments)

    inputs = {
        "sentiment": " ".join(
            part
            for part in [benefits, side_effects, comments]
            if part
        ),
        "effectiveness": " ".join(
            part for part in [benefits, comments] if part
        ),
        "severity": " ".join(
            part for part in [side_effects, comments] if part
        ),
    }

    if not inputs["sentiment"]:
        st.session_state.pop(STATE_KEY, None)
        st.warning("Enter a review description first.")
    else:
        with st.spinner("Analyzing the review..."):
            st.session_state[STATE_KEY] = analyze_review(inputs)

outcomes = st.session_state.get(STATE_KEY)

if outcomes:
    st.divider()
    st.subheader("Prediction Results")

    st.caption(
        "Results correspond to the last submitted review. "
        "After editing the form, click Analyze Review again."
    )

    columns = st.columns(len(outcomes))

    for column, outcome in zip(columns, outcomes):
        with column:
            st.subheader(
                TASK_TITLES.get(
                    outcome["task"],
                    outcome["task"].title(),
                )
            )

            if outcome["status"] == "success":
                st.success(outcome["predicted_label"])

                if outcome["short_input"]:
                    st.caption(
                        "Very short descriptions provide limited context."
                    )

            elif outcome["status"] == "error":
                st.error(outcome["message"])

            elif outcome["status"] == "missing_model":
                st.warning(outcome["message"])

            else:
                st.info(outcome["message"])

    successful = [
        outcome
        for outcome in outcomes
        if outcome["status"] == "success"
    ]

    if successful:
        st.subheader("Written Overview")

        for outcome in successful:
            title = TASK_TITLES[outcome["task"]]
            st.write(
                f"**{title}:** the model predicts "
                f"**{outcome['predicted_label']}** "
                "from the submitted description."
            )

    with st.expander("Text used by each model"):
        for outcome in outcomes:
            st.markdown(
                f"**{TASK_TITLES[outcome['task']]}**"
            )
            st.text(
                outcome["input_text"]
                or "No relevant description provided."
            )

    st.caption(
        "Sentiment predicts rating-derived categories. "
        "Effectiveness and severity predict patient-reported labels. "
        "Confidence percentages are not displayed because the models "
        "have not been probability-calibrated."
    )

    if successful:
        export_results = [
            {
                "task": outcome["task"],
                "predicted_label": outcome["predicted_label"],
                "input_text": outcome["input_text"],
            }
            for outcome in successful
        ]

        st.download_button(
            label="Download predictions",
            data=json.dumps(
                export_results,
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8"),
            file_name="review_predictions.json",
            mime="application/json",
        )

    if st.button("Clear displayed results"):
        st.session_state.pop(STATE_KEY, None)
        st.rerun()