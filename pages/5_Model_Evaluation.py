import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import apply_theme
from src.config import GRAPHS_DIR, METRICS_DIR


st.set_page_config(
    page_title="Model Evaluation",
    page_icon="🧪",
    layout="wide",
)

apply_theme()


def style_chart(figure):
    """Apply the application's chart colors and layout."""
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


def read_optional_csv(path, **kwargs):
    """Show a useful message if an optional report cannot be read."""
    if not path.exists():
        st.info(f"Report not available: {path.name}")
        return None

    try:
        return pd.read_csv(path, **kwargs)
    except (ValueError, OSError) as error:
        st.warning(f"Could not read {path.name}: {error}")
        return None


st.title("Model Evaluation")

st.caption(
    "Inspect held-out test performance, training model comparisons, "
    "and prediction errors."
)

summary_path = METRICS_DIR / "test_results_summary.csv"
details_path = METRICS_DIR / "test_metrics.json"

if not summary_path.exists() or not details_path.exists():
    st.warning("Test evaluation results have not been generated.")
    st.code("python -m src.evaluate_models")
    st.stop()

try:
    summary = pd.read_csv(summary_path)

    with details_path.open("r", encoding="utf-8") as file:
        details = json.load(file)

except (ValueError, OSError) as error:
    st.error(f"Could not load evaluation results: {error}")
    st.stop()

required_columns = [
    "task",
    "model",
    "accuracy",
    "macro_f1",
    "weighted_f1",
]

missing = [
    column for column in required_columns
    if column not in summary.columns
]

if missing:
    st.error(f"Evaluation summary is missing columns: {missing}")
    st.code("python -m src.evaluate_models")
    st.stop()

if summary.empty:
    st.warning("The evaluation summary contains no results.")
    st.stop()

if not isinstance(details, dict):
    st.error("The evaluation details file has an unexpected format.")
    st.stop()

st.subheader("Held-Out Test Results")

st.dataframe(
    summary.round(4),
    hide_index=True,
    use_container_width=True,
)

chart_data = summary.melt(
    id_vars=["task", "model"],
    value_vars=["accuracy", "macro_f1", "weighted_f1"],
    var_name="metric",
    value_name="score",
)

figure = px.bar(
    chart_data,
    x="task",
    y="score",
    color="metric",
    barmode="group",
    color_discrete_map={
        "accuracy": "#176B75",
        "macro_f1": "#2675AD",
        "weighted_f1": "#E79B36",
    },
    hover_data=["model"],
    labels={
        "task": "Classification task",
        "score": "Test score",
        "metric": "Metric",
    },
    title="Classification Performance by Task",
)

figure.update_yaxes(range=[0, 1])

st.plotly_chart(
    style_chart(figure),
    use_container_width=True,
)

with st.expander("What do the metrics mean?"):
    st.markdown(
        """
- **Accuracy:** fraction of reviews classified correctly.
- **Macro-F1:** average F1 across categories, giving each equal weight.
- **Weighted-F1:** average F1 weighted by category review counts.
- **Precision:** how often predictions of a category are correct.
- **Recall:** how many actual examples of a category are identified.
- **Support:** number of test reviews in a category.
"""
    )

    st.write(
        "These metrics evaluate review classification. "
        "They do not measure drug-ranking quality or clinical effectiveness."
    )

st.divider()

task = st.selectbox(
    "Inspect a task",
    summary["task"].tolist(),
    format_func=lambda value: value.title(),
)

task_details = details.get(task)

if not isinstance(task_details, dict):
    st.error(f"Evaluation details are missing for '{task}'.")
    st.stop()

selected_row = summary.loc[summary["task"].eq(task)].iloc[0]

st.subheader(f"{task.title()} Results")
st.write(f"**Selected model:** {selected_row['model']}")

a, b, c = st.columns(3)

a.metric("Accuracy", f"{selected_row['accuracy']:.4f}")
b.metric("Macro-F1", f"{selected_row['macro_f1']:.4f}")
c.metric("Weighted-F1", f"{selected_row['weighted_f1']:.4f}")

performance_tab, comparison_tab, errors_tab = st.tabs([
    "Test performance",
    "Training comparison",
    "Prediction errors",
])

with performance_tab:
    st.subheader("Text Overlap Diagnostics")

    left, right = st.columns(2)

    left.metric(
        "Test inputs matching training text",
        task_details.get(
            "test_rows_with_training_text_match",
            "Unavailable",
        ),
    )

    right.metric(
        "Empty test inputs excluded",
        task_details.get(
            "empty_test_inputs_excluded",
            "Unavailable",
        ),
    )

    unseen = task_details.get("unseen_training_text_subset")

    if unseen is not None:
        st.write(
            f"On **{unseen['reviews']}** test reviews without exact "
            f"training-text matches, macro-F1 is "
            f"**{unseen['macro_f1']:.4f}**."
        )

        st.caption(
            "This subset has a different sample composition. "
            "It is an overlap diagnostic, not a new independent test set."
        )

    image_path = (
        GRAPHS_DIR / f"{task}_test_confusion_matrix.png"
    )

    st.subheader("Confusion Matrix")

    if image_path.exists():
        st.image(str(image_path))

        st.caption(
            "Rows represent actual categories; columns represent "
            "predictions. Diagonal cells are correct predictions. "
            "This image retains the colors used by the evaluation script."
        )
    else:
        st.info(
            "Confusion matrix image is missing. "
            "Run python -m src.evaluate_models."
        )

    st.subheader("Category-Level Metrics")

    report_path = (
        METRICS_DIR / f"{task}_test_classification_report.csv"
    )

    report = read_optional_csv(report_path, index_col=0)

    if report is not None:
        report.index.name = "category"

        st.dataframe(
            report.reset_index().round(4),
            hide_index=True,
            use_container_width=True,
        )

with comparison_tab:
    st.subheader("Training Cross-Validation Comparison")

    cv_path = METRICS_DIR / f"{task}_cv_results.csv"
    cv_results = read_optional_csv(cv_path)

    if cv_results is not None:
        st.dataframe(
            cv_results.round(4),
            hide_index=True,
            use_container_width=True,
        )

        if {
            "model",
            "macro_f1_mean",
            "macro_f1_std",
        }.issubset(cv_results.columns):
            cv_chart = px.bar(
                cv_results.sort_values(
                    "macro_f1_mean",
                    ascending=False,
                ),
                x="model",
                y="macro_f1_mean",
                error_y="macro_f1_std",
                color_discrete_sequence=["#176B75"],
                labels={
                    "model": "Model",
                    "macro_f1_mean": "Mean cross-validation macro-F1",
                },
                title=f"{task.title()}: Training Model Comparison",
            )

            cv_chart.update_yaxes(range=[0, 1])

            st.plotly_chart(
                style_chart(cv_chart),
                use_container_width=True,
            )

    st.caption(
        "Cross-validation selected the model. Test results estimate "
        "held-out performance. Error bars show fold standard deviation, "
        "not confidence intervals."
    )

with errors_tab:
    st.subheader("Incorrect Predictions")

    errors_path = METRICS_DIR / f"{task}_test_errors.csv"

    errors = read_optional_csv(
        errors_path,
        keep_default_na=False,
    )

    if errors is not None:
        if errors.empty:
            st.write("No incorrect predictions were recorded.")
        else:
            text_column = task_details.get(
                "text_column",
                {
                    "sentiment": "combined_review",
                    "effectiveness": "effectiveness_text",
                    "severity": "severity_text",
                }.get(task, "combined_review"),
            )

            columns = [
                column
                for column in [
                    "review_id",
                    "actual_label",
                    "predicted_label",
                    text_column,
                    "text_seen_in_training",
                ]
                if column in errors.columns
            ]

            st.write(f"**Incorrect predictions:** {len(errors):,}")

            st.dataframe(
                errors[columns].head(50),
                hide_index=True,
                use_container_width=True,
            )

            st.caption("The table displays the first 50 errors.")

            st.download_button(
                label="Download all incorrect predictions",
                data=errors.to_csv(
                    sep="\t",
                    index=False,
                ).encode("utf-8-sig"),
                file_name=f"{task}_errors.tsv",
                mime="text/tab-separated-values",
            )

st.divider()

findings_path = METRICS_DIR / "test_findings.txt"

if findings_path.exists():
    with st.expander("Written Evaluation Findings"):
        st.text(findings_path.read_text(encoding="utf-8"))

st.download_button(
    label="Download test results",
    data=summary.to_csv(
        sep="\t",
        index=False,
    ).encode("utf-8-sig"),
    file_name="test_results.tsv",
    mime="text/tab-separated-values",
)