import json
import re

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

from src.config import (
    GRAPHS_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    create_output_folders,
)
from src.text_features import TASKS


def load_clean_split(split_name):
    """Load a cleaned TSV or CSV file."""
    tsv_path = PROCESSED_DIR / f"{split_name}_clean.tsv"
    csv_path = PROCESSED_DIR / f"{split_name}_clean.csv"

    if tsv_path.exists():
        path = tsv_path
        separator = "\t"
    elif csv_path.exists():
        path = csv_path
        separator = ","
    else:
        raise FileNotFoundError(
            f"Cleaned {split_name} data not found. "
            "Run python -m src.clean_data first."
        )

    dataframe = pd.read_csv(
        path,
        sep=separator,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if dataframe.empty:
        raise ValueError(f"{split_name} data is empty.")

    return dataframe


def normalized_text_key(text):
    """Use the same text normalization as training."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def calculate_metrics(y_true, y_pred, labels):
    return {
        "reviews": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def save_classification_report(y_true, y_pred, labels, path):
    """Save category-level metrics and aggregate results."""
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        path,
        encoding="utf-8-sig",
    )


def save_confusion_chart(y_true, y_pred, labels, task_name):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    fig, ax = plt.subplots(figsize=(11, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
        xticks_rotation=35,
    )

    ax.set_title(
        f"{task_name.title()}: Held-Out Test Confusion Matrix"
    )
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")

    fig.tight_layout()

    path = GRAPHS_DIR / f"{task_name}_test_confusion_matrix.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_task(train, test, task_name, settings, metadata):
    text_column = settings["text_column"]
    target_column = settings["target_column"]
    labels = settings["labels"]

    for split_name, dataframe in [("train", train), ("test", test)]:
        missing = [
            column
            for column in [text_column, target_column]
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"{task_name}: missing {split_name} columns: {missing}"
            )

    model_path = MODELS_DIR / f"{task_name}_pipeline.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}\n"
            "Run python -m src.train_models first."
        )

    # Load only trusted models created by your project.
    pipeline = joblib.load(model_path)

    task_data = test.copy()
    task_data[text_column] = (
        task_data[text_column].astype(str).str.strip()
    )

    nonempty = task_data[text_column].ne("")
    excluded_empty = int((~nonempty).sum())

    task_data = task_data.loc[nonempty].copy()

    if task_data.empty:
        raise ValueError(f"{task_name}: no eligible test reviews.")

    y_true = task_data[target_column].astype(str)

    unexpected = set(y_true) - set(labels)

    if unexpected:
        raise ValueError(
            f"{task_name}: unexpected test labels: "
            f"{sorted(unexpected)}"
        )

    predictions = pipeline.predict(task_data[text_column])

    task_data["actual_label"] = y_true
    task_data["predicted_label"] = predictions
    task_data["correct_prediction"] = (
        task_data["actual_label"] == task_data["predicted_label"]
    )

    # Detect task-specific text overlap, which can remain even
    # after removing duplicate complete reviews.
    training_keys = set(
        train[text_column]
        .astype(str)
        .map(normalized_text_key)
    )
    training_keys.discard("")

    task_data["text_seen_in_training"] = (
        task_data[text_column]
        .map(normalized_text_key)
        .isin(training_keys)
    )

    # Mark repeated task text within the test set.
    test_keys = task_data[text_column].map(normalized_text_key)

    task_data["repeated_test_text"] = test_keys.duplicated(
        keep=False
    )

    all_metrics = calculate_metrics(
        task_data["actual_label"],
        task_data["predicted_label"],
        labels,
    )

    save_classification_report(
        task_data["actual_label"],
        task_data["predicted_label"],
        labels,
        METRICS_DIR / f"{task_name}_test_classification_report.csv",
    )

    save_confusion_chart(
        task_data["actual_label"],
        task_data["predicted_label"],
        labels,
        task_name,
    )

    task_data.to_csv(
        METRICS_DIR / f"{task_name}_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    task_data.loc[~task_data["correct_prediction"]].to_csv(
        METRICS_DIR / f"{task_name}_test_errors.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Additional diagnostic: test records without exact text overlap.
    unseen = task_data.loc[
        ~task_data["text_seen_in_training"]
    ]

    unseen_metrics = None

    if not unseen.empty:
        unseen_metrics = calculate_metrics(
            unseen["actual_label"],
            unseen["predicted_label"],
            labels,
        )

        unseen_metrics["class_support"] = {
            label: int(unseen["actual_label"].eq(label).sum())
            for label in labels
        }

        save_classification_report(
            unseen["actual_label"],
            unseen["predicted_label"],
            labels,
            METRICS_DIR / (
                f"{task_name}_unseen_text_classification_report.csv"
            ),
        )

    details = {
        "task": task_name,
        "selected_model": metadata[task_name]["selected_model"],
        "empty_test_inputs_excluded": excluded_empty,
        "test_rows_with_training_text_match": int(
            task_data["text_seen_in_training"].sum()
        ),
        "test_rows_with_repeated_test_text": int(
            task_data["repeated_test_text"].sum()
        ),
        "class_support": {
            label: int(task_data["actual_label"].eq(label).sum())
            for label in labels
        },
        "all_eligible_test": all_metrics,
        "unseen_training_text_subset": unseen_metrics,
    }

    print(f"\nTASK: {task_name.upper()}")
    print(f"Model: {details['selected_model']}")
    print(f"Eligible test reviews: {all_metrics['reviews']}")
    print(f"Empty inputs excluded: {excluded_empty}")
    print(f"Accuracy: {all_metrics['accuracy']:.4f}")
    print(f"Macro-F1: {all_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {all_metrics['weighted_f1']:.4f}")
    print(
        "Test inputs matching training text:",
        details["test_rows_with_training_text_match"],
    )

    if unseen_metrics is not None:
        print(
            "Macro-F1 without exact training-text matches:",
            f"{unseen_metrics['macro_f1']:.4f}",
        )

    print("\nCategory-level results:")
    print(
        classification_report(
            task_data["actual_label"],
            task_data["predicted_label"],
            labels=labels,
            digits=3,
            zero_division=0,
        )
    )

    return details


def save_overview_chart(summary):
    """Compare final test metrics across the three tasks."""
    table = pd.DataFrame([
        {
            "task": task,
            "Accuracy": details["all_eligible_test"]["accuracy"],
            "Macro-F1": details["all_eligible_test"]["macro_f1"],
        }
        for task, details in summary.items()
    ]).set_index("task")

    ax = table.plot(
        kind="bar",
        figsize=(10, 5),
        color=["#2675AD", "#E79B36"],
        rot=0,
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel("Task")
    ax.set_ylabel("Test score")
    ax.set_title("Held-Out Test Performance")

    plt.tight_layout()
    plt.savefig(
        GRAPHS_DIR / "test_performance_overview.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def write_findings(summary):
    lines = [
        "HELD-OUT TEST EVALUATION",
        "",
        "Models were selected using training cross-validation.",
        "No model fitting or selection was performed in this script.",
        "",
    ]

    for task_name, details in summary.items():
        metrics = details["all_eligible_test"]

        lines.extend([
            f"TASK: {task_name.upper()}",
            f"Selected model: {details['selected_model']}",
            f"Eligible test reviews: {metrics['reviews']}",
            f"Accuracy: {metrics['accuracy']:.4f}",
            f"Macro-F1: {metrics['macro_f1']:.4f}",
            f"Weighted-F1: {metrics['weighted_f1']:.4f}",
            f"Empty inputs excluded: "
            f"{details['empty_test_inputs_excluded']}",
            f"Test rows matching training text: "
            f"{details['test_rows_with_training_text_match']}",
            f"Rows containing repeated test text: "
            f"{details['test_rows_with_repeated_test_text']}",
        ])

        unseen = details["unseen_training_text_subset"]

        if unseen is not None:
            lines.extend([
                f"Test rows without training-text matches: "
                f"{unseen['reviews']}",
                f"Macro-F1 on that subset: "
                f"{unseen['macro_f1']:.4f}",
            ])

        lines.append("")

    lines.extend([
        "INTERPRETATION",
        "Accuracy is the fraction of correctly classified reviews.",
        "Macro-F1 gives equal importance to every configured category.",
        "Weighted-F1 weights category scores by their test support.",
        "Missing-category support should be checked in the reports.",
        "The unseen-text subset is a diagnostic with a different "
        "sample composition, not a new independently collected test set.",
        "Exact text checks do not detect all near-duplicate reviews.",
        "Repeated test text can reduce the independence of observations.",
        "Sentiment targets are rating-derived proxy labels.",
        "Effectiveness and severity predictions describe review labels, "
        "not verified clinical outcomes.",
        "Classification scores do not measure drug-ranking quality.",
        "Do not repeatedly tune models against these test results.",
    ])

    path = METRICS_DIR / "test_findings.txt"
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    create_output_folders()

    metadata_path = METRICS_DIR / "training_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Training metadata not found. "
            "Run python -m src.train_models first."
        )

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    train = load_clean_split("train")
    test = load_clean_split("test")

    summary = {}

    for task_name, settings in TASKS.items():
        summary[task_name] = evaluate_task(
            train,
            test,
            task_name,
            settings,
            metadata,
        )

    with (METRICS_DIR / "test_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    overview = pd.DataFrame([
        {
            "task": task,
            "model": details["selected_model"],
            **details["all_eligible_test"],
            "training_text_matches": (
                details["test_rows_with_training_text_match"]
            ),
        }
        for task, details in summary.items()
    ])

    overview.to_csv(
        METRICS_DIR / "test_results_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_overview_chart(summary)
    write_findings(summary)

    print("\nTEST EVALUATION COMPLETE")
    print(f"Charts: {GRAPHS_DIR}")
    print(f"Results: {METRICS_DIR}")
    print("Saved models were not modified.")


if __name__ == "__main__":
    main()