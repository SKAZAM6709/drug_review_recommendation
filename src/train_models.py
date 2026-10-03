import json
import re

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import (
    GRAPHS_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    create_output_folders,
)
from src.text_features import TASKS, build_text_pipeline


def load_training_data():
    """Load only cleaned training data."""
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

    print(f"Training source: {path}")
    return dataframe


def get_classifiers():
    """Return fresh classifiers for one modeling task."""
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=10000,
        ),
    }


def normalized_text_key(text):
    """Group identical text regardless of case and spacing."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def prepare_task(dataframe, task_name, settings):
    text_column = settings["text_column"]
    target_column = settings["target_column"]
    labels = settings["labels"]

    missing_columns = [
        column
        for column in [text_column, target_column]
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{task_name}: missing columns {missing_columns}"
        )

    text = dataframe[text_column].astype(str).str.strip()
    targets = dataframe[target_column].astype(str).str.strip()

    unexpected = set(targets) - set(labels)

    if unexpected:
        raise ValueError(
            f"{task_name}: unexpected labels {sorted(unexpected)}"
        )

    valid = text.ne("")

    X = text.loc[valid].reset_index(drop=True)
    y = targets.loc[valid].reset_index(drop=True)

    if X.empty:
        raise ValueError(f"{task_name}: no nonempty training text.")

    # Identical text must remain within the same fold.
    groups = X.map(normalized_text_key)

    class_counts = y.value_counts().reindex(labels, fill_value=0)

    if (class_counts < 2).any():
        raise ValueError(
            f"{task_name}: every target category needs at least "
            f"two eligible records. Counts: {class_counts.to_dict()}"
        )

    class_group_counts = (
        pd.DataFrame({"label": y, "group": groups})
        .groupby("label")["group"]
        .nunique()
        .reindex(labels, fill_value=0)
    )

    n_splits = min(5, int(class_group_counts.min()))

    if n_splits < 2:
        raise ValueError(
            f"{task_name}: insufficient distinct text groups "
            "for cross-validation."
        )

    # Try fewer folds if grouping creates a fold without a category.
    while n_splits >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        splits = list(splitter.split(X, y, groups))

        expected = set(labels)
        complete = all(
            set(y.iloc[train_index]) == expected
            and set(y.iloc[validation_index]) == expected
            for train_index, validation_index in splits
        )

        if complete:
            break

        n_splits -= 1

    if n_splits < 2:
        raise ValueError(
            f"{task_name}: could not create grouped folds "
            "containing every category."
        )

    print(f"\nTASK: {task_name.upper()}")
    print(f"Eligible reviews: {len(X)}")
    print(f"Empty inputs excluded: {int((~valid).sum())}")
    print(f"Cross-validation folds: {n_splits}")
    print("\nClass counts:")
    print(class_counts.to_string())

    metadata = {
        "eligible_rows": int(len(X)),
        "empty_inputs_excluded": int((~valid).sum()),
        "folds": int(n_splits),
        "unique_text_groups": int(groups.nunique()),
        "class_counts": {
            label: int(count)
            for label, count in class_counts.items()
        },
    }

    return X, y, splits, metadata


def plot_comparison(results, task_name):
    """Plot cross-validation macro-F1 with fold variation."""
    ordered = results.sort_values("macro_f1_mean", ascending=False)

    plt.figure(figsize=(10, 5))

    plt.bar(
        ordered["model"],
        ordered["macro_f1_mean"],
        yerr=ordered["macro_f1_std"],
        capsize=4,
        color="#2675AD",
    )

    plt.ylim(0, 1)
    plt.ylabel("Cross-validation macro-F1")
    plt.xlabel("Model")
    plt.title(f"{task_name.title()}: Model Comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    path = GRAPHS_DIR / f"{task_name}_model_comparison.png"
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def train_task(dataframe, task_name, settings):
    X, y, splits, metadata = prepare_task(
        dataframe, task_name, settings
    )

    scorer = make_scorer(
        f1_score,
        average="macro",
        labels=settings["labels"],
        zero_division=0,
    )

    rows = []

    for name, classifier in get_classifiers().items():
        print(f"\nComparing: {name}")

        pipeline = build_text_pipeline(classifier)

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=splits,
            scoring={
                "macro_f1": scorer,
                "accuracy": "accuracy",
            },
            # One worker keeps memory use manageable on a laptop.
            n_jobs=1,
            error_score="raise",
        )

        row = {
            "task": task_name,
            "model": name,
            "macro_f1_mean": float(
                np.mean(scores["test_macro_f1"])
            ),
            "macro_f1_std": float(
                np.std(scores["test_macro_f1"])
            ),
            "accuracy_mean": float(
                np.mean(scores["test_accuracy"])
            ),
            "accuracy_std": float(
                np.std(scores["test_accuracy"])
            ),
            "mean_fit_seconds": float(
                np.mean(scores["fit_time"])
            ),
        }

        rows.append(row)

        print(
            f"Macro-F1: {row['macro_f1_mean']:.4f} "
            f"(SD {row['macro_f1_std']:.4f})"
        )
        print(f"Accuracy: {row['accuracy_mean']:.4f}")

    results = pd.DataFrame(rows).sort_values(
        "macro_f1_mean", ascending=False
    )

    results.to_csv(
        METRICS_DIR / f"{task_name}_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_comparison(results, task_name)

    # Save the strongest real classifier, while retaining Dummy results
    # so we can check whether learning beats the trivial baseline.
    candidates = results.loc[results["model"].ne("Dummy")]
    best_row = candidates.iloc[0]
    best_name = best_row["model"]

    final_pipeline = build_text_pipeline(
        get_classifiers()[best_name]
    )
    final_pipeline.fit(X, y)

    model_path = MODELS_DIR / f"{task_name}_pipeline.joblib"
    joblib.dump(final_pipeline, model_path)

    dummy_score = float(
        results.loc[
            results["model"].eq("Dummy"), "macro_f1_mean"
        ].iloc[0]
    )

    metadata.update({
        "selected_model": best_name,
        "selected_cv_macro_f1": float(best_row["macro_f1_mean"]),
        "selected_cv_accuracy": float(best_row["accuracy_mean"]),
        "dummy_cv_macro_f1": dummy_score,
        "beats_dummy_macro_f1": bool(
            best_row["macro_f1_mean"] > dummy_score
        ),
        "text_column": settings["text_column"],
        "target_column": settings["target_column"],
        "labels": settings["labels"],
        "model_file": model_path.name,
        "random_state": RANDOM_STATE,
    })

    print(f"\nSelected model: {best_name}")
    print(f"Saved: {model_path}")

    if not metadata["beats_dummy_macro_f1"]:
        print(
            "Finding: the selected classifier did not outperform "
            "the Dummy baseline on mean macro-F1."
        )

    return results, metadata


def write_findings(metadata):
    lines = [
        "NLP MODEL COMPARISON",
        "",
        "Results use training cross-validation only.",
        "They are model-selection estimates, not final test results.",
        "Models were compared with fixed initial settings.",
        "Error bars show fold standard deviation, not confidence intervals.",
        "",
    ]

    for task_name, details in metadata.items():
        lines.extend([
            f"TASK: {task_name.upper()}",
            f"Eligible reviews: {details['eligible_rows']}",
            f"Grouped folds: {details['folds']}",
            f"Selected model: {details['selected_model']}",
            f"Mean macro-F1: "
            f"{details['selected_cv_macro_f1']:.4f}",
            f"Mean accuracy: "
            f"{details['selected_cv_accuracy']:.4f}",
            f"Dummy macro-F1: "
            f"{details['dummy_cv_macro_f1']:.4f}",
            f"Outperforms Dummy on mean macro-F1: "
            f"{details['beats_dummy_macro_f1']}",
            "",
        ])

    lines.extend([
        "Sentiment labels are derived from ratings.",
        "Effectiveness and severity targets are patient-reported categories.",
        "Identical normalized task text was grouped within folds.",
        "Near-duplicate text may still require further inspection.",
        "Final generalization performance requires held-out test evaluation.",
    ])

    path = METRICS_DIR / "training_findings.txt"
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    create_output_folders()
    dataframe = load_training_data()

    all_results = []
    metadata = {}

    for task_name, settings in TASKS.items():
        results, details = train_task(
            dataframe, task_name, settings
        )

        all_results.append(results)
        metadata[task_name] = details

    pd.concat(all_results, ignore_index=True).to_csv(
        METRICS_DIR / "all_model_cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (METRICS_DIR / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    write_findings(metadata)

    print("\nTRAINING COMPLETE")
    print(f"Models: {MODELS_DIR}")
    print(f"Comparison graphs: {GRAPHS_DIR}")
    print(f"Written findings: {METRICS_DIR / 'training_findings.txt'}")
    print("Test data was not used.")


if __name__ == "__main__":
    main()