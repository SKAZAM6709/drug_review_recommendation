from pathlib import Path

import pandas as pd

from src.config import REQUIRED_COLUMNS, TEST_FILE, TRAIN_FILE


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Read a TSV dataset without modifying its contents."""
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}\n"
            "Place both original TSV files inside data/raw/."
        )

    # TSV files use tabs to separate columns.
    dataframe = pd.read_csv(file_path, sep="\t", encoding="utf-8")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing required columns: "
            f"{missing_columns}"
        )

    if dataframe.empty:
        raise ValueError(f"{file_path.name} contains no records.")

    return dataframe


def load_raw_data():
    """Return the original training and test datasets separately."""
    train_data = load_dataset(TRAIN_FILE)
    test_data = load_dataset(TEST_FILE)

    return train_data, test_data