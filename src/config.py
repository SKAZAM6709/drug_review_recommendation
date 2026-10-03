from pathlib import Path

# Main project folder, determined from this file's location.
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folders.
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

# Output folders.
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
GRAPHS_DIR = REPORTS_DIR / "graphs"
METRICS_DIR = REPORTS_DIR / "metrics"

# Original dataset files.
TRAIN_FILE = RAW_DIR / "drugLibTrain_raw.tsv"
TEST_FILE = RAW_DIR / "drugLibTest_raw.tsv"

# Reproducibility.
RANDOM_STATE = 42

# Original dataset columns.
REQUIRED_COLUMNS = [
    "Unnamed: 0",
    "urlDrugName",
    "rating",
    "effectiveness",
    "sideEffects",
    "condition",
    "benefitsReview",
    "sideEffectsReview",
    "commentsReview",
]

TEXT_COLUMNS = [
    "benefitsReview",
    "sideEffectsReview",
    "commentsReview",
]


def create_output_folders():
    """Create folders where later scripts will save their outputs."""
    for folder in [
        PROCESSED_DIR,
        REFERENCE_DIR,
        MODELS_DIR,
        GRAPHS_DIR,
        METRICS_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
        # Initial ranking settings, not clinically validated thresholds.
MIN_REVIEWS = 3
RANKING_PRIOR_STRENGTH = 5