# Drug Review Insights

A Data Science final year project that analyzes patient reviews, classifies review text using machine learning and NLP, and ranks drugs within a selected condition.

Built with Python, scikit-learn, Plotly, and Streamlit.

> This application summarizes patient feedback for research purposes. It does not provide prescriptions or personalized treatment recommendations.

## Features

- **Data Explorer:** Filter reviews by condition, drug, rating, sentiment label, and text.
- **Drug Rankings:** View condition-specific rankings using adjusted patient ratings.
- **Drug Comparison:** Compare rating distributions, reported effectiveness, and side-effect severity.
- **Review Analyzer:** Predict sentiment, effectiveness, and side-effect severity from written descriptions.
- **Model Evaluation:** Inspect cross-validation results, held-out test metrics, confusion matrices, and prediction errors.
- **Written Summaries:** Read statistical descriptions and supporting review excerpts.
- **Downloads:** Export filtered reviews, rankings, summaries, and prediction results.

## Dataset

The project uses two supplied patient-review files:

| File | Original reviews |
|---|---:|
| `drugLibTrain_raw.tsv` | 3,107 |
| `drugLibTest_raw.tsv` | 1,036 |

Fields include drug name, condition, rating, reported effectiveness, reported side-effect severity, benefits descriptions, side-effect descriptions, and additional comments.

The original files remain unchanged. Cleaning normalizes text and names, handles missing descriptions, and removes duplicate training content, including matches with test reviews.

The dashboard explores cleaned training reviews with nonblank drug and condition names. Test data is reserved for classification evaluation.

## Machine Learning and NLP

Three classification tasks are implemented:

| Task | Input | Target |
|---|---|---|
| Sentiment | Combined review text | Negative, neutral, positive |
| Effectiveness | Benefits description and comments | Five patient-reported effectiveness categories |
| Side-effect severity | Side-effect description and comments | Five patient-reported severity categories |

Sentiment labels are derived from ratings:

- **Negative:** 1–4
- **Neutral:** 5–6
- **Positive:** 7–10

These are proxy labels, not independently annotated sentiment.

TF-IDF features use word unigrams and bigrams. Models compared include:

- Dummy Classifier
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

Models are selected separately for each task using training cross-validation and macro-F1. Identical normalized task text is grouped within cross-validation folds.

Final evaluation reports accuracy, macro-F1, weighted-F1, per-category metrics, and confusion matrices. Additional diagnostics identify exact training-text matches among test inputs.

## Drug Ranking Method

Drugs are ranked within a condition using a smoothed average rating:

```text
Adjusted score =
(review count × drug average rating
 + smoothing strength × condition average rating)
÷ (review count + smoothing strength)
```

Small review groups are adjusted more strongly toward the condition average.

Initial defaults:

- Minimum reviews per drug-condition pair: **3**
- Smoothing strength: **5**

These are exploratory settings, not clinically validated thresholds. Review counts and supporting excerpts are displayed alongside results.

## Project Structure

```text
drug_review_recommendation/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── inspect_data.py
│   ├── clean_data.py
│   ├── eda.py
│   ├── text_features.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── recommend.py
│   ├── summaries.py
│   ├── clinical_data.py
│   └── ui.py
├── pages/
│   ├── 1_Data_Explorer.py
│   ├── 2_Drug_Rankings.py
│   ├── 3_Drug_Comparison.py
│   ├── 4_Review_Analyzer.py
│   └── 5_Model_Evaluation.py
├── models/
└── reports/
    ├── graphs/
    └── metrics/
```

## Run Locally

The commands below use Windows PowerShell.

### 1. Clone the repository

```powershell
git clone https://github.com/raisarslan56/drug-review-recommendation.git
cd drug-review-recommendation
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the Python version and dependency versions used to train the saved models.

### 4. Start the dashboard

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed in the terminal, usually:

```text
http://localhost:8501
```

The dashboard requires cleaned training data, saved models, and generated evaluation reports. If these are already included, regeneration is unnecessary.

## Reproduce the Pipeline

Place the original TSV files in `data/raw/`, then run these commands in order:

```powershell
.\.venv\Scripts\python.exe -m src.inspect_data
.\.venv\Scripts\python.exe -m src.clean_data
.\.venv\Scripts\python.exe -m src.eda
.\.venv\Scripts\python.exe -m src.train_models
.\.venv\Scripts\python.exe -m src.evaluate_models
```

Generate a condition-specific ranking:

```powershell
.\.venv\Scripts\python.exe -m src.recommend --condition "depression" --top 5
```

List available condition strings:

```powershell
.\.venv\Scripts\python.exe -m src.recommend --list-conditions
```

## Outputs

- Cleaned training and test datasets
- Exploratory charts and written findings
- Saved TF-IDF classification pipelines
- Model comparison tables
- Held-out test metrics and confusion matrices
- Prediction error reports
- Condition-specific ranking tables
- Written summaries and supporting review excerpts

Actual performance results are stored in `reports/metrics/`.

## Limitations

- Reviews are self-reported and may contain selection bias.
- Ratings and labels do not establish clinical effectiveness or safety.
- Missing side-effect descriptions do not mean that no side effects occurred.
- Condition normalization currently handles case and spacing; medical synonyms may remain separate.
- Sparse drug-condition groups provide limited ranking evidence.
- Exact text checks do not detect all near-duplicate reviews.
- Classification metrics do not measure recommendation quality.
- Review excerpts are examples, not estimates of benefit or side-effect frequency.
- Prediction probabilities have not been calibrated.

## Future Work

- Clinical reference enrichment using RxNorm/openFDA
- Reviewed condition-name mappings
- Ranking stability analysis
- Improved side-effect extraction and negation handling
- Independently annotated sentiment evaluation
- Probability calibration

## Academic Context

**Project:** Data-Driven Drug Recommendation and Patient Review Analysis Using Machine Learning and NLP

**Domain:** Data Science, Natural Language Processing, and Machine Learning