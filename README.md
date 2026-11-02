# RetailMind AI

## Product vision

RetailMind AI is an end-to-end retail intelligence platform that uses machine
learning and AI to help businesses understand customers, predict churn, and
improve customer retention.

## Day 1 — Customer churn prediction

### Objective

Build an end-to-end machine-learning pipeline to predict whether a telecom
customer is likely to leave the company.

### Dataset

Use the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Download the CSV and place it at `data/telco_customer_churn.csv`. The dataset is
not included in this repository.

### Technologies

- Python
- Pandas
- Scikit-learn
- Joblib

### Setup and execution

Run these commands from the repository root:

```shell
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/predict.py
```

`train.py` loads and inspects the data, cleans the target and charge columns,
creates a stratified 80/20 split, trains the preprocessing/model pipeline, and
saves it to `models/churn_pipeline.joblib`. `evaluate.py` trains the same fixed
baseline and prints its held-out metrics. `predict.py` loads the saved pipeline
and runs a sample prediction; run training first.

The model artifact and downloaded CSV are excluded by `.gitignore`. Keep or
publish them according to your dataset and artifact-storage policy.

### ML workflow

1. Data ingestion and inspection
2. Identifier removal and target/charge cleaning
3. Numeric imputation and scaling; categorical imputation and one-hot encoding
4. Stratified train/test split
5. Class-balanced logistic regression training
6. Evaluation and model persistence
7. Prediction with the persisted preprocessing/model pipeline

### Baseline model and metrics

The baseline is logistic regression with `class_weight="balanced"`,
`max_iter=1000`, a 20% stratified test split, and random state 42. Keep these
settings fixed when comparing baseline results.

| Metric | Baseline |
|---|---:|
| Accuracy | Not yet measured |
| Precision | Not yet measured |
| Recall | Not yet measured |
| ROC-AUC | Not yet measured |

After running `python src/evaluate.py` with the dataset, record the four printed
values above. They are intentionally not estimated because no dataset was
provided with this repository.

### Prediction service

`src.predict.predict()` accepts a dictionary containing the dataset's feature
columns (without `customerID` or `Churn`) and returns a prediction (`Churn` or
`No Churn`) and churn probability. It reports missing or unexpected input
features rather than silently accepting a mismatched record.

### Key concepts

1. **Why `ColumnTransformer`?** It applies numeric and categorical
   preprocessing to their respective columns within one estimator.
2. **Why `Pipeline`?** It keeps preprocessing and the classifier together,
   applies the same transformations at training and inference, and helps
   prevent data leakage by fitting transformations only on training data.
3. **Why `stratify=y`?** It preserves the churn/non-churn proportions in both
   train and test splits.
4. **`predict()` vs `predict_proba()`:** `predict()` returns a class label;
   `predict_proba()` returns estimated probabilities for each class.
5. **Why save the whole pipeline?** The persisted artifact includes both the
   fitted preprocessing and classifier, so inference uses the same learned
   transformations.
6. **Why is recall important for churn?** Higher recall finds more customers
   who may leave, reducing missed retention opportunities. It should be
   balanced against precision and the cost of retention outreach.

## Future development

RetailMind AI will incrementally evolve into a complete retail AI platform
with additional machine-learning models, analytics, and deployment
capabilities.
