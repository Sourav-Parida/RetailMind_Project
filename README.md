# RetailMind AI — Customer Churn

RetailMind is a customer-churn prediction project built from the Telco Customer
Churn dataset. It implements four stages: a baseline, model comparison,
hyperparameter tuning, and feature-engineered ensemble selection.

## Project structure

```text

data/          Downloaded dataset
models/        Saved Day 1–4 model pipelines
notebooks/     
reports/       Evaluation tables and the Day 3 learning curve
src/           Reusable data, training, evaluation, and prediction code

```

## Setup

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and place its CSV in `data/`. The project recognizes these filenames:

- `Telco-Customer-Churn.csv`
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- `telco_customer_churn.csv`

Install the requirements:

```shell
python -m pip install -r requirements.txt
```

XGBoost is optional. If installed, it is included as a Day 4 candidate; the
Random Forest candidate runs without it.

## Run the project

Run all four stages from the repository root:

```shell
python -m src.run_pipeline
```

Optionally provide the CSV path:

```shell
python -m src.run_pipeline --data path\to\Telco-Customer-Churn.csv
```

The workflow uses one stratified 80/20 split with random state 42. Days 2–4
select candidates using three-fold cross-validation on the training partition.
The held-out test partition is reserved for reporting final metrics.

Run an individual stage with:

```shell
python -m src.train
python -m src.model_comparison
python -m src.hyperparameter_tuning
python -m src.day4_ensemble
python -m src.evaluate
python -m src.predict
```

`src.predict.predict(customer_data)` accepts a mapping for one customer or a
DataFrame for multiple customers. It returns `predicted_churn` (`Yes` or `No`)
and `churn_probability` using the Day 4 champion. Run the full pipeline first
to create that model artifact.

## Workflow

- **Day 1 — Baseline:** class-balanced Logistic Regression with numeric
  imputation/scaling and categorical imputation/one-hot encoding.

- **Day 2 — Model comparison:** Decision Tree, Random Forest, and RBF SVM;
  the champion is chosen by mean cross-validation recall.

- **Day 3 — Tuning:** GridSearchCV for those candidates, final test metrics,
  and a recall learning curve.

- **Day 4 — Feature engineering and ensembles:** tenure bands, average monthly
  charges, contract/payment indicators, `SelectKBest`, and Random Forest
  (plus optional XGBoost). The Day 4 pipeline includes feature engineering,
  preprocessing, and selection so predictions use the same transformations.

Churn `Yes` is the positive class. Downloaded data and generated model files
are excluded by Git. Do not publish the raw dataset unless its license and
repository policy allow it.
