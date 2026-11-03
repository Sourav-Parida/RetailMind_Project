# RetailMind AI — Customer Churn

RetailMind is a customer-churn prediction project built from the Telco Customer
Churn dataset. It covers the notebook workflow through **Day 3 only**: a
baseline, model comparison, and hyperparameter tuning with a learning curve.
Day 4 is intentionally excluded.

## Project structure

```text
data/          Downloaded dataset (not committed)
models/        Saved Day 1–3 model pipelines
notebooks/     RetailMind_Churn_Days_1_to_4.ipynb (Days 1–3 sections only)
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

## Run the project

Run all three stages from the repository root:

```shell
python -m src.run_pipeline
```

You can optionally provide a CSV path:

```shell
python -m src.run_pipeline --data path\to\Telco-Customer-Churn.csv
```

The workflow uses one stratified 80/20 split with random state 42. Days 2 and 3
select models using three-fold cross-validation on the training partition.
The held-out test partition is reserved for reporting final metrics.

Run individual stages with:

```shell
python -m src.train
python -m src.model_comparison
python -m src.hyperparameter_tuning
python -m src.evaluate
python -m src.predict
```

`src.predict.predict(customer_data)` accepts a mapping for one customer or a
DataFrame for multiple customers. It returns `predicted_churn` (`Yes` or `No`)
and `churn_probability` using the Day 3 tuned pipeline.

## Workflow

- **Day 1 — Baseline:** class-balanced Logistic Regression with numeric
  imputation/scaling and categorical imputation/one-hot encoding.
- **Day 2 — Model comparison:** Decision Tree, Random Forest, and RBF SVM;
  the champion is chosen by mean cross-validation recall.
- **Day 3 — Tuning:** GridSearchCV for those candidates, final test metrics,
  and a recall learning curve.

Churn `Yes` is the positive class. Downloaded data and generated model files
are excluded by Git. Do not publish the raw dataset unless its license and
repository policy allow it.
