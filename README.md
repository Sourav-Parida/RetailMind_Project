# RetailMind AI — Customer Churn

RetailMind is an incrementally built retail AI project. Days 1–5 develop the
customer-churn workflow; Day 6 adds a separate TensorFlow/Keras image
classification module for product catalog auto-tagging.

## Project structure

```text

data/          Downloaded dataset
models/        Saved Day 1–6 model artifacts
notebooks/     Day 1–6 project lessons
reports/       Evaluation tables, learning curves, ANN results, and CNN results
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

Run all five stages from the repository root:

```shell
python -m src.run_pipeline
```

Optionally provide the CSV path:

```shell
python -m src.run_pipeline --data path\to\Telco-Customer-Churn.csv
```

The workflow uses one stratified 80/20 split with random state 42. Days 2–4
select candidates using three-fold cross-validation on the training partition.
The Day 5 ANN uses an internal validation split and early stopping. The held-out
test partition is reserved for reporting final metrics and the Day 4/5 benchmark.

Run an individual stage with:

```shell
python -m src.train
python -m src.model_comparison
python -m src.hyperparameter_tuning
python -m src.day4_ensemble
python -m src.day5_ann
python -m src.evaluate
python -m src.predict
```

`src.predict.predict(customer_data)` accepts a mapping for one customer or a
DataFrame for multiple customers. It returns `predicted_churn` (`Yes` or `No`)
and `churn_probability` using the Day 4 champion. Run the full pipeline first
to create that model artifact.

The Day 5 ANN can be run after Day 4 with `python -m src.day5_ann`. Its Keras
model and preprocessing metadata are saved separately from the classical
prediction pipeline.

Run both Day 6 models directly from the repository root:

```shell
python src\day6_cnn.py
```

The script trains Fashion-MNIST first and then the catalog classifier using
`data\fashion-product-images-small\`. To run only one model, comment out the
other training call in the `if __name__ == "__main__":` block at the bottom of
`src/day6_cnn.py`.

The Fashion-MNIST prototype is saved to `models/day6_fashion_mnist_cnn.keras`.
The product catalog model is saved to `models/day6_product_catalog_cnn.keras`.
Both modes save test accuracy and a confusion matrix under `reports/`. Product
labels use the dataset's `articleType` column; classes with fewer than three
available images are excluded so each split can include that class.
The downloaded catalog folder is Git-ignored and will not be added to commits.

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

- **Day 5 — Neural network benchmark:** a Keras ANN normalizes numeric inputs,
  encodes categories with `StringLookup`, and trains two hidden layers with
  early stopping on validation loss. It is compared with the Day 4 classical
  champion on the same held-out split. The model is saved as
  `models/churn_day5_ann.keras`; its category vocabularies and numeric
  imputation values are saved alongside it.

## When to use which model

The ANN is an experiment, not an automatic promotion. For this modest,
structured Telco dataset, the Day 4 Random Forest/SVM may be preferable when
they match or beat ANN quality: they are easier to train, need fewer runtime
dependencies, and are simpler to inspect. Consider the ANN when repeated
validation evidence shows a useful lift, or when the product has enough
tabular data/scale to justify neural-network training and serving. Compare
recall, precision, ROC-AUC, training cost, and operational needs—not depth
alone—and keep the same held-out test set out of model selection.

On the current fixed split at a 0.5 threshold, the ANN reached 0.841 ROC-AUC
and 0.794 accuracy versus 0.826 and 0.774 for the Day 4 champion, but its churn
recall was lower (0.559 versus 0.623) and its F1 was similar (0.590 versus
0.594). That is a trade-off, not an across-the-board win; the comparison is
recorded in `reports/day5_ann_vs_classical.csv`.

## Day 6 — CNN image classification

The CNN uses Fashion-MNIST to validate the image model workflow and supports
training on the Kaggle Fashion Product Images Small catalog. Product images
are resized and augmented during training; the held-out catalog test split is
evaluated with accuracy and a confusion matrix. Record the catalog test
accuracy here after running the catalog training command: **not yet measured**
(the Kaggle product-image dataset is local-only and Git-ignored).

The Fashion-MNIST prototype reached **86.38% test accuracy** after three
training epochs on the standard 10,000-image test set. This validates the CNN
workflow only; it is not the accuracy of the catalog classifier.

Churn `Yes` is the positive class. Downloaded data and generated model files
are excluded by Git. Do not publish the raw dataset unless its license and
repository policy allow it.
