# RetailMind AI

RetailMind is a learning project built in small steps. Each day adds a new
machine-learning feature to the same project.

## Project map: what each group of days does

- **Phase 1 — Days 1–4: Customer churn.** Use customer account details to
  estimate who may leave. Start with a simple model, compare alternatives,
  tune them, then add useful customer features.
- **Phase 2 — Day 5: Neural network for churn.** Try a Keras neural network
  and compare it with the Day 4 model.
- **Phase 3 — Days 6–8: Product photos.** Classify catalog photos with a
  CNN, improve classification with ResNet50, then explain predictions with
  Grad-CAM.
- **Phase 4 — Days 9–10: Demand forecasting.** Forecast next-day sales with
  an LSTM, then use attention for multi-day planning.
- **Phase 5 — Day 11: Vision augmentation.** Generate candidate images for
  the underrepresented Free Items category and evaluate approved images.
- **Phase 6 — Day 12: Text intelligence.** Fine-tune a pretrained
  Transformers model for review sentiment and review-derived ticket routing.
- **Phase 7 — Day 13: Classical text baseline.** Use NLTK and TF-IDF Logistic
  Regression as a reference for the Day 12 BERT sentiment model.

### Results at a glance

These are the test results saved in the project's `reports\` files. Accuracy
is the percentage of held-out test examples classified correctly. Demand
forecasting uses error measures instead of classification accuracy.

| Day | Topic / task | One-line explanation | Accuracy / result |
|---:|---|---|---:|
| 1 | Churn baseline — Logistic Regression | Uses customer account details for a first estimate of who may leave. | **73.81% accuracy** |
| 2 | Compare churn models — RBF SVM selected | Tests three models and selects the SVM based on cross-validation recall. | **74.38% accuracy** |
| 3 | Tune churn models — RBF SVM | Tunes model settings to find more customers who may leave. | **72.39% accuracy** |
| 4 | Feature engineering — Random Forest | Adds useful customer details and selects the strongest churn features. | **77.36% accuracy** |
| 5 | Churn neural network — Keras ANN | Compares a neural network with the Day 4 churn model. | **79.42% accuracy** |
| 6 | Clothing image classification — Fashion-MNIST CNN | Classifies images into Fashion-MNIST's ten clothing labels. | **91.42% accuracy** |
| 6 | Catalog classification — CNN | Predicts one of six broad product categories from catalog photos. | **66.88% accuracy** |
| 7 | Catalog classification — ResNet50 | Fine-tunes a pretrained image model and compares it with Day 6. | **92.93% accuracy** |
| 8 | Explainable vision — Grad-CAM | Highlights image regions that influenced the Day 7 category prediction. | No new accuracy score |
| 9 | Demand forecasting — LSTM | Uses 28 days of sales to forecast next-day store-item demand. | **MAE 6.70 / RMSE 8.83 sales** |
| 10 | Multi-horizon demand — Seq2Seq + attention | Forecasts 7 days and compares its first-day prediction with Day 9. | **7-day MAE 8.72 / RMSE 11.57 sales** |
| 11 | Vision augmentation — VAE | Generates candidates for a rare category; only visually approved samples enter training. | Candidates were too blurry; none approved, so no classifier lift is claimed |
| 12 | Text intelligence — BERT-family model | Fine-tunes a pretrained model for review sentiment and ticket routing. | **91.67% accuracy** (separate split from baseline) |
| 13 | Review analytics — TF-IDF + Logistic Regression | Classifies review sentiment and surfaces terms associated with each class. | **87.33% accuracy** |

Day 2's Random Forest had higher accuracy (**78.14%**) than the selected SVM,
but the SVM was selected because its cross-validation recall was higher. For
churn, recall is important because it measures how many customers who actually
left were found. Day 5's higher accuracy came with lower churn recall than
Day 4 (**55.88%** compared with **62.30%**).

Day 7's **92.93%** and Day 6 catalog's **66.88%** are from the saved
Day 6/Day 7 comparison report. Day 8 reuses the Day 7 model and creates
visual explanations; it is not a new model or accuracy test.

## Set up the project

Install the project's Python packages:

```powershell
python -m pip install -r requirements.txt
```

### Customer dataset for Days 1–5

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Put its CSV file in `data\` with one of these names:

- `Telco-Customer-Churn.csv`
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- `telco_customer_churn.csv`

The `Churn` column tells the model whether a customer left. Training uses
part of the data and keeps a separate test portion for evaluation.

### Product photos for Days 6–8

Fashion-MNIST is downloaded automatically by Keras when needed. For the
product catalog, download [Fashion Product Images Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
and place the files like this:

```text
data\fashion-product-images-small\
    styles.csv
    images\
        15970.jpg
        ...
```

The loader also accepts a `myntradataset\` subfolder. Day 7 downloads
ResNet50's pretrained weights the first time it runs. Day 8 uses the saved
Day 7 model. Day 11 uses the same catalog and saved Day 7 model. Product
photos and downloaded model weights are not included in the repository.

### Sales dataset for Day 9

Download either [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
or [Store Item Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only).
Place its `train.csv` file in one of these folders:

```text
data\rossmann-store-sales\train.csv
data\store-item-demand-forecasting\train.csv
```

The loader recognizes the expected date, store, and sales columns for either
dataset. To use a different file location, change `DATA_CSV` near the top of
`src\day9_demand_forecasting.py`.

### Amazon reviews for Days 12–13

Download [Amazon Polarity](https://huggingface.co/datasets/amazon_polarity)
or use a Kaggle review CSV. Amazon Polarity CSVs with or without a header are
supported; headerless rows should contain `label,title,content`. The loader
also accepts named Kaggle columns such as `Score,Summary,Text`. Save it as
`data\amazon-reviews\train.csv` or pass its path with `--data`. The script
trains on a balanced sample of up to 3,000 reviews per sentiment to keep
training manageable.

The Day 12 BERT module uses the same review CSV and downloads its pretrained
model and tokenizer the first time it runs. The support-ticket categories are
weak labels generated from review keywords for a routing demonstration; they
are not a substitute for labeled support-ticket data.

## Phase 1 — Days 1–4: Build customer-churn models

The first four days teach the computer to estimate whether a customer may
leave, then improve and compare different models. They use the same Telco
customer dataset.

### Day 1 — First churn prediction

**In brief:** Train a simple Logistic Regression model as a starting point.

**What it does:** Fills missing values, prepares numeric and text customer
details, trains the model, and evaluates its predictions. The saved test
results are **73.81% accuracy**, **78.34% recall**, and **0.841 ROC-AUC**.

**Run:**

```powershell
python -m src.train
```

**Outputs:** `models\day1_churn_baseline.joblib` and
`reports\day1_baseline_metrics.csv`.

**Notebook:** `notebooks\RetailMind_Churn_Days_1_to_3.ipynb`.

### Day 2 — Compare churn models

**In brief:** Compare a Decision Tree, Random Forest, and SVM.

**What it does:** Uses cross-validation to choose a model, emphasizing recall
for customers who leave. The RBF SVM was selected; its test accuracy was
**74.38%**. The Random Forest reached **78.14%** accuracy but had lower
cross-validation recall.

**Run:**

```powershell
python -m src.model_comparison
```

**Outputs:** `models\day2_churn_champion.joblib` and
`reports\day2_model_comparison.csv`.

**Notebook:** `notebooks\RetailMind_Churn_Days_1_to_3.ipynb`.

### Day 3 — Tune the churn models

**In brief:** Try different model settings and measure whether more data
helps.

**What it does:** Tunes the Decision Tree, Random Forest, and SVM using
cross-validation. The tuned RBF SVM had **72.39% test accuracy** and
**80.48% test recall**. It was chosen for cross-validation recall, not for
having the highest accuracy.

**Run:**

```powershell
python -m src.hyperparameter_tuning
```

**Outputs:** `models\day3_churn_tuned.joblib`,
`reports\day3_cv_tuning_results.csv`, `reports\day3_test_metrics.csv`, and
the learning-curve files `reports\day3_learning_curve.csv` and
`reports\day3_learning_curve.png`.

**Notebook:** `notebooks\RetailMind_Churn_Days_1_to_4.ipynb`.

### Day 4 — Add customer features and select a model

**In brief:** Create extra details from customer data and train a
feature-selected Random Forest.

**What it does:** Adds useful features such as tenure groups and charge
relationships, then compares results with Day 3. The saved model reached
**77.36% test accuracy** and **62.30% recall**.

**Run:**

```powershell
python -m src.day4_ensemble
```

Run Day 3 first, or run the full Days 1–5 customer workflow described near
the end of this README.

**Outputs:** `models\day4_churn_champion.joblib`,
`reports\day4_ensemble_comparison.csv`, and
`reports\day4_vs_day3_comparison.csv`.

**Notebook:** `notebooks\RetailMind_Churn_Days_1_to_4.ipynb`.

## Phase 2 — Day 5: Try a neural network for churn

### Day 5 — Keras ANN

**In brief:** Train a small neural network and compare it with Day 4.

**What it does:** Prepares customer data for a Keras ANN, stops training if
validation results stop improving, and compares both models on the same test
data. The ANN reached **79.42% accuracy**, compared with **77.36%** for Day 4.
However, its recall was **55.88%**, lower than Day 4's **62.30%**.

**Run Day 4 first, then run:**

```powershell
python -m src.day5_ann
```

**Outputs:** `models\day5_churn_ann.keras`,
`models\day5_churn_ann_preprocessing.json`, and
`reports\day5_ann_vs_classical.csv`.

**Notebook:** `notebooks\RetailMind_Churn_Day_5_ANN.ipynb`.

## Phase 3 — Days 6–8: Classify and explain product photos

These days use product pictures. The catalog label is `masterCategory`, a
broad category such as Apparel or Footwear.

### Day 6 — Train a CNN

**In brief:** Try a CNN on Fashion-MNIST, then train another CNN on catalog
photos.

**What it does:** Fashion-MNIST has ten clothing labels. The catalog model
predicts six broad `masterCategory` labels. Their saved test accuracies are
**91.42%** and **66.88%**, respectively; the scores use different datasets
and should not be compared as if they were the same test.

**Run:**

```powershell
python src\day6_cnn.py
```

The script trains both models. Training the full photo catalog can take a
long time, especially on a CPU.

**Outputs:** Fashion-MNIST model `models\day6_fashion_mnist_prototype.keras`,
catalog model `models\day6_master_category_cnn.keras`, and the
metrics files `reports\day6_fashion_mnist_metrics.csv` and
`reports\day6_master_category_metrics.csv`. Confusion tables are also saved
for each model.

**Notebook:** `notebooks\RetailMind_Day_6_CNN_Image_Classification.ipynb`.

### Day 7 — Transfer learning with ResNet50

**In brief:** Reuse a ResNet50 model that learned general image features,
adapt it to the catalog categories, and compare it with Day 6.

**What it does:** First freezes ResNet50 and trains a new category-prediction
head. Then fine-tunes the top ResNet50 layers using a smaller learning rate.
The saved comparison report records **92.93%** Day 7 test accuracy and
**66.88%** for the Day 6 CNN on the same held-out catalog photos.

**Before running:** Train the Day 6 catalog model first so Day 7 can compare
against it.

**Run:**

```powershell
python src\day7_transfer_learning.py
```

To classify one photo with the saved Day 7 model:

```powershell
python src\day7_transfer_learning.py --classify data\fashion-product-images-small\images\15970.jpg
```

**Outputs:** `models\day7_master_category_resnet50.keras`,
`models\day7_master_category_class_names.json`, and
`reports\day7_transfer_learning_comparison.csv`.

**Notebook:** `notebooks\RetailMind_Day_7_Transfer_Learning.ipynb`.

### Day 8 — Explain a prediction with Grad-CAM

**In brief:** Highlight image regions that influenced the Day 7 classifier's
prediction.

**What it does:** Reuses the Day 7 ResNet50 model; it does not train a new
classifier and does not produce a new accuracy score. The heatmap is a visual
explanation, not proof that the prediction is correct.

**Run:**

```powershell
python src\day8_explainable_vision.py
```

This explains a few catalog photos by default. To choose a photo, pass its
path:

```powershell
python src\day8_explainable_vision.py data\fashion-product-images-small\images\15970.jpg
```

**Outputs:** PNG overlays in `reports\day8_gradcam\`.

**Notebook:** `notebooks\RetailMind_Day_8_Explainable_Vision.ipynb`.

## Phase 4 — Days 9–10: Forecast future demand

### Day 9 — LSTM demand forecasting

**In brief:** Use recent daily sales to estimate the next day's demand for a
store or a store/product combination.

**What it does:** Loads either supported sales dataset, combines duplicate
daily records, fills missing dates, and uses the previous 28 days to predict
the next day. The trained Store Item Demand model covers **500 store-item
series** and was evaluated on **183,000 test windows** from
**2016-12-31 through 2017-12-31**. Its test **MAE is 6.70 sales** and its
test **RMSE is 8.83 sales**; lower values mean predictions are closer to
actual sales. The model trained for 6 epochs.

**Run:**

```powershell
python src\day9_demand_forecasting.py
```

**Outputs:** `models\day9_demand_forecaster.keras`,
`reports\day9_demand_forecast_metrics.csv` (including the test scores),
`reports\day9_demand_test_predictions.csv`,
`reports\day9_demand_next_day_forecast.csv`, and
`reports\day9_demand_forecast.svg` (actual versus predicted sales).

**Notebook:** `notebooks\RetailMind_Day_9_Demand_Forecasting.ipynb`.

### Day 10 — Seq2Seq forecasting with attention

**In brief:** Use the previous 28 days to forecast the next 7 days.

**What it does:** Encodes a short sales history, decodes a 7-day forecast,
and applies attention so each forecast day can use information from the
history. To keep the run manageable, training uses up to 24 representative
store or store-item series, a small 16-unit LSTM, and at most 5 epochs.
On the current Store Item Demand run, the 7-day test results were **MAE
8.72** and **RMSE 11.57 sales** across **8,640 test windows**. On the same
windows, the Day 9 model had **6.62 one-day MAE**, while Day 10 had **9.11
one-day MAE**; lower error is better. Day 10 is a compact multi-day
experiment, and its current one-day result is worse than the Day 9 baseline.
The run also saves an attention heatmap.

Train Day 9 first so its saved model is available for the one-day comparison.

**Run:**

```powershell
python src\day10_seq2seq_attention.py
```

Optionally specify the sales CSV:

```powershell
python src\day10_seq2seq_attention.py --data data\store-item-demand-forecasting\train.csv
```

**Outputs:** `models\day10_seq2seq_attention.keras`,
`reports\day10_forecast_metrics.csv`, `reports\day10_test_predictions.csv`,
`reports\day10_next_7_days.csv`, and attention weights in
`reports\day10_attention_weights.csv` with a visualization at
`reports\day10_attention.svg`.

**Notebook:** `notebooks\RetailMind_Day_10_Seq2Seq_Attention.ipynb`.

## Phase 5 — Day 11: Augment a rare product-image category

### Day 11 — VAE image augmentation

**In brief:** Train a convolutional VAE on the training split of the rare
`Free Items` class. Its largest article subtype, `Free Gifts`, provides a
subset for learning a generator.

**What it does:** Day 6's saved confusion matrix shows zero correct
predictions for `Free Items` and `Sporting Goods`; the catalog has 105
`Free Items` images versus thousands for the common categories. The VAE
trains only on the `Free Gifts` subset of the training split, keeping the
classifier's validation and test images out of generator training.

Generate candidate samples:

```powershell
python src\day11_vae_augmentation.py --generate
```

Inspect `reports\day11_vae_candidates.png` for realism and diversity. Copy
only realistic candidate JPEGs into
`data\fashion-product-images-small\day11_synthetic_free_items\approved\`.
Then fine-tune the saved Day 7 classifier and compare per-class results on
the unchanged Day 6 test split:

```powershell
python src\day11_vae_augmentation.py --evaluate
```

The generated candidates were visibly blurry and lacked realistic product
detail, so none were approved. The classifier comparison was therefore not
run; no fairness or accuracy lift is claimed. The script will not train with
unreviewed candidates. It saves VAE training history and candidate diversity;
baseline/augmented confusion matrices, per-class accuracy, and the augmented
classifier are produced only after approved images are supplied. Synthetic
images should not be treated as real customer or product diversity.

## Phase 6 — Day 12: Understand review sentiment and route tickets

### Day 12 — Fine-tune a Transformers text classifier

**In brief:** Fine-tune a pretrained DistilBERT model for review sentiment
and a second classifier for review-derived support-ticket categories.

**What it does:** Uses the Day 13 TF-IDF baseline's saved metrics as a
reference, evaluates BERT with per-class precision, recall, and F1, and
exports a comparison report. Ticket categories are assigned by transparent
keyword rules over reviews, so their scores are a weak-label demonstration,
not an evaluation on real support tickets. The completed CPU run reached
**91.67% sentiment accuracy** versus the Day 13 baseline's **87.33%**; these
are separate holdouts, so the 4.34-point difference is indicative, not a
controlled same-example comparison. The review-derived router reached
**33.65% accuracy** and is not ready for real ticket routing.

Run the Day 13 baseline first so the comparison metrics are available:

```powershell
python src\day13_review_sentiment.py
```

Then train both Day 12 models:

```powershell
python src\day12_bert_text_intelligence.py
```

The default model is `distilbert-base-uncased`, trained for one epoch. Use
`--max-per-class`, `--ticket-max-per-class`, `--epochs`, or `--model-name` to
adjust training. The first run downloads the pretrained weights from the
Hugging Face Hub.

**Outputs:** `models\day12_bert_sentiment\`,
`models\day12_bert_ticket_router\`, `reports\day12_bert_metrics.csv`,
`reports\day12_ticket_routing_metrics.csv`, and
`reports\day12_bert_comparison.csv`. For inference, use
`predict_sentiment(text)` or `route_support_ticket(text)` from
`src.day12_bert_text_intelligence`.

## Phase 7 — Day 13: Preserve the classical review baseline

### Day 13 — Classical sentiment baseline

**In brief:** Clean review text, represent it with TF-IDF, and classify it
as positive or negative with Logistic Regression.

**What it does:** Uses NLTK to lowercase and tokenize reviews, removes
English stopwords while retaining negation terms, and trains a classical
sentiment model. It reports held-out accuracy and precision/recall, and
exports the terms with the strongest positive and negative model weights.
The default balanced sample is capped at 3,000 reviews per class. On the
provided Amazon Polarity training CSV, the saved 80/20 split reached
**87.33% accuracy**.

Download or prepare a review CSV with labels and review text. Install the
NLTK stopwords once:

```powershell
python -c "import nltk; nltk.download('stopwords')"
```

**Run:**

```powershell
python src\day13_review_sentiment.py
```

For a different CSV path or sample size:

```powershell
python src\day13_review_sentiment.py --data path\to\reviews.csv --max-per-class 3000
```

**Outputs:** `models\day13_review_sentiment.joblib`,
`reports\day13_review_metrics.csv`, and
`reports\day13_informative_terms.csv`. The module also provides
`predict_reviews()` for scoring new review text after training.

**Notebook:** `notebooks\RetailMind_Day_13_Review_Sentiment.ipynb`.

## Run the full customer-churn workflow

This command runs **Days 1–5 only**. It does not run the product-photo or
demand-forecasting projects.

```powershell
python -m src.run_pipeline
```

If the Telco CSV is somewhere else, provide its path:

```powershell
python -m src.run_pipeline --data path\to\Telco-Customer-Churn.csv
```

To check the saved Day 4 model, run `python -m src.evaluate`. For code-based
predictions, use `src.predict.predict(customer_data)` after training Day 4.

## Simple terms

- **Accuracy:** how often a model's predictions were correct.
- **Recall:** of the customers who actually left, how many the model found.
- **MAE and RMSE:** measures of how far demand forecasts are from actual
  sales; lower values are better.
- **Grad-CAM:** a heatmap showing which image regions influenced a
  classification prediction.

Datasets and trained models are kept off GitHub. Check dataset permissions
before sharing downloaded data.
