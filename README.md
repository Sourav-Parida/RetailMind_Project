# RetailMind AI

RetailMind is a learning project built one step at a time. Its increments cover:

- **Days 1–5:** use customer information to predict whether a phone or internet
  customer may leave.

- **Days 6–7:** use product photos to predict what type of product is shown.

- **Day 9:** use past sales to forecast future store or product demand.

Each section below explains what the day does, how to run it, and where to
find its results. Day 7 covers image classification only; YOLO and object
detection are not included.

## What is in the folders?

- `data\` — datasets used by the project. These stay on your computer and are
  not uploaded to GitHub.
- `src\` — the Python programs that train or use the models.
- `notebooks\` — interactive lessons with the full training code in their
  cells, so you can run the lessons without relying on the Python scripts in
  `src\`.
- `models\` — trained models saved by the programs.
- `reports\` — result tables, charts, and confusion matrices.

## Set up the project

Install the project's Python packages:

```powershell
python -m pip install -r requirements.txt
```

### Get the customer dataset (Days 1–5)

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
Put its CSV file in `data\` and use one of these names:

- `Telco-Customer-Churn.csv`
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- `telco_customer_churn.csv`

The project uses the customer details as examples and the `Churn` column as
the answer it learns to predict. It sets aside 20% of the rows for a final
check and uses the rest to train and compare models.

### Get the product photos (Days 6–7)

Fashion-MNIST is downloaded automatically by Keras the first time its training
code runs. For catalog photos, download [Fashion Product Images Small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
and extract it like this:

```text
data\fashion-product-images-small\
    styles.csv
    images\
        15970.jpg
        ...
```

The loader also accepts a `myntradataset\` folder inside
`fashion-product-images-small\`. Day 7 downloads ResNet50's pretrained weights
from the internet the first time it runs. The catalog photos and downloaded
weights are not included in this repository.

### Get the demand dataset (Day 9)

Download either the [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
training CSV or the [Store Item Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only)
training CSV. Put it in one of these locations:

```text
data\rossmann-store-sales\train.csv
data\store-item-demand-forecasting\train.csv
```

The program recognizes the `Date`, `Store`, and `Sales` columns in Rossmann
data, and the `date`, `store`, `item`, and `sales` columns in Store Item Demand
data. If you use another location, set `DATA_CSV` near the top of
`src\day9_demand_forecasting.py`.

## Which model was used each day, and how accurate was it?

These are the scores recorded in the saved result files. For Days 1–5,
accuracy measures churn predictions on the held-back customer test data. Day 6
shows results for both Fashion-MNIST and the product catalog; Day 9 uses
forecast-error measurements instead of accuracy.

| Day | Model | Test accuracy or forecast error | What the result means |
|---|---|---:|---|
| 1 | Logistic Regression | **73.81%** | First customer-churn baseline |
| 2 | RBF Support Vector Machine (SVM) | **74.38%** | Chosen for its cross-validation recall; Random Forest had higher accuracy, at 78.14% |
| 3 | Tuned RBF SVM | **72.39%** | Best tuned model by cross-validation recall |
| 4 | Random Forest with feature selection | **77.36%** | Day 4 churn champion |
| 5 | Keras neural network (ANN) | **79.42%** | Higher accuracy than Day 4, but lower recall for customers who left |
| 6 | CNN on Fashion-MNIST | **91.42%** | Accuracy on the ten-class Fashion-MNIST test set |
| 6 | CNN on broad catalog labels (`masterCategory`) | **66.88%** | Accuracy on six broad product categories |
| 7 | ImageNet-pretrained ResNet50 | **Not recorded** | Model file exists, but no Day 7 test results or comparison report are available yet |
| 9 | LSTM daily demand forecaster | **Not measured** | Demand uses MAE and RMSE, not classification accuracy |

Accuracy is not the only measure of a model. In particular, a churn model can
have higher accuracy while missing more customers who leave. The detailed
sections below explain what each day's model does and where to find its
complete results. For Day 9, MAE and RMSE are not available until you train on
a demand dataset.

## Day 1 — Start with a simple churn model

**In plain language:** teach the computer to make a first guess about which
customers may leave, using their account details.

**What the program does**

- Fills in missing information and prepares text answers for the model.
- Trains a Logistic Regression model, with extra attention to customers who
  left.
- Measures how often its predictions are correct and how well it finds
  customers who leave.

**Run it**

```powershell
python -m src.train
```

**Files it creates**

- Trained model: `models\churn_day1_baseline.joblib`
- Results: `reports\day1_baseline_metrics.csv`
- Lesson: `notebooks\RetailMind_Churn_Days_1_to_3.ipynb`

The saved results show **73.81% accuracy**, **78.34% recall** for customers
who left, and **0.841 ROC-AUC**. See the results file for all measurements.

## Day 2 — Compare different churn models

**In plain language:** try three different kinds of models and keep the one
that does best at finding customers who leave.

**What the program tries**

- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

It tests each model several times on different parts of the training data.
This helps choose a model without relying on just one split.

**Run it**

```powershell
python -m src.model_comparison
```

**Files it creates**

- Selected model: `models\churn_day2_champion.joblib`
- Model results: `reports\day2_model_comparison.csv`
- Lesson: `notebooks\RetailMind_Churn_Days_1_to_3.ipynb`

The RBF SVM was selected by cross-validation recall. Its test accuracy was
**74.38%**. The Random Forest had the highest accuracy of these three models
at **78.14%**, but it found fewer customers who left, so it was not selected.
See the results file for each model's full measurements.

## Day 3 — Improve the Day 2 models

**In plain language:** try different settings for the Day 2 models and check
whether more training examples help.

**What the program does**

- Tries several settings for the Decision Tree, Random Forest, and SVM.
- Chooses settings using the training data, then checks the chosen model on
  the held-back test data.
- Draws a chart showing how results change as the amount of training data
  increases.

**Run it**

```powershell
python -m src.hyperparameter_tuning
```

**Files it creates**

- Tuned model: `models\churn_day3_tuned.joblib`
- Settings tried: `reports\day3_cv_tuning_results.csv`
- Test results: `reports\day3_test_metrics.csv`
- Learning chart and data: `reports\day3_learning_curve.png` and
  `reports\day3_learning_curve.csv`
- Lesson: `notebooks\RetailMind_Churn_Days_1_to_4.ipynb`

The tuned RBF SVM was selected by cross-validation recall. Its saved test
accuracy was **72.39%**. See the results file for the other measurements.

## Day 4 — Add useful customer details and compare models

**In plain language:** create a few extra details from the customer data, then
see if a group of decision trees can predict churn better.

**Extra details include**

- A customer's time with the company grouped into ranges.
- Average charges per month.
- Whether the customer has a month-to-month contract.
- Whether the customer pays by electronic check.

The program keeps the steps for creating and preparing these details together
with the model. This helps make sure predictions use the same steps as
training. XGBoost can also be tried if it is installed; it is optional.

**Run it**

Run Day 3 first, or run the full Days 1–5 workflow below. Then run:

```powershell
python -m src.day4_ensemble
```

**Files it creates**

- Selected model: `models\churn_day4_champion.joblib`
- Day 4 model results: `reports\day4_ensemble_comparison.csv`
- Day 3 and Day 4 comparison: `reports\day4_vs_day3_comparison.csv`
- Lesson: `notebooks\RetailMind_Churn_Days_1_to_4.ipynb`

The Random Forest with feature selection was selected in the saved run. Its
test accuracy was **77.36%**.

## Day 5 — Try a neural network for churn

**In plain language:** build a small neural network and compare it with the
Day 4 model using the same customer data.

**What the program does**

- Prepares numbers and text answers for a Keras neural network.
- Stops training when validation results stop improving.
- Compares the neural network and Day 4 model using the same final test data.

**Run it**

Run Day 4 first, or run the full Days 1–5 workflow below. Then run:

```powershell
python -m src.day5_ann
```

**Files it creates**

- Neural-network model: `models\churn_day5_ann.keras`
- Saved text and number settings:
  `models\churn_day5_ann_preprocessing.json`
- Comparison results: `reports\day5_ann_vs_classical.csv`
- Lesson: `notebooks\RetailMind_Churn_Day_5_ANN.ipynb`

In the saved results, the neural network was correct **79.42%** of the time,
compared with **77.36%** for Day 4. However, it found **55.88%** of customers
who left, compared with **62.30%** for Day 4. Neither model is best in every
measure, so check the comparison file before choosing one.

## Day 6 — Recognize product categories in photos

**In plain language:** first try a CNN on a small standard clothing-photo
dataset, then train a separate CNN to assign broad categories to catalog
photos. Fashion-MNIST keeps its own ten labels; the catalog model uses
`masterCategory`.

**What the program does**

- Trains a small image model on Fashion-MNIST's ten clothing categories.
- Reads product photo names and broad `masterCategory` labels from `styles.csv`.
- Skips photos that are missing and categories with fewer than three
  examples.
- Resizes and slightly changes training images to help the model learn.
- Saves the catalog model and a table showing which categories it confused.

**Run it**

```powershell
python src\day6_cnn.py
```

This trains Fashion-MNIST first and the product catalog second. Both models
save their test accuracy and test loss to separate report files. The full
catalog has tens of thousands of photos and can take a long time to train,
especially on a CPU. The [Day 6 notebook](notebooks/RetailMind_Day_6_CNN_Image_Classification.ipynb)
also has separate cells if you want to run one model at a time.

**Files it creates**

- Fashion-MNIST model: `models\fashion_mnist_prototype.keras`
- Fashion-MNIST accuracy and loss: `reports\day6_fashion_mnist_metrics.csv`
- Fashion-MNIST confusion table:
  `reports\day6_fashion_mnist_confusion_matrix.csv`
- Master-category model: `models\vision_master_category_classifier.keras`
- Category names: `models\vision_master_category_class_names.json`
- Catalog confusion table: `models\vision_master_category_confusion_matrix.csv`
- Catalog test loss and accuracy:
  `reports\day6_master_category_metrics.csv`

The Fashion-MNIST model reached **91.42% accuracy** on its test photos. The
earlier catalog model reached **35.31% accuracy** when predicting detailed
`articleType` labels (132 product types). The new catalog model predicts the
broader `masterCategory` labels instead, so that old score does not describe
the new model. The new model reached **66.88% accuracy** across six broad
categories. Both current Day 6 scores are recorded in their corresponding
`reports\day6_*_metrics.csv` files.

To predict an image after training the catalog model, open `src\day6_cnn.py`
and set `image_path` near the bottom of the file to your image's path. Running
the script will predict that image instead of training. Leave the example
assignment commented out to train both models. The prediction prints the most
likely master category and confidence, plus the top three category guesses.

## Day 7 — Use a pretrained model for product photos

**In plain language:** start with a ResNet50 model that has already learned
general image patterns, teach it the catalog's broad master categories, then
adjust some of its image-recognition layers.

**What the program does**

- Learns a new category output layer while the pretrained image model is
  frozen.
- Then fine-tunes the top part of the pretrained model using a smaller
  learning rate.
- Compares Day 7 with the Day 6 catalog CNN using the same test photos.
- Saves the ResNet50 model, its category labels, test comparison, and
  confusion table.

**Before running**

Train the Day 6 catalog model first. Day 7 uses its saved model and labels for
the comparison.

**Run it**

```powershell
python src\day7_transfer_learning.py
```

Or follow the [Day 7 notebook](notebooks/RetailMind_Day_7_Transfer_Learning.ipynb).

**Files it creates**

- ResNet50 model: `models\vision_master_category_resnet50.keras`
- Master-category names: `models\vision_master_category_resnet50_classes.json`
- Day 6 and Day 7 comparison: `reports\day7_transfer_learning_comparison.csv`
- Confusion table: `models\day7_master_category_confusion_matrix.csv`

The ResNet50 model file is present, but a Day 7 test-accuracy or comparison
report is not available. Its accuracy is therefore **not recorded** here.
Training the full photo collection can take a long time on a CPU. YOLO and
object detection are not included.

## Day 9 — Forecast store and product demand

**In plain language:** use recent daily sales to estimate the next day's
demand for each store, or each store and product combination.

**What the program does**

- Reads either Rossmann Store Sales or Store Item Demand data.
- Sums duplicate daily sales and fills missing dates with zero sales.
- Uses the previous 28 days to predict the next day.
- Trains one LSTM across the available store or store-item series.
- Keeps the latest 20% of dates aside for the final test and uses an earlier
  10% for validation, preserving date order.
- Reports MAE and RMSE, saves test predictions, forecasts the next day for
  each series, and creates an actual-versus-predicted SVG plot.

**Run it**

```powershell
python src\day9_demand_forecasting.py
```

You can also run the shorter step-by-step version in
[the Day 9 notebook](notebooks/RetailMind_Day_9_Demand_Forecasting.ipynb).

Download one of the datasets and place its `train.csv` in a location described
in [Get the demand dataset (Day 9)](#get-the-demand-dataset-day-9). Training
may take time, depending on how many store/product series the dataset contains.

**Files it creates**

- Trained LSTM: `models\day9_demand_forecaster.keras`
- Model settings and per-series scaling: `models\day9_demand_forecaster_metadata.json`
- Test MAE/RMSE: `reports\day9_demand_forecast_metrics.csv`
- Actual and predicted test sales: `reports\day9_demand_test_predictions.csv`
- Next-day demand by store/product: `reports\day9_demand_next_day_forecast.csv`
- Example actual-versus-predicted chart: `reports\day9_demand_forecast.svg`

MAE and RMSE will be available in the metrics file after training. The
`forecast_next_demand` function in `src\day9_demand_forecasting.py` can also
produce a next-day estimate for a trained store or store-product series.

## Run all customer-churn days

This command runs **Days 1–5 only**. It does not run the separate image
projects for Days 6–7 or the demand forecast for Day 9.

```powershell
python -m src.run_pipeline
```

If the Telco CSV is somewhere else, provide its path:

```powershell
python -m src.run_pipeline --data path\to\Telco-Customer-Churn.csv
```

To check the saved Day 4 model:

```powershell
python -m src.evaluate
```

For code-based predictions, `src.predict.predict(customer_data)` accepts
details for one customer or several customers. It returns a churn prediction
(`Yes` or `No`) and a probability. Run the customer-churn workflow first so
the Day 4 model exists.

## A few things to remember

- **Accuracy** is how often the model is right overall.
- **Recall** tells how many of the customers who actually left were found.
- A **confusion matrix** shows which types the model predicted correctly or
  mixed up.
- Day 6 and Day 7 classify one product type per image. They do not draw boxes
  around products or find several products in one image.
- Datasets and trained model files are kept off GitHub. Check dataset
  permissions before sharing downloaded data.
