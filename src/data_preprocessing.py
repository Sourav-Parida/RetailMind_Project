"""Load and prepare the Telco Customer Churn dataset."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_FILENAMES = (
    "Telco-Customer-Churn.csv",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "telco_customer_churn.csv",
)


def find_dataset(data_dir: str | Path = DATA_DIR) -> Path:
    """Find a supported Telco CSV in the project's data directory."""
    directory = Path(data_dir)
    for filename in DATA_FILENAMES:
        candidate = directory / filename
        if candidate.is_file():
            return candidate

    expected = ", ".join(DATA_FILENAMES)
    raise FileNotFoundError(
        f"No Telco Customer Churn CSV found in {directory}. "
        f"Place the downloaded dataset there using one of these names: {expected}"
    )


def load_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset from an explicit path or the project's data directory."""
    path = Path(file_path) if file_path is not None else find_dataset()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean the target and return model features and a binary churn target."""
    if "Churn" not in data.columns:
        raise ValueError("Dataset must include the 'Churn' target column.")

    cleaned = data.copy()
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"], errors="coerce"
        )

    target = cleaned.pop("Churn").map({"No": 0, "Yes": 1})
    if target.isna().any():
        invalid_targets = data.loc[target.isna(), "Churn"].drop_duplicates().tolist()
        raise ValueError(
            "Unexpected or missing values in 'Churn'; expected only 'Yes' and "
            f"'No'. Invalid values: {invalid_targets}"
        )

    features = cleaned.drop(columns=["customerID"], errors="ignore")
    if features.empty or features.shape[1] == 0:
        raise ValueError("Dataset must include at least one usable feature column.")

    return features, target.astype(int)


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the fixed stratified split used by all three project stages."""
    class_counts = target.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError(
            "A stratified train/test split requires both churn classes with at "
            "least two rows each."
        )

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build numeric and categorical preprocessing fitted within each pipeline."""
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
    transformers = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("Cannot preprocess a dataset without usable features.")

    return ColumnTransformer(transformers=transformers)


def make_model_pipeline(features: pd.DataFrame, estimator) -> Pipeline:
    """Keep preprocessing and the estimator together to prevent data leakage."""
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(features)),
            ("model", estimator),
        ]
    )


def make_cv() -> StratifiedKFold:
    """Return the shared three-fold stratified cross-validation strategy."""
    return StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def get_train_test_data(
    data_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, prepare, and split the dataset for a standalone project stage."""
    features, target = prepare_data(load_data(data_path))
    return split_data(features, target)
