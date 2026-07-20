"""Credit Risk Prediction

This script loads a credit dataset, preprocesses the data, trains three
classification models, evaluates them, and saves a feature importance plot.

Expected dataset location:
    data/german_credit_data.csv
"""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings("ignore")


DATA_PATH = Path("data") / "german_credit_data.csv"
OUTPUT_DIR = Path("outputs")
PLOT_PATH = OUTPUT_DIR / "feature_importance.png"
RANDOM_STATE = 42


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the credit dataset from disk."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. Please place german_credit_data.csv inside the data folder."
        )
    return pd.read_csv(file_path)


def find_target_column(dataframe: pd.DataFrame) -> str:
    """Identify the most likely target column in the dataset."""
    candidate_names = [
        "Risk",
        "risk",
        "Creditability",
        "creditability",
        "Class",
        "class",
        "Target",
        "target",
        "Label",
        "label",
    ]

    for column_name in candidate_names:
        if column_name in dataframe.columns:
            return column_name

    return dataframe.columns[-1]


def prepare_target(target_series: pd.Series) -> pd.Series:
    """Convert the target column into a binary numeric series."""
    cleaned = target_series.dropna().copy()

    if cleaned.nunique() != 2:
        raise ValueError("This project expects a binary classification target with exactly two classes.")

    if pd.api.types.is_numeric_dtype(cleaned):
        unique_values = set(cleaned.unique())
        if unique_values <= {0, 1}:
            return cleaned.astype(int)

        encoded = pd.factorize(cleaned)[0]
        return pd.Series(encoded, index=cleaned.index).astype(int)

    normalized = cleaned.astype(str).str.strip().str.lower()
    mapping = {
        "good": 1,
        "bad": 0,
        "yes": 1,
        "no": 0,
        "approved": 1,
        "rejected": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "positive": 1,
        "negative": 0,
    }

    if normalized.isin(mapping.keys()).all():
        return normalized.map(mapping).astype(int)

    encoded = pd.factorize(cleaned)[0]
    return pd.Series(encoded, index=cleaned.index).astype(int)


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """Create the preprocessing pipeline for numeric and categorical features."""
    numeric_features = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = feature_frame.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_model_pipeline(preprocessor: ColumnTransformer, classifier) -> Pipeline:
    """Wrap preprocessing and model training into a single pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def evaluate_model(
    model_name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Train a model pipeline and calculate evaluation metrics."""
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Pipeline": pipeline,
    }


def plot_feature_importance(trained_pipeline: Pipeline) -> None:
    """Save a feature importance chart from the random forest model."""
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    model = trained_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_frame = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values(by="Importance", ascending=False)

    top_features = importance_frame.head(15).iloc[::-1]

    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"], color="#1f77b4")
    plt.title("Top Feature Importances - Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    dataset = load_data(DATA_PATH)
    target_column = find_target_column(dataset)

    dataset = dataset.dropna(subset=[target_column]).copy()
    y = prepare_target(dataset[target_column])
    X = dataset.loc[y.index].drop(columns=[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if y.value_counts().min() >= 2 else None,
    )

    preprocessor = build_preprocessor(X_train)

    model_pipelines = {
        "Logistic Regression": build_model_pipeline(
            preprocessor,
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": build_model_pipeline(
            preprocessor,
            DecisionTreeClassifier(random_state=RANDOM_STATE),
        ),
        "Random Forest": build_model_pipeline(
            preprocessor,
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        ),
    }

    evaluation_results = []
    random_forest_pipeline = None

    for model_name, pipeline in model_pipelines.items():
        result = evaluate_model(model_name, pipeline, X_train, y_train, X_test, y_test)
        evaluation_results.append(result)

        if model_name == "Random Forest":
            random_forest_pipeline = result["Pipeline"]

    results_table = pd.DataFrame(evaluation_results)
    results_table = results_table[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]

    print("\nCredit Risk Model Evaluation")
    print("=" * 40)
    print(results_table.round(4).to_string(index=False))

    if random_forest_pipeline is not None:
        plot_feature_importance(random_forest_pipeline)
        print(f"\nFeature importance plot saved to: {PLOT_PATH}")
