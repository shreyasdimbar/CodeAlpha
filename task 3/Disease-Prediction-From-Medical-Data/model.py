"""Disease prediction from medical data.

This script loads a heart disease dataset, preprocesses the features,
trains multiple classification models, evaluates them, saves a comparison
table, generates visual reports, and stores the trained Random Forest model.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


# Project paths
DATA_PATH = Path("data") / "heart_disease.csv"
OUTPUT_DIR = Path("outputs")
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "random_forest_feature_importance.png"
MODEL_PATH = Path("disease_prediction_model.pkl")
RANDOM_STATE = 42


# Keep generated output folders available before saving plots.
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the medical dataset from CSV."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. Place heart_disease.csv inside the data folder."
        )
    return pd.read_csv(file_path)


def detect_target_column(dataframe: pd.DataFrame) -> str:
    """Detect the most likely target column name."""
    candidate_names = [
        "target",
        "Target",
        "output",
        "Output",
        "disease",
        "Disease",
        "presence",
        "Presence",
        "condition",
        "Condition",
        "heart_disease",
        "HeartDisease",
        "class",
        "Class",
    ]

    for column_name in candidate_names:
        if column_name in dataframe.columns:
            return column_name

    return dataframe.columns[-1]


def encode_target(target_series: pd.Series) -> pd.Series:
    """Convert the target column to binary numeric labels."""
    cleaned = target_series.dropna().copy()

    if cleaned.nunique() != 2:
        raise ValueError("The dataset must contain exactly two classes for binary classification.")

    if pd.api.types.is_numeric_dtype(cleaned):
        unique_values = set(cleaned.unique())
        if unique_values <= {0, 1}:
            return cleaned.astype(int)

        encoded = pd.factorize(cleaned)[0]
        return pd.Series(encoded, index=cleaned.index).astype(int)

    normalized = cleaned.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "no": 0,
        "present": 1,
        "absent": 0,
        "true": 1,
        "false": 0,
        "positive": 1,
        "negative": 0,
        "disease": 1,
        "no disease": 0,
        "1": 1,
        "0": 0,
    }

    if normalized.isin(mapping.keys()).all():
        return normalized.map(mapping).astype(int)

    encoded = pd.factorize(cleaned)[0]
    return pd.Series(encoded, index=cleaned.index).astype(int)


def build_preprocessor(feature_frame: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipelines for numeric and categorical features."""
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
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """Combine preprocessing and classification into one pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )


def get_probability_scores(model, x_test: pd.DataFrame) -> np.ndarray:
    """Return probability-like scores for ROC-AUC computation."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]

    decision_scores = model.decision_function(x_test)
    if decision_scores.ndim > 1:
        decision_scores = decision_scores[:, 0]
    return decision_scores


def evaluate_model(name: str, pipeline: Pipeline, x_train, y_train, x_test, y_test) -> dict:
    """Train a model and calculate all requested metrics."""
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    scores = get_probability_scores(pipeline, x_test)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, scores),
        "Pipeline": pipeline,
        "Predictions": predictions,
    }


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, file_path: Path) -> None:
    """Save a confusion matrix image for the selected model."""
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])

    threshold = matrix.max() / 2.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            plt.text(
                col_index,
                row_index,
                format(matrix[row_index, col_index], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row_index, col_index] > threshold else "black",
            )

    plt.tight_layout()
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_random_forest_feature_importance(pipeline: Pipeline, file_path: Path) -> None:
    """Save the top feature importances from the trained Random Forest model."""
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_frame = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances,
        }
    ).sort_values(by="Importance", ascending=False)

    top_features = importance_frame.head(15).iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"], color="#2a6fdb")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Load the dataset.
    data = load_data(DATA_PATH)
    target_column = detect_target_column(data)

    # Separate features and target.
    data = data.dropna(subset=[target_column]).copy()
    y = encode_target(data[target_column])
    X = data.loc[y.index].drop(columns=[target_column])

    # Split the data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y if y.value_counts().min() >= 2 else None,
    )

    # Prepare preprocessing for numeric and categorical columns.
    preprocessor = build_preprocessor(x_train)

    # Define the classification models.
    models = {
        "Logistic Regression": build_pipeline(
            preprocessor,
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "SVM": build_pipeline(
            preprocessor,
            SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        ),
        "Random Forest": build_pipeline(
            preprocessor,
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        ),
        "XGBoost": build_pipeline(
            preprocessor,
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                use_label_encoder=False,
            ),
        ),
    }

    # Train and evaluate every model.
    results = []
    random_forest_result = None
    for model_name, pipeline in models.items():
        result = evaluate_model(model_name, pipeline, x_train, y_train, x_test, y_test)
        results.append(result)
        if model_name == "Random Forest":
            random_forest_result = result

    comparison_table = pd.DataFrame(results)[
        ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ]

    # Print the model comparison table.
    print("\nDisease Prediction Model Comparison")
    print("=" * 45)
    print(comparison_table.round(4).to_string(index=False))

    # Use the Random Forest model to generate the confusion matrix.
    if random_forest_result is not None:
        plot_confusion_matrix(y_test, random_forest_result["Predictions"], CONFUSION_MATRIX_PATH)
        plot_random_forest_feature_importance(random_forest_result["Pipeline"], FEATURE_IMPORTANCE_PATH)

        # Save the trained Random Forest pipeline for later reuse.
        joblib.dump(random_forest_result["Pipeline"], MODEL_PATH)
        print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_PATH}")
        print(f"Feature importance saved to: {FEATURE_IMPORTANCE_PATH}")
        print(f"Trained model saved to: {MODEL_PATH}")
