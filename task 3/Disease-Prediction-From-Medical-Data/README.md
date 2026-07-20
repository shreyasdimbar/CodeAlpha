# Disease Prediction From Medical Data

## Project Overview

Disease Prediction From Medical Data is a beginner-friendly machine learning project that uses patient medical information to predict the possibility of disease. The project demonstrates a complete supervised learning workflow with preprocessing, model training, evaluation, and visualization.

## Objective

The objective of this project is to predict the possibility of diseases based on patient medical data using classification algorithms.

## Features

- Loads the dataset from `data/heart_disease.csv`
- Handles missing values automatically
- Encodes categorical variables when needed
- Trains multiple classification models
- Compares model performance using standard evaluation metrics
- Saves a confusion matrix image
- Saves a feature importance graph for the Random Forest model
- Saves the trained Random Forest pipeline as a reusable model file

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- scikit-learn
- XGBoost
- joblib

## Dataset Description

This project is designed around a Heart Disease dataset stored as `data/heart_disease.csv`. The dataset should contain patient medical measurements and a binary target indicating disease presence or absence.

As future extensions, the same project structure can be adapted for datasets such as the Diabetes dataset or the Breast Cancer dataset to build a broader medical prediction portfolio.

## Machine Learning Algorithms

### Logistic Regression

A simple and interpretable baseline classifier.

### Support Vector Machine (SVM)

A strong classification algorithm that can model non-linear decision boundaries.

### Random Forest

An ensemble method that improves robustness by combining multiple decision trees.

### XGBoost Classifier

A gradient boosting model that often performs strongly on structured medical data.

## Data Preprocessing

- Missing values are handled with median imputation for numeric columns and most-frequent imputation for categorical columns.
- Numeric features are scaled for models that benefit from normalization.
- Categorical features are one-hot encoded so they can be used by machine learning algorithms.
- The dataset is split into training and testing sets for fair evaluation.

## Model Evaluation

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The script prints a comparison table so the model performance can be reviewed easily.

## Folder Structure

```text
Disease-Prediction-From-Medical-Data/
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── model.py
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── project_structure.txt
```

## Installation

1. Clone or open the project folder.
2. Create and activate a virtual environment.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your dataset at `data/heart_disease.csv`.

## Usage

Run the training and evaluation script with:

```bash
python model.py
```

After execution, the project will:

- Print the comparison table in the terminal
- Save the confusion matrix image to `outputs/confusion_matrix.png`
- Save the Random Forest feature importance graph to `outputs/random_forest_feature_importance.png`
- Save the trained Random Forest pipeline to `disease_prediction_model.pkl`

## Future Improvements

- Add hyperparameter tuning for all models
- Include cross-validation for more reliable scores
- Expand the project to multi-class or multi-dataset medical prediction tasks
- Add a web app or dashboard for interactive predictions
- Add model explainability with SHAP or permutation importance

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
