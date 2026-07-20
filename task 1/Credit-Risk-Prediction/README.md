# Credit Risk Prediction

## Project Overview

This project is a beginner-friendly machine learning classification pipeline that predicts an individual's creditworthiness from historical financial data. It uses a structured preprocessing workflow and compares three popular supervised learning algorithms.

## Objective

The goal is to classify applicants as creditworthy or risky based on patterns in the dataset. The project demonstrates a complete end-to-end machine learning workflow, from data loading and preprocessing to model training, evaluation, and visualization.

## Features

- Loads the dataset from `data/german_credit_data.csv`
- Handles missing values with automated imputation
- Encodes categorical variables for machine learning models
- Splits the dataset into training and testing sets
- Trains three classification algorithms
- Evaluates each model with standard classification metrics
- Saves a feature importance chart for interpretability

## Technologies Used

- Python
- pandas
- numpy
- matplotlib
- scikit-learn

## Dataset Description

The project expects a credit risk dataset saved as `data/german_credit_data.csv`. The dataset should contain financial and demographic features used to assess credit risk. The final column is treated as the target if a standard credit target column is not detected automatically.

If you are using the commonly available German Credit dataset, place the CSV file in the `data/` directory before running the script.

## Machine Learning Algorithms

### Logistic Regression

A strong baseline classifier that is fast, interpretable, and effective for binary classification problems.

### Decision Tree

A rule-based model that captures non-linear relationships and is easy to explain.

### Random Forest

An ensemble method that combines multiple decision trees to improve robustness and predictive performance.

## Evaluation Metrics

The project prints the following metrics for every model:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

## Folder Structure

```text
Credit-Risk-Prediction/
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── model.py
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
└── project_structure.txt
```

## Installation Steps

1. Clone or open the project folder.
2. Create and activate a virtual environment.
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add the dataset file to `data/german_credit_data.csv`.

## Usage Instructions

Run the project with:

```bash
python model.py
```

After execution, the script prints evaluation metrics in the terminal and saves the feature importance plot to `outputs/feature_importance.png`.

## Future Improvements

- Add hyperparameter tuning for each model
- Perform cross-validation for a more stable performance estimate
- Add SHAP or permutation importance for deeper explainability
- Build a small web app for credit risk predictions
- Include data validation and automated tests

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
