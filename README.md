# Adult Census Income Prediction

A comprehensive end-to-end Machine Learning pipeline to predict whether an individual's annual income exceeds $50K using the UCI Adult Census Income dataset.

---

## Project Overview
- **Objective:** Classify income level (`<=50K` vs `>50K`) based on demographic and employment attributes.
- **Data Source:** UCI Machine Learning Repository (Adult Dataset).
- **Core Techniques:** Exploratory Data Analysis (EDA), Skewness Handling, Pipeline Construction with `ColumnTransformer`, Stratified K-Fold Cross-Validation, Regularization (L1), Hyperparameter Tuning via `GridSearchCV`, and Post-prediction Error Analysis.

---

## Data Preprocessing & Pipeline
- **Missing Value Imputation:**
  - Categorical variables (`workclass`, `occupation`, `native-country`): Imputed using **Most Frequent** strategy.
  - Numerical variables: Imputed using **Median** strategy.
- **Feature Transformation:**
  - One-Hot Encoding on nominal features (`sparse_output=False`, `handle_unknown='ignore'`).
  - Standard Scaling on numerical features (`StandardScaler`).
- **Data Leakage Prevention:** All preprocessing transformers fit strictly within cross-validation training folds.

---

##  Model Comparison & Benchmark

Evaluated using 5-Fold Stratified Cross-Validation:

| Model | Precision | Recall | F1-Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.7309 | 0.5891 | 0.6523 | 0.9042 |
| **KNN (n_neighbors=5)** | 0.6670 | 0.5997 | 0.6314 | 0.8574 |
| **Random Forest (Initial)** | 0.7287 | 0.6189 | 0.6693 | 0.9006 |

---

## Final Model & Test Evaluation
The **Random Forest Classifier** was selected and fine-tuned using `GridSearchCV`:
- **Best Parameters:** `n_estimators=400`, `max_depth=20`
- **Test Set Performance:**
  - **Accuracy:** 87%
  - **F1-Score (>50K):** 0.70
  - **ROC-AUC Score:** 0.9222
