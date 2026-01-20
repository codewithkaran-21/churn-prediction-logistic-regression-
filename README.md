# Churn Prediction — Logistic Regression

A reproducible project that predicts customer churn using logistic regression. This repository contains data processing, exploratory data analysis (EDA), model training, evaluation, and deliverables (trained model and reports). It demonstrates a complete ML workflow: data ingestion → cleaning/feature engineering → model training → evaluation → deployment artifacts.

## Table of contents
- [Project Summary](#project-summary)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Data](#data)
- [Modeling approach](#modeling-approach)
- [How to run](#how-to-run)
- [Evaluation & metrics](#evaluation--metrics)
- [Reproducibility](#reproducibility)
- [Results](#results)
- [Development & contribution](#development--contribution)
- [Suggested improvements](#suggested-improvements)
- [License](#license)
- [Contact](#contact)

## Project summary
This project trains a logistic regression classifier to predict whether customers will churn. It includes:
- EDA and baseline performance.
- Data preprocessing and feature engineering (missing values, encodings, scaling).
- Model training with cross-validation and hyperparameter tuning (regularization).
- Evaluation with confusion matrix, ROC-AUC, precision/recall, and calibration checks.
- Scripts/notebooks to reproduce the analysis and save the trained model.

## Repository structure
(If any files are missing, adapt the structure to the repo's actual layout.)
- data/
  - raw/                  — raw dataset(s) (CSV)
  - processed/            — cleaned / feature-engineered data
- notebooks/
  - 01_eda.ipynb
  - 02_feature_engineering.ipynb
  - 03_model_training.ipynb
- src/
  - data_processing.py    — loading, cleaning, feature engineering
  - features.py           — feature transformations/pipelines
  - train.py              — training/evaluation entrypoint
  - evaluate.py           — evaluation utilities and reports generation
  - predict.py            — inference script
  - utils.py              — helper functions
- models/
  - logistic_model.joblib — trained model artifact
- reports/
  - figures/              — saved plots (ROC, confusion matrix, etc.)
  - performance.md
- requirements.txt
- README.md
- LICENSE

## Getting started

1. Clone the repository:
   git clone https://github.com/codewithkaran-21/churn-prediction-logistic-regression-.git
   cd churn-prediction-logistic-regression-

2. Create and activate an environment (recommended):
   # Using venv
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows

   # Or using conda
   conda env create -f environment.yml
   conda activate churn-env

3. Install dependencies:
   pip install -r requirements.txt

Common packages you should expect:
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- joblib
- jupyterlab / notebook

(If a requirements.txt is missing, run `pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyterlab`.)

## Data
- Expected input: a CSV with customer records and target column `churn` (binary: 0/1 or Yes/No).
- Typical features: demographics, account info, service usage, tenure, contract type, monthly charges.
- Data preprocessing steps (in src/data_processing.py):
  - Drop/flag duplicates
  - Handle missing values (impute median for numeric, constant or mode for categoricals)
  - Convert categorical variables (one-hot / ordinal encoding)
  - Scale numeric features (StandardScaler or MinMax)
  - Save processed dataset to `data/processed/`

## Modeling approach
- Baseline: Logistic Regression (scikit-learn)
- Pipeline components:
  - Preprocessing: ColumnTransformer with numeric and categorical pipelines
  - Feature selection or regularization (L2)
  - Cross-validation (StratifiedKFold)
- Hyperparameter tuning: GridSearchCV for `C` (inverse regularization), penalty and solver as appropriate.
- Handling class imbalance: class_weight='balanced' or resampling (SMOTE) if needed.

## How to run

From the repo root:

- Run EDA notebook:
  - Open `notebooks/01_eda.ipynb` and run cells in JupyterLab.

- Train model (script):
  python src/train.py \
    --data data/processed/train.csv \
    --output-dir models/ \
    --cv 5 \
    --random-seed 42

- Evaluate model:
  python src/evaluate.py --model models/logistic_model.joblib --test data/processed/test.csv --out reports/

- Predict for new data:
  python src/predict.py --model models/logistic_model.joblib --input data/processed/new_customers.csv --output predictions.csv

(Replace file paths with actual file names in your repo.)

## Evaluation & metrics
Key evaluation metrics:
- Accuracy (not sufficient on imbalanced datasets)
- Precision and Recall
- F1 score
- ROC-AUC (primary ranking metric)
- Confusion matrix to inspect false positives/negatives
- Precision-Recall curve if classes are imbalanced
- Calibration plot to check probability estimates

Recommended reporting:
- Train/validation/test splits (stratified)
- Cross-validated ROC-AUC with mean ± std
- Confusion matrix at business-relevant threshold(s)
- Feature coefficients (explainability): show top positive/negative coefficients with confidence intervals

## Reproducibility
- Use a fixed random seed (`random_state=42`) everywhere.
- Save preprocessing pipeline and trained model (joblib.dump).
- Document package versions in requirements.txt or environment.yml.
- Optionally add a Dockerfile for a reproducible runtime.

Example to save model and pipeline:
from joblib import dump
dump(pipeline, "models/logistic_model.joblib")

Load and predict:
from joblib import load
model = load("models/logistic_model.joblib")
preds = model.predict(X_new)

## Results
(Include summary once you run training. Example placeholders:)
- Test ROC-AUC: 0.86
- Test F1-score: 0.62
- Important features: tenure (decr. churn), monthly_charges (incr. churn), contract_type=Month-to-month (incr. churn)

Include figures in reports/figures: ROC curve, confusion matrix, feature importance bar plot.

## Development & contribution
- Branching: feature branches named feature/<name> or fix/<issue>.
- Tests: add unit tests for preprocessing and utilities (pytest).
- CI: recommended GitHub Actions to run linting and tests on push/PR.
- To contribute:
  1. Fork repository
  2. Create a branch
  3. Add changes and tests
  4. Open a pull request

## Suggested improvements
- Try non-linear models (RandomForest, XGBoost) and compare performance.
- Use SHAP or LIME for model explainability.
- Automate hyperparameter tuning with Optuna.
- Add calibration and business-cost-based threshold optimization.
- Add a small web app (Flask/FastAPI) or a Streamlit demo for interactive predictions.

## License
This project is available under the MIT License. See LICENSE file.

## Contact
Maintainer: codewithkaran-21
Open an issue in the repo for questions or contributions.
