# ---------------------------------------------
# Experimentation & Tracking (Production)
# Tourism Package Prediction - train.py
# ---------------------------------------------

import os
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import joblib
import mlflow

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub.utils import HfHubHTTPError

# -----------------------------
# MLflow settings (Production)
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Package_Training_Experiment")

# -----------------------------
# HF configuration
# -----------------------------
DATASET_REPO_ID = "avatar2102/tourism-package-dataset"  # HF dataset repo
MODEL_REPO_ID   = "avatar2102/tourism-package-model"    # HF model repo to store joblib
REPO_TYPE_DATASET = "dataset"
REPO_TYPE_MODEL   = "model"

token = os.getenv("TPACKAGE_TOKEN")
if token is None:
    raise ValueError("TPACKAGE_TOKEN environment variable not set")

api = HfApi(token=token)

# -----------------------------
# Load train/test splits from HF dataset
# -----------------------------
Xtrain_path = f"hf://datasets/{DATASET_REPO_ID}/data_splits/Xtrain.csv"
Xtest_path  = f"hf://datasets/{DATASET_REPO_ID}/data_splits/Xtest.csv"
ytrain_path = f"hf://datasets/{DATASET_REPO_ID}/data_splits/ytrain.csv"
ytest_path  = f"hf://datasets/{DATASET_REPO_ID}/data_splits/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze("columns")
ytest  = pd.read_csv(ytest_path).squeeze("columns")

print("Loaded splits:")
print("Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape)
print("ytrain:", ytrain.shape, "ytest:", ytest.shape)

# -----------------------------
# Feature lists (explicit)
# -----------------------------
numeric_features = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch",
]

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "ProductPitched",
    "Designation",
]

# -----------------------------
# Class imbalance weight for XGBoost
# -----------------------------
neg = (ytrain == 0).sum()
pos = (ytrain == 1).sum()
class_weight = (neg / pos) if pos != 0 else 1.0
print("scale_pos_weight:", class_weight)

# -----------------------------
# Preprocessor + Model
# -----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# -----------------------------
# Hyperparameter grid (production-style: broader than dev)
# -----------------------------
param_grid = {
    "xgbclassifier__n_estimators": [100, 200, 300],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__colsample_bytree": [0.8, 1.0],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__reg_lambda": [1.0, 2.0],
}

# -----------------------------
# Train + Track
# -----------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1"
    )
    grid_search.fit(Xtrain, ytrain)

    # Log each combination as nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", float(mean_score))
            mlflow.log_metric("std_test_score", float(std_score))

    # Log best params + best CV score in main run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", float(grid_search.best_score_))

    best_model = grid_search.best_estimator_

    # Evaluate with threshold like your sample
    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"],
    })

    # Save model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log as MLflow artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

# -----------------------------
# Upload best model to HF Model Hub
# -----------------------------
try:
    api.repo_info(repo_id=MODEL_REPO_ID, repo_type=REPO_TYPE_MODEL)
    print(f"Model repo '{MODEL_REPO_ID}' already exists. Using it.")
except (RepositoryNotFoundError, HfHubHTTPError):
    print(f"Model repo '{MODEL_REPO_ID}' not found. Creating new repo...")
    api.create_repo(repo_id=MODEL_REPO_ID, repo_type=REPO_TYPE_MODEL, private=False)
    print(f"Model repo '{MODEL_REPO_ID}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=MODEL_REPO_ID,
    repo_type=REPO_TYPE_MODEL,
)

print(f"Uploaded model to Hugging Face: {MODEL_REPO_ID}")
