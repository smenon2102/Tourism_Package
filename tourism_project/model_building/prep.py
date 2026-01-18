
# -----------------------------
# Data Preparation Script
# Tourism Package Prediction
# -----------------------------

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# -----------------------------
# Hugging Face configuration
# -----------------------------
repo_id = "avatar2102/tourism-package-dataset"
repo_type = "dataset"

DATASET_PATH = f"hf://datasets/{repo_id}/tourism.csv"

token = os.getenv("TPACKAGE_TOKEN")
if token is None:
    raise ValueError("TPACKAGE_TOKEN environment variable not set")

api = HfApi(token=token)

# -----------------------------
# Load dataset from HF
# -----------------------------
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully")
print("Shape:", df.shape)

# -----------------------------
# Data Cleaning (explicit)
# -----------------------------

# Drop unnecessary / ID columns
df = df.drop(columns=[
    "CustomerID"
])

# Target variable
target = "ProdTaken"

# -----------------------------
# Handle missing values (explicit)
# -----------------------------

# Numerical columns
df["Age"] = df["Age"].fillna(df["Age"].median())
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
df["DurationOfPitch"] = df["DurationOfPitch"].fillna(df["DurationOfPitch"].median())
df["NumberOfTrips"] = df["NumberOfTrips"].fillna(df["NumberOfTrips"].median())
df["NumberOfFollowups"] = df["NumberOfFollowups"].fillna(df["NumberOfFollowups"].median())
df["PitchSatisfactionScore"] = df["PitchSatisfactionScore"].fillna(df["PitchSatisfactionScore"].median())
df["PreferredPropertyStar"] = df["PreferredPropertyStar"].fillna(df["PreferredPropertyStar"].median())
df["NumberOfChildrenVisiting"] = df["NumberOfChildrenVisiting"].fillna(0)

# Categorical columns
df["TypeofContact"] = df["TypeofContact"].fillna(df["TypeofContact"].mode()[0])
df["Occupation"] = df["Occupation"].fillna(df["Occupation"].mode()[0])
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["MaritalStatus"] = df["MaritalStatus"].fillna(df["MaritalStatus"].mode()[0])
df["ProductPitched"] = df["ProductPitched"].fillna(df["ProductPitched"].mode()[0])
df["Designation"] = df["Designation"].fillna(df["Designation"].mode()[0])

# -----------------------------
# Split features & target
# -----------------------------
X = df.drop(columns=[target])
y = df[target].astype(int)

# -----------------------------
# Train-test split
# -----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train/Test split completed")

# -----------------------------
# Save locally
# -----------------------------
os.makedirs("tourism_project/model_building/data_splits", exist_ok=True)

Xtrain_path = "tourism_project/model_building/data_splits/Xtrain.csv"
Xtest_path  = "tourism_project/model_building/data_splits/Xtest.csv"
ytrain_path = "tourism_project/model_building/data_splits/ytrain.csv"
ytest_path  = "tourism_project/model_building/data_splits/ytest.csv"

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("Split files saved locally")

# -----------------------------
# Upload splits to HF dataset
# -----------------------------
api.upload_file(
    path_or_fileobj=Xtrain_path,
    path_in_repo="data_splits/Xtrain.csv",
    repo_id=repo_id,
    repo_type=repo_type,
)

api.upload_file(
    path_or_fileobj=Xtest_path,
    path_in_repo="data_splits/Xtest.csv",
    repo_id=repo_id,
    repo_type=repo_type,
)

api.upload_file(
    path_or_fileobj=ytrain_path,
    path_in_repo="data_splits/ytrain.csv",
    repo_id=repo_id,
    repo_type=repo_type,
)

api.upload_file(
    path_or_fileobj=ytest_path,
    path_in_repo="data_splits/ytest.csv",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Train-test data uploaded to Hugging Face dataset repo")
