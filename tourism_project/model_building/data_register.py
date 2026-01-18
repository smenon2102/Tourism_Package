
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Dataset repository
repo_id = "avatar2102/tourism-package-dataset"
repo_type = "dataset"

# Read token safely
token = os.getenv("TPACKAGE_TOKEN")
if token is None:
    raise ValueError("TPACKAGE_TOKEN environment variable not set")

# Initialize API client
api = HfApi(token=token)

# Step 1: Check if dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Step 2: Upload local data folder
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Add tourism dataset"
)

print("Dataset uploaded successfully.")
