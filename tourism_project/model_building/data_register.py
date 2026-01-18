from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os

repo_id = "avatar2102/tourism-package-dataset"
repo_type = "dataset"

token = os.getenv("TPACKAGE_TOKEN")
if token is None:
    raise ValueError("TPACKAGE_TOKEN environment variable not set")

api = HfApi(token=token)

# Check if dataset repo exists; if not create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except (RepositoryNotFoundError, HfHubHTTPError):
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Upload local folder
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
    commit_message="Add tourism dataset"
)

print("Dataset uploaded successfully.")
