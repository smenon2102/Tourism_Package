from huggingface_hub import HfApi
import os

# -----------------------------
# Hugging Face configuration
# -----------------------------
SPACE_REPO_ID = "avatar2102/Tourism-Package"   # HF Space (Docker + Streamlit)
REPO_TYPE = "space"

token = os.getenv("TPACKAGE_TOKEN")
if token is None:
    raise ValueError("TPACKAGE_TOKEN environment variable not set")

api = HfApi(token=token)

# -----------------------------
# Upload deployment folder to HF Space
# -----------------------------
api.upload_folder(
    folder_path="tourism_project/deployment",  # local deployment files
    repo_id=SPACE_REPO_ID,                      # HF Space repo
    repo_type=REPO_TYPE,
    path_in_repo=""                             # root of the Space
)

print(f"Deployment files uploaded successfully to HF Space: {SPACE_REPO_ID}")
