#!/usr/bin/env python3
"""
Data Registration Script
Uploads the raw tourism dataset to the Hugging Face Dataset Hub.
Required env vars: HF_TOKEN, HF_USERNAME
"""
import os
from huggingface_hub import HfApi, login

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ.get("HF_USERNAME", "ssingh94")
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"
DATA_FILE    = "data/tourism.csv"


def register_dataset() -> None:
    login(token=HF_TOKEN)
    api = HfApi()

    api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)
    print(f"Dataset repo ready: {DATASET_REPO}")

    api.upload_file(
        path_or_fileobj=DATA_FILE,
        path_in_repo="tourism.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded {DATA_FILE} → {DATASET_REPO}/tourism.csv")


if __name__ == "__main__":
    register_dataset()
