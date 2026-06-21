#!/usr/bin/env python3
"""
Hosting Script: Push all deployment files to a Hugging Face Streamlit Space.
Required env vars: HF_TOKEN, HF_USERNAME
"""
import os
from huggingface_hub import HfApi, login

HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ.get("HF_USERNAME", "ssingh94")
SPACE_REPO  = f"{HF_USERNAME}/Great-Learning-Visit-With-Us"

DEPLOYMENT_FILES = {
    "deployment/app.py":           "app.py",
    "deployment/requirements.txt": "requirements.txt",
    "deployment/Dockerfile":       "Dockerfile",
}


def deploy() -> None:
    login(token=HF_TOKEN)
    api = HfApi()

    api.create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print(f"Space ready: {SPACE_REPO}")

    for local_path, repo_path in DEPLOYMENT_FILES.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=SPACE_REPO,
            repo_type="space",
        )
        print(f"  Uploaded {local_path} → {repo_path}")

    print(f"\nApp live at: https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    deploy()
