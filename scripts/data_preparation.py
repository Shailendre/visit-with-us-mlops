#!/usr/bin/env python3
"""
Data Preparation Script
Loads raw data from HF, cleans it, splits into train/test,
saves locally, and uploads both splits back to HF.
Required env vars: HF_TOKEN, HF_USERNAME
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login

HF_TOKEN     = os.environ["HF_TOKEN"]
HF_USERNAME  = os.environ.get("HF_USERNAME", "ssingh94")
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"
TARGET       = "ProdTaken"
RANDOM_STATE = 42
TEST_SIZE    = 0.20


def load_raw_data() -> pd.DataFrame:
    url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/tourism.csv"
    df = pd.read_csv(url)
    print(f"Loaded raw data: {df.shape}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop index duplicates
    df.drop(columns=["Unnamed: 0", "CustomerID"], errors="ignore", inplace=True)

    # Fix dirty categorical values found during EDA
    df["Gender"]        = df["Gender"].replace("Fe Male", "Female")
    df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

    # Impute residual nulls (robust to future data)
    num_cols = df.select_dtypes("number").columns.tolist()
    cat_cols = df.select_dtypes("object").columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    print(f"After cleaning: {df.shape}  |  nulls: {df.isnull().sum().sum()}")
    return df


def split_and_save(df: pd.DataFrame) -> tuple[str, str]:
    X, y = df.drop(TARGET, axis=1), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv( "data/test.csv",  index=False)
    print(f"Train: {train_df.shape}  |  Test: {test_df.shape}")
    return "data/train.csv", "data/test.csv"


def upload_splits(train_path: str, test_path: str) -> None:
    login(token=HF_TOKEN)
    api = HfApi()
    for local, remote in [(train_path, "train.csv"), (test_path, "test.csv")]:
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=remote,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
    print(f"Uploaded train/test splits to {DATASET_REPO}")


def main() -> None:
    df = load_raw_data()
    df = clean(df)
    train_path, test_path = split_and_save(df)
    upload_splits(train_path, test_path)


if __name__ == "__main__":
    main()
