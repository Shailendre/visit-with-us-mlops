#!/usr/bin/env python3
"""
Model Training Script with MLflow Experiment Tracking (MLflow 3.x / Python 3.13+)
Loads train/test from HF, tunes multiple classifiers, logs all runs to MLflow,
and registers the best model to the Hugging Face Model Hub.
Required env vars: HF_TOKEN, HF_USERNAME
Optional env vars: MLFLOW_TRACKING_URI (default: ./mlruns)
"""
import os, tempfile
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import OneHotEncoder, OrdinalEncoder
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from xgboost                 import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score)
from huggingface_hub         import HfApi, login

# ─── Configuration ────────────────────────────────────────────────────────────
HF_TOKEN     = os.environ["HF_TOKEN"]
HF_USERNAME  = os.environ.get("HF_USERNAME", "ssingh94")
DATASET_REPO = f"{HF_USERNAME}/tourism-dataset"
MODEL_REPO   = f"{HF_USERNAME}/tourism-model"
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
EXPERIMENT   = "Tourism_Package_Prediction"
RANDOM_STATE = 42
TARGET       = "ProdTaken"

# ─── Feature Definitions ──────────────────────────────────────────────────────
CAT_FEATURES = ["TypeofContact","Occupation","Gender","ProductPitched","MaritalStatus"]
ORD_FEATURES = ["Designation"]
ORD_CATS     = [["Executive","Manager","Senior Manager","AVP","VP"]]
NUM_FEATURES = [
    "Age","CityTier","DurationOfPitch","NumberOfPersonVisiting",
    "NumberOfFollowups","PreferredPropertyStar","NumberOfTrips",
    "Passport","PitchSatisfactionScore","OwnCar",
    "NumberOfChildrenVisiting","MonthlyIncome",
]

# ─── Experiments: (name, estimator, param_grid) ───────────────────────────────
EXPERIMENTS = [
    (
        "Decision Tree",
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"max_depth":[3,5,7,None], "min_samples_leaf":[10,20,30]},
    ),
    (
        "Random Forest",
        RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
        {"n_estimators":[100,200], "max_depth":[5,10,None], "min_samples_leaf":[5,10]},
    ),
    (
        "Gradient Boosting",
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.05,0.1]},
    ),
    (
        "XGBoost",
        XGBClassifier(scale_pos_weight=4, eval_metric="logloss", random_state=RANDOM_STATE),
        {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.05,0.1]},
    ),
]


def load_data():
    base = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main"
    train = pd.read_csv(f"{base}/train.csv")
    test  = pd.read_csv(f"{base}/test.csv")
    return (
        train.drop(TARGET, axis=1), test.drop(TARGET, axis=1),
        train[TARGET],              test[TARGET],
    )


def build_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), CAT_FEATURES),
        ("ord", OrdinalEncoder(categories=ORD_CATS),                         ORD_FEATURES),
        ("num", "passthrough",                                                NUM_FEATURES),
    ], remainder="drop")


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall"   : recall_score(y_test, y_pred, zero_division=0),
        "f1_score" : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"  : roc_auc_score(y_test, y_prob),
    }


def run_experiment(name, estimator, param_grid, preprocessor,
                   X_train, X_test, y_train, y_test):
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    prefixed = {f"classifier__{k}": v for k, v in param_grid.items()}

    with mlflow.start_run(run_name=name):
        gs = GridSearchCV(pipeline, prefixed, cv=5, scoring="f1", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        best    = gs.best_estimator_
        metrics = evaluate(best, X_test, y_test)

        mlflow.log_params({k.replace("classifier__", ""): v
                           for k, v in gs.best_params_.items()})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best, "model")

        print(f"{name:26s} | F1={metrics['f1_score']:.4f} | "
              f"ROC-AUC={metrics['roc_auc']:.4f} | Recall={metrics['recall']:.4f}")
        return best, metrics


def register_model(model) -> None:
    login(token=HF_TOKEN)
    api = HfApi()
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pkl")
        joblib.dump(model, path)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo="model.pkl",
            repo_id=MODEL_REPO,
            repo_type="model",
        )
    print(f"Best model registered → {MODEL_REPO}")


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_train, X_test, y_train, y_test = load_data()
    preprocessor = build_preprocessor()

    results = []
    for name, estimator, params in EXPERIMENTS:
        model, metrics = run_experiment(
            name, estimator, params, preprocessor,
            X_train, X_test, y_train, y_test,
        )
        results.append((name, model, metrics))

    best_name, best_model, best_metrics = max(results, key=lambda x: x[2]["f1_score"])
    print(f"\nBest: {best_name}  F1={best_metrics['f1_score']:.4f}")
    register_model(best_model)


if __name__ == "__main__":
    main()
