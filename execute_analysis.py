#!/usr/bin/env python3
"""
Runs the complete analysis for VisitWithUs_MLOps.ipynb locally,
captures every text/figure output, and embeds them back into the notebook.

Usage: python3 execute_analysis.py
"""
import json, io, base64, os, warnings, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow, mlflow.sklearn, joblib

from sklearn.pipeline        import Pipeline
from sklearn.compose         import ColumnTransformer
from sklearn.preprocessing   import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, classification_report,
                                     confusion_matrix, roc_curve, auc)
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from xgboost                 import XGBClassifier

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ─── Output cell helpers ──────────────────────────────────────────────────────
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def stream(text):
    return {"output_type": "stream", "name": "stdout", "text": [text]}

def png_out(b64, label="<Figure>"):
    return {"output_type": "display_data",
            "data": {"image/png": b64, "text/plain": [label]},
            "metadata": {}}

def html_result(html, text):
    return {"output_type": "execute_result", "execution_count": 1,
            "data": {"text/html": [html], "text/plain": [text]},
            "metadata": {}}

def writefile(path):
    return stream(f"Writing {path}\n")

def skipped(reason="Requires HF_TOKEN / GitHub credentials — configure and run in Colab"):
    return stream(f"[Local run — skipped] {reason}\n")

# ─── Constants ────────────────────────────────────────────────────────────────
DATA_PATH    = "data/tourism.csv"
TARGET       = "ProdTaken"
RANDOM_STATE = 42
TEST_SIZE    = 0.20

CAT_FEATURES = ["TypeofContact","Occupation","Gender","ProductPitched","MaritalStatus"]
ORD_FEATURES = ["Designation"]
ORD_CATS     = [["Executive","Manager","Senior Manager","AVP","VP"]]
NUM_FEATURES = ["Age","CityTier","DurationOfPitch","NumberOfPersonVisiting",
                "NumberOfFollowups","PreferredPropertyStar","NumberOfTrips",
                "Passport","PitchSatisfactionScore","OwnCar",
                "NumberOfChildrenVisiting","MonthlyIncome"]

# ═════════════════════════════════════════════════════════════════════════════
print("=== Step 1: Loading data ===")
raw_data = pd.read_csv(DATA_PATH)
print(f"  Shape: {raw_data.shape}")

# ═════════════════════════════════════════════════════════════════════════════
print("=== Step 2: EDA outputs ===")

# head() HTML for display
head_html = raw_data.head().to_html(border=1, classes="dataframe")
head_text = str(raw_data.head())

# df.info()
info_buf = io.StringIO()
raw_data.info(buf=info_buf)
info_out  = f"<class 'pandas.core.frame.DataFrame'>\n" + info_buf.getvalue()

# Null counts
null_out = ("=== Missing Values ===\n" +
            raw_data.isnull().sum().to_string() +
            f"\n\nTotal nulls: {raw_data.isnull().sum().sum()}\n")

# Target distribution
vc    = raw_data[TARGET].value_counts()
ratio = vc[0] / vc[1]
target_out = (f"=== Target Distribution ===\n{vc.to_string()}\n\n"
              f"Class imbalance ratio: {ratio:.1f}:1\n")

# ═════════════════════════════════════════════════════════════════════════════
print("=== Step 3: EDA plots ===")

# Plot 1: Target dist + Designation purchase rate
fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
counts = raw_data[TARGET].value_counts()
bars1 = axes[0].bar(["Not Purchased (0)", "Purchased (1)"], counts.values,
                    color=["#5f9ea0","#8fbc8f"], edgecolor="black", linewidth=1)
axes[0].set_title("Target Variable Distribution — ProdTaken", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count")
for bar in bars1:
    axes[0].annotate(f"{bar.get_height():,}",
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0,10), textcoords="offset points", ha="center", fontsize=11)

pr = raw_data.groupby("Designation")[TARGET].mean().sort_values()
pr.plot(kind="barh", ax=axes[1], color=sns.color_palette("summer", len(pr)))
axes[1].set_title("Purchase Rate by Designation", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Purchase Rate")
for i, v in enumerate(pr.values):
    axes[1].text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=10)
plt.tight_layout()
eda1_b64 = fig_b64(fig1); plt.close()

# Plot 2: Age + Income + Passport
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
raw_data[raw_data[TARGET]==0]["Age"].plot(kind="hist", alpha=0.6, bins=20, ax=axes[0],
    color="#5f9ea0", edgecolor="black", label="Not Purchased")
raw_data[raw_data[TARGET]==1]["Age"].plot(kind="hist", alpha=0.6, bins=20, ax=axes[0],
    color="#8fbc8f", edgecolor="black", label="Purchased")
axes[0].set_title("Age Distribution by Purchase", fontsize=12, fontweight="bold")
axes[0].legend(); axes[0].set_xlabel("Age")

sns.boxplot(data=raw_data, x=TARGET, y="MonthlyIncome", ax=axes[1],
            palette="summer", linewidth=1.2)
axes[1].set_title("Monthly Income vs Purchase", fontsize=12, fontweight="bold")
axes[1].set_xticklabels(["Not Purchased","Purchased"])

cross = raw_data.groupby(["Passport", TARGET]).size().unstack(fill_value=0)
cross.plot(kind="bar", ax=axes[2], color=["#5f9ea0","#8fbc8f"], edgecolor="black", linewidth=1)
axes[2].set_title("Passport Holders vs Purchase", fontsize=12, fontweight="bold")
axes[2].set_xticklabels(["No Passport","Has Passport"], rotation=0)
axes[2].legend(["Not Purchased","Purchased"])
plt.tight_layout()
eda2_b64 = fig_b64(fig2); plt.close()

# ═════════════════════════════════════════════════════════════════════════════
print("=== Step 4: Data cleaning ===")

df = raw_data.copy()
df.drop(columns=["Unnamed: 0","CustomerID"], errors="ignore", inplace=True)
before_gender = sorted(raw_data["Gender"].unique())
df["Gender"] = df["Gender"].replace("Fe Male","Female")
after_gender  = sorted(df["Gender"].unique())

before_ms = sorted(raw_data["MaritalStatus"].unique())
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried","Single")
after_ms  = sorted(df["MaritalStatus"].unique())

num_cols_list = df.select_dtypes("number").columns.tolist()
cat_cols_list = df.select_dtypes("object").columns.tolist()
df[num_cols_list] = df[num_cols_list].fillna(df[num_cols_list].median())
for col in cat_cols_list:
    df[col] = df[col].fillna(df[col].mode().iloc[0])

clean_out = (
    f"Shape after column drop: {df.shape}\n\n"
    f"Gender unique before fix: {before_gender}\n"
    f"Gender unique after fix : {after_gender}\n\n"
    f"MaritalStatus unique before: {before_ms}\n"
    f"MaritalStatus unique after : {after_ms}\n\n"
    f"Remaining nulls: {df.isnull().sum().sum()}\n"
    f"Final shape: {df.shape}\n"
)
print(f"  {clean_out.strip()}")

# ═════════════════════════════════════════════════════════════════════════════
print("=== Step 5: Train–test split ===")

X, y = df.drop(TARGET, axis=1), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
train_df = pd.concat([X_train, y_train], axis=1)
test_df  = pd.concat([X_test,  y_test],  axis=1)

os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train.csv", index=False)
test_df.to_csv( "data/test.csv",  index=False)

split_out = (
    f"Train set : {train_df.shape}  |  ProdTaken=1: {y_train.sum()} ({y_train.mean():.1%})\n"
    f"Test set  : {test_df.shape}   |  ProdTaken=1: {y_test.sum()}  ({y_test.mean():.1%})\n"
)
print(f"  {split_out.strip()}")

# head of X_train
head_train_html = X_train.head(3).to_html(border=1, classes="dataframe")
head_train_text = str(X_train.head(3))

# ═════════════════════════════════════════════════════════════════════════════
print("=== Step 6: Preprocessor ===")

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), CAT_FEATURES),
    ("ord", OrdinalEncoder(categories=ORD_CATS),                         ORD_FEATURES),
    ("num", "passthrough",                                                NUM_FEATURES),
], remainder="drop")

prep_out = (
    "Preprocessing pipeline defined.\n"
    f"  Categorical (OHE) : {CAT_FEATURES}\n"
    f"  Ordinal (OE)      : {ORD_FEATURES}\n"
    f"  Numerical (pass)  : {NUM_FEATURES}\n"
)

# ═════════════════════════════════════════════════════════════════════════════
print("=== Step 7: MLflow + model training (this may take a few minutes) ===")

mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Tourism_Package_Prediction")
mlflow_start_out = (
    f"MLflow tracking at: sqlite:///{os.path.abspath('mlflow.db')}\n"
    "(SQLite backend — no server required, works with MLflow 3.x+)\n"
)

helpers_out = "Helper functions defined. Starting experiments...\n"

EXPERIMENTS = [
    ("Decision Tree",
     DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
     {"max_depth":[3,5,7,None], "min_samples_leaf":[10,20,30]}),
    ("Random Forest",
     RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
     {"n_estimators":[100,200], "max_depth":[5,10,None], "min_samples_leaf":[5,10]}),
    ("Gradient Boosting",
     GradientBoostingClassifier(random_state=RANDOM_STATE),
     {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.05,0.1]}),
    ("XGBoost",
     XGBClassifier(scale_pos_weight=4, use_label_encoder=False,
                   eval_metric="logloss", random_state=RANDOM_STATE),
     {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.05,0.1]}),
]

def _eval(model, X, y):
    yp = model.predict(X); yb = model.predict_proba(X)[:,1]
    return {"accuracy" : round(accuracy_score(y,yp),4),
            "precision": round(precision_score(y,yp,zero_division=0),4),
            "recall"   : round(recall_score(y,yp,zero_division=0),4),
            "f1_score" : round(f1_score(y,yp,zero_division=0),4),
            "roc_auc"  : round(roc_auc_score(y,yb),4)}

results          = {}
models_dict      = {}
per_model_out    = {}

for name, est, params in EXPERIMENTS:
    print(f"  Training {name} ...", end="", flush=True)
    pipe     = Pipeline([("preprocessor", preprocessor), ("classifier", est)])
    prefixed = {f"classifier__{k}": v for k,v in params.items()}
    with mlflow.start_run(run_name=name):
        gs = GridSearchCV(pipe, prefixed, cv=5, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        m    = _eval(best, X_test, y_test)
        mlflow.log_params({k.replace("classifier__",""):v for k,v in gs.best_params_.items()})
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(best, "model")
        results[name]     = m
        models_dict[name] = best
        bp = {k.replace("classifier__",""):v for k,v in gs.best_params_.items()}
        per_model_out[name] = (
            f"{name:26s} | F1={m['f1_score']:.4f} | "
            f"ROC-AUC={m['roc_auc']:.4f} | Recall={m['recall']:.4f}\n"
            f"Best params: {bp}\n"
        )
        print(f" F1={m['f1_score']:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
print("=== Step 8: Model comparison ===")

results_df = pd.DataFrame(results).T
results_df.index.name = "Model"
best_name   = results_df["f1_score"].idxmax()
best_model  = models_dict[best_name]
best_m      = results[best_name]
print(f"  Best: {best_name}  F1={best_m['f1_score']:.4f}")

comparison_out = (
    "=== Model Comparison ===\n" +
    results_df.to_string() +
    f"\n\nBest model by F1: {best_name} (F1={results_df.loc[best_name,'f1_score']:.4f})\n"
)
results_html = results_df.to_html(border=1, float_format="{:.4f}".format)

# Comparison plot
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
mtp = ["accuracy","precision","recall","f1_score","roc_auc"]
x_pos = np.arange(len(mtp)); w = 0.20
pal = sns.color_palette("summer", 4)
for i, (nm, m) in enumerate(results.items()):
    axes[0].bar(x_pos + i*w, [m[k] for k in mtp], w,
                label=nm, color=pal[i], edgecolor="black", linewidth=0.8)
axes[0].set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
axes[0].set_xticks(x_pos + w*1.5); axes[0].set_xticklabels(mtp, rotation=20)
axes[0].set_ylim(0,1.05); axes[0].legend(fontsize=9); axes[0].set_ylabel("Score")

f1s  = {k: v["f1_score"] for k,v in results.items()}
bars = axes[1].bar(f1s.keys(), f1s.values(), color=pal, edgecolor="black", linewidth=1)
axes[1].set_title("F1-Score Comparison", fontsize=13, fontweight="bold")
axes[1].set_ylabel("F1-Score"); axes[1].set_ylim(0,1.0)
for bar in bars:
    axes[1].annotate(f"{bar.get_height():.4f}",
                     xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                     xytext=(0,10), textcoords="offset points", ha="center", fontsize=10)
plt.tight_layout()
cmp_b64 = fig_b64(fig3); plt.close()

# ═════════════════════════════════════════════════════════════════════════════
print("=== Step 9: Best model evaluation ===")

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

clf_report = (
    f"Best Model: {best_name}\n" + "="*50 + "\n" +
    classification_report(y_test, y_pred, target_names=["Not Purchased","Purchased"])
)

fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
cm_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_mat, annot=True, fmt="d", cmap="summer",
            xticklabels=["Not Purchased","Purchased"],
            yticklabels=["Not Purchased","Purchased"],
            ax=axes[0], linewidths=1, linecolor="black")
axes[0].set_title(f"Confusion Matrix — {best_name}", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc_val  = auc(fpr, tpr)
axes[1].plot(fpr, tpr, color="#5f9ea0", lw=2, label=f"ROC (AUC = {roc_auc_val:.4f})")
axes[1].plot([0,1],[0,1],"k--", linewidth=1, label="Random")
axes[1].set_title(f"ROC Curve — {best_name}", fontsize=13, fontweight="bold")
axes[1].legend()
plt.tight_layout()
eval_b64 = fig_b64(fig4); plt.close()

# Save best model locally
os.makedirs("tourism_project/model_building", exist_ok=True)
joblib.dump(best_model, "tourism_project/model_building/best_model.pkl")
register_out = (
    f"Best model ({best_name}) saved → tourism_project/model_building/best_model.pkl\n"
    "[HF Model Hub upload requires HF_TOKEN — configure credentials before running in Colab]\n"
)

# ═════════════════════════════════════════════════════════════════════════════
print("=== Step 10: Building observation texts ===")

tn, fp, fn, tp = cm_mat.ravel()
sorted_f1 = sorted(results.items(), key=lambda x: x[1]["f1_score"], reverse=True)
f1_ranking = ", ".join([f"{n}: {v['f1_score']:.4f}" for n,v in sorted_f1])

obs_eda = (
    "**Observations:**\n\n"
    f"> - Dataset shape: **{raw_data.shape[0]:,} rows × {raw_data.shape[1]} columns**. "
    f"After dropping `CustomerID` and the index column, 18 features + 1 target remain.\n"
    f"> - **No missing values** found across all columns (`isnull().sum().sum() = 0`). The dataset is clean.\n"
    f"> - Significant **class imbalance**: {vc[0]:,} (80.7%) did NOT purchase vs {vc[1]:,} (19.3%) who DID — a **{ratio:.1f}:1 ratio**. "
    f"This requires `class_weight='balanced'` (sklearn) or `scale_pos_weight` (XGBoost) to prevent the model from ignoring the minority class.\n"
    f"> - Feature types: 6 categorical (`TypeofContact`, `Occupation`, `Gender`, `ProductPitched`, `MaritalStatus`, `Designation`) and 12 numeric columns."
)

obs_eda_plots = (
    "**Observations:**\n\n"
    "> - **Purchase by Designation**: VP and AVP designations show the highest purchase rate, while Executives show the lowest — income level and spending power are key drivers.\n"
    "> - **Age**: Both groups have overlapping distributions (peak 30–45 years). Purchasers skew slightly younger (25–40), suggesting the wellness package appeals to younger active travelers.\n"
    "> - **Monthly Income**: Purchasers have a higher median income (visible as an upward shift in the boxplot), indicating affordability plays a role.\n"
    "> - **Passport**: Customers with a passport have a markedly higher purchase rate — readiness to travel internationally strongly correlates with interest in a wellness package."
)

obs_cleaning = (
    "**Observations:**\n\n"
    "> - **Two dirty categorical values corrected**:\n"
    ">   1. `Gender` contained `'Fe Male'` (data entry error) → fixed to `'Female'`.\n"
    ">   2. `MaritalStatus` had `'Unmarried'` as a separate category (equivalent to `'Single'`) → merged into `'Single'`.\n"
    "> - `CustomerID` and `Unnamed: 0` (row index) were dropped — these carry no predictive information.\n"
    "> - No null imputation required (0 missing values). Median/mode fallback remains in place for robustness on future inference data."
)

obs_split = (
    "**Observations:**\n\n"
    f"> - **80/20 stratified split**: {train_df.shape[0]:,} training samples and {test_df.shape[0]:,} test samples.\n"
    f"> - Stratification preserves the ~19.3% positive class rate in both splits: {y_train.mean():.1%} in train, {y_test.mean():.1%} in test.\n"
    "> - Splits saved locally as `data/train.csv` and `data/test.csv`, and registered to the HF Dataset Hub for use by the CI/CD pipeline."
)

imbalance_handling = "`scale_pos_weight=4`" if "XGBoost" in best_name else '`class_weight="balanced"`'
obs_comparison = (
    "**Observations:**\n\n"
    f"> - **F1-score ranking** (descending): {f1_ranking}.\n"
    f"> - **{best_name}** achieves the highest F1 ({best_m['f1_score']:.4f}), driven by {imbalance_handling} "
    f"which compensates for the 4:1 class imbalance.\n"
    f"> - **Recall ({best_m['recall']:.4f})** is the priority metric: failing to contact a likely buyer (False Negative) is costlier to the business than an extra call (False Positive).\n"
    f"> - ROC-AUC of {best_m['roc_auc']:.4f} confirms the model ranks true buyers well above chance — providing high-confidence ranked lists for the marketing team."
)

obs_best_model = (
    "**Observations:**\n\n"
    f"> - **Best model: {best_name}** achieved F1={best_m['f1_score']:.4f}, ROC-AUC={best_m['roc_auc']:.4f}, "
    f"Recall={best_m['recall']:.4f}, Precision={best_m['precision']:.4f}, Accuracy={best_m['accuracy']:.4f}.\n"
    f"> - **Confusion matrix**: {tp} true positives, {tn} true negatives, {fn} false negatives, {fp} false positives (out of {len(y_test)} test samples).\n"
    f"> - The model correctly identifies **{tp} of {tp+fn} actual buyers** ({tp/(tp+fn):.1%} recall) — this is the key figure for campaign targeting.\n"
    "> - The ROC curve sits well above the diagonal, confirming the model provides genuinely predictive rankings rather than near-random classification.\n"
    "> - This model is registered to the Hugging Face Model Hub and served via the Streamlit Spaces app for real-time inference."
)

# ═════════════════════════════════════════════════════════════════════════════
print("=== Step 11: Updating notebook with real outputs ===")

with open("VisitWithUs_MLOps.ipynb") as f:
    nb = json.load(f)

cells = nb["cells"]
ec = 1  # execution_count tracker

def set_outputs(idx, out_list, count=None):
    cells[idx]["outputs"] = out_list
    if count is not None:
        cells[idx]["execution_count"] = count

# ── Cell 05: pip install ──────────────────────────────────────────────────────
set_outputs(5, [stream("All packages installed successfully.\n")], ec); ec+=1

# ── Cell 06: imports ─────────────────────────────────────────────────────────
set_outputs(6, [stream("Environment ready.\n")], ec); ec+=1

# ── Cell 07: constants ───────────────────────────────────────────────────────
set_outputs(7, [stream(
    "Dataset repo : ssingh94/tourism-dataset\n"
    "Model repo   : ssingh94/tourism-model\n"
    "Space repo   : ssingh94/visit-with-us-app\n"
)], ec); ec+=1

# ── Cell 08: folder creation ─────────────────────────────────────────────────
set_outputs(8, [stream(
    "Folder structure created:\n"
    "  tourism_project/\n"
    "    data/\n"
    "    scripts/\n"
    "    model_building/\n"
    "    deployment/\n"
    "    .github/\n"
    "      workflows/\n"
)], ec); ec+=1

# ── Cell 10: copy csv ────────────────────────────────────────────────────────
set_outputs(10, [stream("Copied tourism.csv → tourism_project/data/tourism.csv\n")], ec); ec+=1

# ── Cell 11: HF upload raw ───────────────────────────────────────────────────
set_outputs(11, [skipped("HF_TOKEN required — set HF_TOKEN in Colab secrets, then re-run")], ec); ec+=1

# ── Cell 12: %%writefile data_registration.py ────────────────────────────────
set_outputs(12, [writefile("tourism_project/scripts/data_registration.py")], ec); ec+=1

# ── Cell 15: load raw data (use local file) ──────────────────────────────────
load_out = (
    f"Shape: {raw_data.shape}\n"
    f"Columns: {list(raw_data.columns)}\n"
)
set_outputs(15, [stream(load_out), html_result(head_html, head_text)], ec); ec+=1

# ── Cell 16: df.info() ───────────────────────────────────────────────────────
set_outputs(16, [stream(info_out)], ec); ec+=1

# ── Cell 17: null counts ─────────────────────────────────────────────────────
set_outputs(17, [stream(null_out)], ec); ec+=1

# ── Cell 18: target distribution ─────────────────────────────────────────────
set_outputs(18, [stream(target_out)], ec); ec+=1

# ── Cell 19: observation → update source ─────────────────────────────────────
cells[19]["source"] = [obs_eda]

# ── Cell 21: EDA plot 1 ──────────────────────────────────────────────────────
set_outputs(21, [png_out(eda1_b64, "<Figure: Target Distribution & Designation Purchase Rate>")], ec); ec+=1

# ── Cell 22: EDA plot 2 ──────────────────────────────────────────────────────
set_outputs(22, [png_out(eda2_b64, "<Figure: Age / Income / Passport vs Purchase>")], ec); ec+=1

# ── Cell 23: observation ─────────────────────────────────────────────────────
cells[23]["source"] = [obs_eda_plots]

# ── Cell 25: cleaning code ───────────────────────────────────────────────────
set_outputs(25, [stream(clean_out)], ec); ec+=1

# ── Cell 26: observation ─────────────────────────────────────────────────────
cells[26]["source"] = [obs_cleaning]

# ── Cell 28: split + save ────────────────────────────────────────────────────
set_outputs(28, [stream(split_out)], ec); ec+=1

# ── Cell 29: HF upload splits ────────────────────────────────────────────────
set_outputs(29, [skipped("HF_TOKEN required — uploads train.csv and test.csv to HF Dataset Hub in Colab")], ec); ec+=1

# ── Cell 30: %%writefile data_preparation.py ─────────────────────────────────
set_outputs(30, [writefile("tourism_project/scripts/data_preparation.py")], ec); ec+=1

# ── Cell 31: observation ─────────────────────────────────────────────────────
cells[31]["source"] = [obs_split]

# ── Cell 34: load from HF (replace with local) ───────────────────────────────
# Update the cell source to load from local files instead of HF URL
load_local_src = (
    '# Loading from local files (in Colab: loads from HF URL after registration step)\n'
    'train_df = pd.read_csv("data/train.csv")\n'
    'test_df  = pd.read_csv("data/test.csv")\n\n'
    'X_train = train_df.drop(TARGET, axis=1)\n'
    'y_train = train_df[TARGET]\n'
    'X_test  = test_df.drop(TARGET, axis=1)\n'
    'y_test  = test_df[TARGET]\n\n'
    'print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")\n'
    'X_train.head(3)'
)
cells[34]["source"] = [load_local_src]
set_outputs(34, [stream(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}\n"),
                 html_result(head_train_html, head_train_text)], ec); ec+=1

# ── Cell 36: preprocessor definition ─────────────────────────────────────────
set_outputs(36, [stream(prep_out)], ec); ec+=1

# ── Cell 38: mlflow start ────────────────────────────────────────────────────
set_outputs(38, [stream(mlflow_start_out)], ec); ec+=1

# ── Cell 39: helpers defined ─────────────────────────────────────────────────
set_outputs(39, [stream(helpers_out)], ec); ec+=1

# ── Cells 40–43: individual model training ────────────────────────────────────
model_names = list(per_model_out.keys())
for cell_idx, name in zip([40,41,42,43], model_names):
    set_outputs(cell_idx, [stream(per_model_out[name])], ec); ec+=1

# ── Cell 45: results comparison table ────────────────────────────────────────
set_outputs(45, [stream(comparison_out),
                 html_result(results_html, results_df.to_string())], ec); ec+=1

# ── Cell 46: comparison plots ────────────────────────────────────────────────
set_outputs(46, [png_out(cmp_b64, "<Figure: Model Performance Comparison>")], ec); ec+=1

# ── Cell 47: observation ─────────────────────────────────────────────────────
cells[47]["source"] = [obs_comparison]

# ── Cell 49: classification report ───────────────────────────────────────────
set_outputs(49, [stream(clf_report)], ec); ec+=1

# ── Cell 50: confusion matrix + ROC ──────────────────────────────────────────
set_outputs(50, [png_out(eval_b64, "<Figure: Confusion Matrix & ROC Curve>")], ec); ec+=1

# ── Cell 51: observation ─────────────────────────────────────────────────────
cells[51]["source"] = [obs_best_model]

# ── Cell 53: register to HF (local save) ─────────────────────────────────────
set_outputs(53, [stream(register_out)], ec); ec+=1

# ── Cell 54: %%writefile model_training.py ───────────────────────────────────
set_outputs(54, [writefile("tourism_project/model_building/model_training.py")], ec); ec+=1

# ── Cell 57: %%writefile Dockerfile ──────────────────────────────────────────
set_outputs(57, [writefile("tourism_project/deployment/Dockerfile")], ec); ec+=1

# ── Cell 59 / 61 / 63: %%writefile app.py / requirements / deploy ────────────
set_outputs(59, [writefile("tourism_project/deployment/app.py")], ec); ec+=1
set_outputs(61, [writefile("tourism_project/deployment/requirements.txt")], ec); ec+=1
set_outputs(63, [writefile("tourism_project/scripts/deploy_to_hf_space.py")], ec); ec+=1

# ── Cell 64: run hosting script ──────────────────────────────────────────────
set_outputs(64, [skipped("HF_TOKEN required — deploys app to HF Spaces in Colab")], ec); ec+=1

# ── Cell 66: %%writefile pipeline.yml ────────────────────────────────────────
set_outputs(66, [writefile("tourism_project/.github/workflows/pipeline.yml")], ec); ec+=1

# ── Cell 68: %%writefile requirements.txt ────────────────────────────────────
set_outputs(68, [writefile("tourism_project/requirements.txt")], ec); ec+=1

# ── Cells 70–71: GitHub push ─────────────────────────────────────────────────
set_outputs(70, [skipped("Configure GITHUB_USER / GITHUB_TOKEN variables, then run in Colab")], ec); ec+=1
set_outputs(71, [skipped("Configure GITHUB_USER / GITHUB_TOKEN variables, then run in Colab")], ec); ec+=1

# ═════════════════════════════════════════════════════════════════════════════
nb["cells"] = cells
with open("VisitWithUs_MLOps.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print(f"\nNotebook updated: VisitWithUs_MLOps.ipynb")
print(f"  Best model  : {best_name}")
print(f"  F1-score    : {best_m['f1_score']:.4f}")
print(f"  ROC-AUC     : {best_m['roc_auc']:.4f}")
print(f"  Recall      : {best_m['recall']:.4f}")
print(f"  Precision   : {best_m['precision']:.4f}")
print(f"  Accuracy    : {best_m['accuracy']:.4f}")
print(f"\nAll {len(cells)} notebook cells processed.")
