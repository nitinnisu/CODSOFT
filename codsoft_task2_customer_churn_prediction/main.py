"""
Customer Churn Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)


# LOAD DATA

df = pd.read_csv("Churn_Modelling.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape        : {df.shape}")
print(f"Churn count  : {df['Exited'].sum()} ({df['Exited'].mean()*100:.1f}%)")
print(f"\nFeatures:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")


# EXPLORATORY DATA ANALYSIS (EDA)

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nChurn by Geography:")
print(df.groupby("Geography")["Exited"].mean().mul(100).round(2).to_string())

print("\nChurn by Gender:")
print(df.groupby("Gender")["Exited"].mean().mul(100).round(2).to_string())

print("\nChurn by NumOfProducts:")
print(df.groupby("NumOfProducts")["Exited"].mean().mul(100).round(2).to_string())

print("\nAge — churned vs retained:")
print(df.groupby("Exited")["Age"].mean().round(1))

print("\nBalance — churned vs retained:")
print(df.groupby("Exited")["Balance"].mean().round(2))

# EDA plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Customer Churn — EDA", fontsize=16, fontweight="bold")

# Churn distribution
axes[0, 0].bar(["Retained", "Churned"], df["Exited"].value_counts().sort_index(),
               color=["#2ecc71", "#e74c3c"])
axes[0, 0].set_title("Churn Distribution")
axes[0, 0].set_ylabel("Count")
for i, v in enumerate(df["Exited"].value_counts().sort_index()):
    axes[0, 0].text(i, v + 50, str(v), ha="center", fontweight="bold")

# Churn by Geography
geo_churn = df.groupby("Geography")["Exited"].mean() * 100
axes[0, 1].bar(geo_churn.index, geo_churn.values, color=["#3498db", "#e74c3c", "#2ecc71"])
axes[0, 1].set_title("Churn Rate by Geography")
axes[0, 1].set_ylabel("Churn Rate (%)")
for i, v in enumerate(geo_churn.values):
    axes[0, 1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

# Churn by Gender
gender_churn = df.groupby("Gender")["Exited"].mean() * 100
axes[0, 2].bar(gender_churn.index, gender_churn.values, color=["#e91e8c", "#3498db"])
axes[0, 2].set_title("Churn Rate by Gender")
axes[0, 2].set_ylabel("Churn Rate (%)")
for i, v in enumerate(gender_churn.values):
    axes[0, 2].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")

# Age distribution by churn
df[df["Exited"] == 0]["Age"].plot(kind="hist", bins=30, ax=axes[1, 0],
                                   alpha=0.6, color="#2ecc71", label="Retained")
df[df["Exited"] == 1]["Age"].plot(kind="hist", bins=30, ax=axes[1, 0],
                                   alpha=0.6, color="#e74c3c", label="Churned")
axes[1, 0].set_title("Age Distribution by Churn")
axes[1, 0].set_xlabel("Age")
axes[1, 0].legend()

# Balance distribution by churn
df[df["Exited"] == 0]["Balance"].plot(kind="hist", bins=30, ax=axes[1, 1],
                                       alpha=0.6, color="#2ecc71", label="Retained")
df[df["Exited"] == 1]["Balance"].plot(kind="hist", bins=30, ax=axes[1, 1],
                                       alpha=0.6, color="#e74c3c", label="Churned")
axes[1, 1].set_title("Balance Distribution by Churn")
axes[1, 1].set_xlabel("Balance")
axes[1, 1].legend()

# Churn by NumOfProducts
prod_churn = df.groupby("NumOfProducts")["Exited"].mean() * 100
axes[1, 2].bar(prod_churn.index.astype(str), prod_churn.values,
               color=["#2ecc71", "#f39c12", "#e74c3c", "#c0392b"])
axes[1, 2].set_title("Churn Rate by Number of Products")
axes[1, 2].set_xlabel("Number of Products")
axes[1, 2].set_ylabel("Churn Rate (%)")
for i, v in enumerate(prod_churn.values):
    axes[1, 2].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nEDA plot saved → eda_plots.png")


# PREPROCESSING
df_model = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1).copy()

le = LabelEncoder()
df_model["Geography"] = le.fit_transform(df_model["Geography"])
df_model["Gender"]    = le.fit_transform(df_model["Gender"])

X = df_model.drop("Exited", axis=1)
y = df_model["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)
print(f"Train : {X_train.shape[0]} rows | Test : {X_test.shape[0]} rows")
print(f"Churn in train : {y_train.mean()*100:.1f}% | test : {y_test.mean()*100:.1f}%")


# MODEL TRAINING

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        True   # needs scaled features
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=100, random_state=42),
        False
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=42),
        False
    ),
}

results  = {}
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

for name, (model, needs_scale) in models.items():
    Xtr = X_train_sc if needs_scale else X_train
    Xte = X_test_sc  if needs_scale else X_test

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]

    cv_scores = cross_val_score(model, Xtr, y_train, cv=cv,
                                scoring="roc_auc", n_jobs=-1)

    results[name] = {
        "model"    : model,
        "preds"    : preds,
        "proba"    : proba,
        "accuracy" : accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall"   : recall_score(y_test, preds),
        "f1"       : f1_score(y_test, preds),
        "roc_auc"  : roc_auc_score(y_test, proba),
        "cv_auc"   : cv_scores.mean(),
        "cv_std"   : cv_scores.std(),
    }

    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Accuracy  : {results[name]['accuracy']*100:.2f}%")
    print(f"  Precision : {results[name]['precision']*100:.2f}%")
    print(f"  Recall    : {results[name]['recall']*100:.2f}%")
    print(f"  F1 Score  : {results[name]['f1']*100:.2f}%")
    print(f"  ROC AUC   : {results[name]['roc_auc']*100:.2f}%")
    print(f"  CV AUC    : {results[name]['cv_auc']*100:.2f}% ± {results[name]['cv_std']*100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=["Retained", "Churned"]))


# COMPARISON PLOT

metrics    = ["accuracy", "precision", "recall", "f1", "roc_auc"]
model_names = list(results.keys())
x          = np.arange(len(metrics))
width      = 0.25
colors     = ["#3498db", "#2ecc71", "#e74c3c"]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")

for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [results[name][m] * 100 for m in metrics]
    axes[0].bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)

axes[0].set_title("Performance Metrics")
axes[0].set_xticks(x + width)
axes[0].set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC AUC"], rotation=10)
axes[0].set_ylabel("Score (%)")
axes[0].set_ylim(0, 105)
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# ROC curves
for (name, color) in zip(model_names, colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]["proba"])
    auc_score   = results[name]["roc_auc"]
    axes[1].plot(fpr, tpr, color=color, lw=2,
                 label=f"{name} (AUC = {auc_score:.3f})")

axes[1].plot([0, 1], [0, 1], "k--", lw=1, label="Random")
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.02])
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curves")
axes[1].legend(loc="lower right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Model comparison plot saved → model_comparison.png")


# CONFUSION MATRICES

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")

for ax, (name, color) in zip(axes, zip(model_names, colors)):
    cm = confusion_matrix(y_test, results[name]["preds"])
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("Confusion matrices saved → confusion_matrices.png")


# FEATURE IMPORTANCE — Gradient Boosting

gb_model  = results["Gradient Boosting"]["model"]
feat_imp  = pd.Series(gb_model.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = ["#e74c3c" if v >= feat_imp.quantile(0.75) else
              "#f39c12" if v >= feat_imp.median() else "#3498db"
              for v in feat_imp.values]

feat_imp.plot(kind="barh", ax=ax, color=colors_bar)
ax.set_title("Feature Importance — Gradient Boosting", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
for i, v in enumerate(feat_imp.values):
    ax.text(v + 0.001, i, f"{v*100:.2f}%", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Feature importance plot saved → feature_importance.png")


# BEST MODEL SUMMARY

best_name  = max(results, key=lambda k: results[k]["roc_auc"])
best       = results[best_name]

print("\n" + "=" * 60)
print(f"BEST MODEL : {best_name}")
print("=" * 60)
print(f"  ROC AUC   : {best['roc_auc']*100:.2f}%")
print(f"  F1 Score  : {best['f1']*100:.2f}%")
print(f"  Accuracy  : {best['accuracy']*100:.2f}%")
print(f"  Precision : {best['precision']*100:.2f}%")
print(f"  Recall    : {best['recall']*100:.2f}%")

print("\nTop 5 features driving churn:")
top5 = pd.Series(gb_model.feature_importances_,
                 index=X.columns).sort_values(ascending=False).head(5)
for feat, score in top5.items():
    print(f"  {feat:<20} {score*100:.2f}%")

print("\nDone! All plots saved to disk.")
