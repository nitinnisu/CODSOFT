"""
SMS Spam Detection — Full ML Pipeline

"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import re
import string

# 1. Load & Clean

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})

print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")
print(f"\nSpam ratio: {df['label'].value_counts(normalize=True)['spam']:.1%}")

# 2. Text Preprocessing 

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', 'NUM', text)           # numbers → NUM
    text = re.sub(r'http\S+|www\S+', 'URL', text) # URLs → URL
    text = re.sub(r'[^\w\s]', ' ', text)           # strip punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_msg'] = df['message'].apply(preprocess)

# Encode labels: spam=1, ham=0
le = LabelEncoder()
y = le.fit_transform(df['label'])   # ham→0, spam→1
X = df['clean_msg']

# 3. Train / Test Split 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# 4. Build Pipelines 

tfidf_params = dict(
    max_features=10_000,
    ngram_range=(1, 2),   # unigrams + bigrams
    sublinear_tf=True,
    min_df=2
)

pipelines = {
    'Naive Bayes':           Pipeline([('tfidf', TfidfVectorizer(**tfidf_params)),
                                       ('clf',  MultinomialNB(alpha=0.1))]),
    'Logistic Regression':   Pipeline([('tfidf', TfidfVectorizer(**tfidf_params)),
                                       ('clf',  LogisticRegression(C=5, max_iter=1000, random_state=42))]),
    'Support Vector Machine':Pipeline([('tfidf', TfidfVectorizer(**tfidf_params)),
                                       ('clf',  LinearSVC(C=1.0, max_iter=2000, random_state=42))]),
}

# 5. Train, Evaluate & Collect Metrics

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # SVM doesn't have predict_proba natively via LinearSVC
    if hasattr(pipe.named_steps['clf'], 'predict_proba'):
        y_prob = pipe.predict_proba(X_test)[:, 1]
    else:
        y_prob = pipe.decision_function(X_test)

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1')

    results[name] = {
        'pipeline':   pipe,
        'y_pred':     y_pred,
        'y_prob':     y_prob,
        'accuracy':   accuracy_score(y_test, y_pred),
        'precision':  precision_score(y_test, y_pred),
        'recall':     recall_score(y_test, y_pred),
        'f1':         f1_score(y_test, y_pred),
        'roc_auc':    roc_auc_score(y_test, y_prob),
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std':  cv_scores.std(),
        'cm':         confusion_matrix(y_test, y_pred),
    }
    print(f"\n── {name} ──")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# 6. Visualisations 

COLORS = {
    'Naive Bayes':            '#6366f1',
    'Logistic Regression':    '#10b981',
    'Support Vector Machine': '#f59e0b',
}

fig = plt.figure(figsize=(20, 22), facecolor='#0f172a')
fig.suptitle('SMS Spam Detection — Model Comparison', fontsize=20,
             fontweight='bold', color='white', y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# Row 0: Metric Bar Charts 
metric_names = ['accuracy', 'precision', 'recall', 'f1']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metric_names))
width = 0.25

ax_metrics = fig.add_subplot(gs[0, :])
ax_metrics.set_facecolor('#1e293b')
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metric_names]
    bars = ax_metrics.bar(x + i*width, vals, width, label=name,
                          color=COLORS[name], alpha=0.9, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8,
                        color='white', fontweight='bold')

ax_metrics.set_xticks(x + width)
ax_metrics.set_xticklabels(metric_labels, fontsize=12, color='white')
ax_metrics.set_ylim(0.85, 1.02)
ax_metrics.set_title('Metric Comparison Across Models', color='white', fontsize=14, pad=10)
ax_metrics.legend(loc='lower right', framealpha=0.3, labelcolor='white')
ax_metrics.tick_params(colors='white')
ax_metrics.set_facecolor('#1e293b')
for spine in ax_metrics.spines.values():
    spine.set_edgecolor('#334155')

# Row 1: Confusion Matrices
for col, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, col])
    ax.set_facecolor('#1e293b')
    cm = res['cm']
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap='Blues', cbar=False,
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                linewidths=1, linecolor='#334155')
    ax.set_title(f'{name}\nConfusion Matrix', color='white', fontsize=10)
    ax.set_xlabel('Predicted', color='#94a3b8', fontsize=9)
    ax.set_ylabel('Actual', color='#94a3b8', fontsize=9)
    ax.tick_params(colors='white', labelsize=9)
    for text in ax.texts:
        text.set_color('white' if float(text.get_text()) < cm.max()*0.7 else '#0f172a')

# Row 2: ROC Curves + Cross-val F1 
ax_roc = fig.add_subplot(gs[2, :2])
ax_roc.set_facecolor('#1e293b')
ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax_roc.plot(fpr, tpr, color=COLORS[name], lw=2,
                label=f'{name} (AUC={res["roc_auc"]:.4f})')
ax_roc.set_xlabel('False Positive Rate', color='#94a3b8')
ax_roc.set_ylabel('True Positive Rate', color='#94a3b8')
ax_roc.set_title('ROC Curves', color='white', fontsize=13)
ax_roc.legend(framealpha=0.3, labelcolor='white', fontsize=9)
ax_roc.tick_params(colors='white')
for spine in ax_roc.spines.values():
    spine.set_edgecolor('#334155')

ax_cv = fig.add_subplot(gs[2, 2])
ax_cv.set_facecolor('#1e293b')
names_short = ['NB', 'LR', 'SVM']
cv_means = [r['cv_f1_mean'] for r in results.values()]
cv_stds  = [r['cv_f1_std']  for r in results.values()]
bars = ax_cv.bar(names_short, cv_means, color=list(COLORS.values()),
                 alpha=0.9, edgecolor='white', linewidth=0.5)
ax_cv.errorbar(names_short, cv_means, yerr=cv_stds, fmt='none',
               color='white', capsize=5, linewidth=2)
for bar, v in zip(bars, cv_means):
    ax_cv.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{v:.4f}', ha='center', va='bottom', fontsize=9,
               color='white', fontweight='bold')
ax_cv.set_ylim(min(cv_means) - 0.02, 1.01)
ax_cv.set_title('5-Fold Cross-Val F1', color='white', fontsize=12)
ax_cv.tick_params(colors='white')
for spine in ax_cv.spines.values():
    spine.set_edgecolor('#334155')

# Row 3: Precision-Recall Curves + Top TF-IDF Features
ax_pr = fig.add_subplot(gs[3, :2])
ax_pr.set_facecolor('#1e293b')
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ax_pr.plot(rec, prec, color=COLORS[name], lw=2, label=name)
ax_pr.set_xlabel('Recall', color='#94a3b8')
ax_pr.set_ylabel('Precision', color='#94a3b8')
ax_pr.set_title('Precision-Recall Curves', color='white', fontsize=13)
ax_pr.legend(framealpha=0.3, labelcolor='white', fontsize=9)
ax_pr.tick_params(colors='white')
for spine in ax_pr.spines.values():
    spine.set_edgecolor('#334155')

# Top spam-indicative TF-IDF features (from Logistic Regression)
ax_feat = fig.add_subplot(gs[3, 2])
ax_feat.set_facecolor('#1e293b')
lr_pipe = results['Logistic Regression']['pipeline']
tfidf   = lr_pipe.named_steps['tfidf']
clf     = lr_pipe.named_steps['clf']
feature_names = tfidf.get_feature_names_out()
top_idx = np.argsort(clf.coef_[0])[-15:][::-1]
top_feats = feature_names[top_idx]
top_coefs = clf.coef_[0][top_idx]
y_pos = np.arange(len(top_feats))
ax_feat.barh(y_pos, top_coefs, color='#f59e0b', alpha=0.85, edgecolor='white', linewidth=0.4)
ax_feat.set_yticks(y_pos)
ax_feat.set_yticklabels(top_feats, fontsize=8, color='white')
ax_feat.set_title('Top 15 Spam Words\n(LR Coefficients)', color='white', fontsize=10)
ax_feat.tick_params(colors='white', labelsize=8)
for spine in ax_feat.spines.values():
    spine.set_edgecolor('#334155')

plt.savefig('spam_detection_results.png',
            dpi=150, bbox_inches='tight', facecolor='#0f172a')
print("\n✅ Chart saved.")

# 7. Summary Table
print("\n" + "="*70)
print(f"{'Model':<28} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'CV-F1':>12}")
print("="*70)
for name, res in results.items():
    print(f"{name:<28} {res['accuracy']:>7.4f} {res['precision']:>7.4f} "
          f"{res['recall']:>7.4f} {res['f1']:>7.4f} {res['roc_auc']:>7.4f} "
          f"{res['cv_f1_mean']:>6.4f}±{res['cv_f1_std']:.4f}")

# 8. Demo Predictions 
print("\n── Live Predictions (Best Model: SVM) ──")
test_msgs = [
    "Congratulations! You've won a FREE iPhone. Click here to claim now!",
    "Hey, are we still meeting at 6 for dinner?",
    "URGENT: Your account has been SUSPENDED. Call 0800-FREE now!",
    "Can you pick up some milk on your way home?",
    "Win £1000 cash prize! Text WIN to 80085 NOW. Ts&Cs apply.",
]
svm_pipe = pipelines['Support Vector Machine']
for msg in test_msgs:
    pred = svm_pipe.predict([preprocess(msg)])[0]
    label = '🚫 SPAM' if pred == 1 else '✅ HAM '
    print(f"  {label}  |  {msg[:65]}")