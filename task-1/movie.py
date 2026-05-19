import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import warnings, textwrap
warnings.filterwarnings('ignore')

TRAIN_PATH = "train_data.txt"
TEST_PATH  = "test_data.txt"
SOL_PATH   = "test_data_solution.txt"
OUT = r"c:\Users\nitin\OneDrive\Desktop\intern\task-1" 

# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_train(path):
    rows = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if len(parts) == 4:
                rows.append({'id': parts[0], 'title': parts[1],
                             'genre': parts[2].lower().strip(),
                             'description': parts[3]})
    return pd.DataFrame(rows)

def load_test(path):
    rows = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if len(parts) == 3:
                rows.append({'id': parts[0], 'title': parts[1],
                             'description': parts[2]})
    return pd.DataFrame(rows)

print("Loading data …")
train_df = load_train(TRAIN_PATH)
test_df  = load_test(TEST_PATH)
sol_df   = load_train(SOL_PATH)   # solution file has genre column

print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
print(f"Genres: {train_df['genre'].nunique()}")

# Keep top 15 genres (removes rare categories that hurt metrics)
TOP_N = 15
top_genres = [g for g, _ in Counter(train_df['genre']).most_common(TOP_N)]
train_df = train_df[train_df['genre'].isin(top_genres)].copy()
test_df_merged = test_df.merge(sol_df[['id', 'genre']], on='id', how='inner')
test_df_merged = test_df_merged[test_df_merged['genre'].isin(top_genres)].copy()

X_train = train_df['description']
y_train = train_df['genre']
X_test  = test_df_merged['description']
y_test  = test_df_merged['genre']

print(f"After filtering — Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── 2. Pipelines ──────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=60_000, ngram_range=(1, 2),
                        sublinear_tf=True, min_df=2)

pipelines = {
    'Naive Bayes':          Pipeline([('tfidf', tfidf), ('clf', MultinomialNB(alpha=0.1))]),
    'Logistic Regression':  Pipeline([('tfidf', tfidf), ('clf', LogisticRegression(max_iter=1000, C=5, solver='saga', n_jobs=-1))]),
    'Linear SVM':           Pipeline([('tfidf', tfidf), ('clf', LinearSVC(C=1.0, max_iter=2000))]),
}

results = {}
for name, pipe in pipelines.items():
    print(f"  Training {name} …")
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    f1    = f1_score(y_test, preds, average='weighted')
    results[name] = {'preds': preds, 'acc': acc, 'f1': f1}
    print(f"    → Acc: {acc:.4f}  |  F1: {f1:.4f}")

best_name = max(results, key=lambda k: results[k]['f1'])
best_preds = results[best_name]['preds']
print(f"\nBest model: {best_name}")

# ── 3. Colour palette ─────────────────────────────────────────────────────────
PALETTE = sns.color_palette("tab20", TOP_N)
genre_colors = dict(zip(sorted(top_genres), PALETTE))
PLT_STYLE = {'axes.facecolor': '#f8f9fa', 'figure.facecolor': '#ffffff',
             'axes.grid': True, 'grid.color': '#e0e0e0', 'axes.spines.top': False,
             'axes.spines.right': False}
plt.rcParams.update(PLT_STYLE)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Genre Distribution in Training Data
# ══════════════════════════════════════════════════════════════════════════════
genre_counts = train_df['genre'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Movie Genre Distribution – Training Data', fontsize=16, fontweight='bold', y=1.01)

# Bar chart
bars = axes[0].barh(genre_counts.index, genre_counts.values,
                    color=[genre_colors[g] for g in genre_counts.index])
axes[0].set_xlabel('Number of Movies', fontsize=11)
axes[0].set_title('Sample Count per Genre', fontsize=12)
for bar, val in zip(bars, genre_counts.values):
    axes[0].text(val + 50, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', fontsize=8)
axes[0].invert_yaxis()

# Pie chart
wedges, texts, autotexts = axes[1].pie(
    genre_counts.values, labels=None,
    colors=[genre_colors[g] for g in genre_counts.index],
    autopct='%1.1f%%', pctdistance=0.82,
    wedgeprops=dict(width=0.55, edgecolor='white', linewidth=1.5))
for at in autotexts:
    at.set_fontsize(7)
axes[1].legend(genre_counts.index, loc='center left',
               bbox_to_anchor=(1, 0.5), fontsize=8)
axes[1].set_title('Genre Proportion', fontsize=12)

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_genre_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Fig 1 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Model Comparison (Accuracy & F1)
# ══════════════════════════════════════════════════════════════════════════════
model_names = list(results.keys())
accs = [results[m]['acc'] for m in model_names]
f1s  = [results[m]['f1']  for m in model_names]

x = np.arange(len(model_names))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - width/2, accs, width, label='Accuracy',
            color='#4C72B0', alpha=0.88, edgecolor='white')
b2 = ax.bar(x + width/2, f1s,  width, label='Weighted F1',
            color='#DD8452', alpha=0.88, edgecolor='white')
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison\n(TF-IDF + Classifier)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
# Highlight best
ax.axvline(model_names.index(best_name), color='green', linestyle='--', alpha=0.4, label='Best')
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Fig 2 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – Confusion Matrix (Best Model)
# ══════════════════════════════════════════════════════════════════════════════
cm = confusion_matrix(y_test, best_preds, labels=sorted(top_genres))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(14, 11))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Recall (row-normalised)')
ax.set_xticks(range(len(sorted(top_genres))))
ax.set_yticks(range(len(sorted(top_genres))))
ax.set_xticklabels(sorted(top_genres), rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(sorted(top_genres), fontsize=9)
for i in range(len(sorted(top_genres))):
    for j in range(len(sorted(top_genres))):
        val = cm_norm[i, j]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=7, color='white' if val > 0.6 else 'black')
ax.set_xlabel('Predicted Genre', fontsize=12)
ax.set_ylabel('True Genre', fontsize=12)
ax.set_title(f'Confusion Matrix – {best_name}\n(Row-normalised Recall)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Fig 3 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Per-Genre Precision, Recall, F1
# ══════════════════════════════════════════════════════════════════════════════
report = classification_report(y_test, best_preds,
                                labels=sorted(top_genres),
                                output_dict=True, zero_division=0)
genres_sorted = sorted(top_genres)
prec = [report[g]['precision'] for g in genres_sorted]
rec  = [report[g]['recall']    for g in genres_sorted]
f1_per = [report[g]['f1-score']  for g in genres_sorted]

x = np.arange(len(genres_sorted))
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(x, prec,   marker='o', linewidth=2, label='Precision', color='#4C72B0')
ax.plot(x, rec,    marker='s', linewidth=2, label='Recall',    color='#DD8452')
ax.plot(x, f1_per, marker='^', linewidth=2, label='F1-Score',  color='#55A868')
ax.set_xticks(x); ax.set_xticklabels(genres_sorted, rotation=45, ha='right', fontsize=10)
ax.set_ylim(0, 1.05); ax.set_ylabel('Score', fontsize=12)
ax.set_title(f'Per-Genre Metrics – {best_name}', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.axhline(report['weighted avg']['f1-score'], color='gray', linestyle='--',
           alpha=0.6, label='Avg F1')
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_per_genre_metrics.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Fig 4 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 – Top TF-IDF Features per Genre
# ══════════════════════════════════════════════════════════════════════════════
best_pipe = pipelines[best_name]
vectorizer = best_pipe.named_steps['tfidf']
clf        = best_pipe.named_steps['clf']
feature_names = np.array(vectorizer.get_feature_names_out())

SHOW_GENRES = ['drama', 'comedy', 'thriller', 'horror', 'documentary',
               'action', 'romance', 'sci-fi', 'western', 'animation']

if hasattr(clf, 'coef_'):
    classes = list(clf.classes_)
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    fig.suptitle(f'Top 10 TF-IDF Features per Genre – {best_name}',
                 fontsize=14, fontweight='bold')
    for ax, genre in zip(axes.flatten(), SHOW_GENRES):
        if genre not in classes:
            ax.axis('off'); continue
        idx = classes.index(genre)
        coefs = clf.coef_[idx] if hasattr(clf.coef_, 'ndim') and clf.coef_.ndim == 2 else clf.coef_
        top_idx = np.argsort(coefs)[-10:][::-1]
        words   = feature_names[top_idx]
        scores  = coefs[top_idx]
        color   = genre_colors.get(genre, '#4C72B0')
        bars = ax.barh(range(10), scores[::-1], color=color, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(10))
        ax.set_yticklabels(words[::-1], fontsize=8)
        ax.set_title(genre.title(), fontsize=10, fontweight='bold')
        ax.set_xlabel('Coefficient', fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_top_features.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Fig 5 saved")
else:
    print("  (Skipping Fig 5 – model has no coef_)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 – Description Length Distribution by Genre
# ══════════════════════════════════════════════════════════════════════════════
train_df['desc_len'] = train_df['description'].str.split().str.len()
fig, ax = plt.subplots(figsize=(14, 6))
data_by_genre = [train_df[train_df['genre'] == g]['desc_len'].values
                 for g in sorted(top_genres)]
bp = ax.boxplot(data_by_genre, patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2),
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
for patch, genre in zip(bp['boxes'], sorted(top_genres)):
    patch.set_facecolor(genre_colors[genre]); patch.set_alpha(0.8)
ax.set_xticks(range(1, len(sorted(top_genres)) + 1))
ax.set_xticklabels(sorted(top_genres), rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Word Count', fontsize=12)
ax.set_title('Description Length Distribution by Genre', fontsize=14, fontweight='bold')
ax.set_ylim(0, 500)
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_description_length.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Fig 6 saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 – F1 Score per Genre (all 3 models, grouped bars)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 7))
n_models = len(pipelines)
x = np.arange(len(genres_sorted))
width = 0.25
model_palette = ['#4C72B0', '#DD8452', '#55A868']
for i, (mname, mpipe) in enumerate(pipelines.items()):
    preds_m = results[mname]['preds']
    rep_m   = classification_report(y_test, preds_m,
                                     labels=genres_sorted,
                                     output_dict=True, zero_division=0)
    f1s_m = [rep_m[g]['f1-score'] for g in genres_sorted]
    ax.bar(x + (i - 1) * width, f1s_m, width,
           label=mname, color=model_palette[i], alpha=0.85, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(genres_sorted, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_title('F1-Score per Genre – All Models Compared', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_f1_all_models.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Fig 7 saved")

print("\n✅ All figures saved!")
print(f"\n{'='*55}")
print(f"  FINAL RESULTS SUMMARY")
print(f"{'='*55}")
for name, res in results.items():
    marker = " ⭐ BEST" if name == best_name else ""
    print(f"  {name:<22}  Acc={res['acc']:.4f}  F1={res['f1']:.4f}{marker}")
print(f"{'='*55}")