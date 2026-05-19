# SMS Spam Detection

A machine learning model that classifies SMS messages as **spam** or **ham** using TF-IDF and three classifiers.

## Models Used
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM) ✅ Best

## Results

| Model | Accuracy | F1-Score |
|---|---|---|
| Naive Bayes | 98.57% | 94.44% |
| Logistic Regression | 98.57% | 94.59% |
| SVM | **98.74%** | **95.27%** |

## Setup

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

Place `spam.csv` in the same folder and run:

```bash
python main.py
```

## Output
- Metrics printed in terminal
- `spam_detection_results.png` chart saved in the folder

## Dataset
[SMS Spam Collection – Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — 5,572 messages (86.6% ham, 13.4% spam)
