# 🎬 Movie Genre Classification

Predicts a movie's genre from its plot summary using TF-IDF and machine learning classifiers.

---

## 📁 Project Structure

```
task-1/
├── movie.py                    # Main script
├── train_data.txt              # Training data (54,214 movies)
├── test_data.txt               # Test data (54,200 movies)
├── test_data_solution.txt      # Test labels
└── fig1–fig7 *.png             # Output charts
```

---

## 🧠 Models & Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Naive Bayes | 60.5% | 0.561 |
| Linear SVM | 63.5% | 0.616 |
| **Logistic Regression** ✅ | **64.4%** | **0.622** |

---

## 🚀 How to Run

```bash
pip install scikit-learn matplotlib seaborn pandas numpy
python movie.py
```

> Update the file paths at the top of `movie.py` to match your local folder.

---

## 📊 Output Charts

| File | Description |
|------|-------------|
| fig1 | Genre distribution (bar + pie) |
| fig2 | Model accuracy & F1 comparison |
| fig3 | Confusion matrix |
| fig4 | Precision / Recall / F1 per genre |
| fig5 | Top TF-IDF keywords per genre |
| fig6 | Plot summary length by genre |
| fig7 | F1 score across all models & genres |

---

## 🛠️ Tech Stack

`Python` `scikit-learn` `TF-IDF` `pandas` `matplotlib` `seaborn`

---

## 👤 Author
**Nitin choudhary** 
