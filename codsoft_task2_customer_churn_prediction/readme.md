# 🔄 Customer Churn Prediction

Predict customer churn using machine learning on 10,000 bank customers.

---

## 📁 Dataset
**Churn_Modelling.csv** — 10,000 rows, 14 features including demographics, account info, and activity status.

| Feature | Description |
|---|---|
| CreditScore, Age, Tenure | Customer profile |
| Balance, EstimatedSalary | Financial info |
| Geography, Gender | Demographics |
| NumOfProducts, IsActiveMember | Engagement |
| **Exited** | Target — 1 = Churned |

---

## ⚙️ Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python customer_churn_prediction.py
```

> Place `Churn_Modelling.csv` in the same directory.

---

## 🤖 Models

| Model | Accuracy | F1 Score | ROC AUC |
|---|---|---|---|
| Logistic Regression | 80.50% | 22.92% | 77.10% |
| Random Forest | 86.40% | 57.89% | 84.64% |
| **Gradient Boosting** ★ | **86.75%** | **59.42%** | **86.73%** |

---

## 📊 Key Insights

- **20.4%** overall churn rate
- **Germany** has the highest churn (32.4%)
- Customers with **3–4 products** churn at 82–100%
- **Age** is the #1 predictive feature (39.6% importance)
- Churned customers are ~7 years older on average

---

## 📈 Output Plots

| File | Description |
|---|---|
| `eda_plots.png` | Churn by geography, gender, products, age, balance |
| `model_comparison.png` | Metrics bar chart + ROC curves |
| `confusion_matrices.png` | All three models |
| `feature_importance.png` | Gradient Boosting feature ranking |

---

## 🗂️ Project Structure

```
├── Churn_Modelling.csv
├── customer_churn_prediction.py
├── eda_plots.png
├── model_comparison.png
├── confusion_matrices.png
├── feature_importance.png
└── README.md
```

---
## 👤 Author
**Nitin choudhary** 
