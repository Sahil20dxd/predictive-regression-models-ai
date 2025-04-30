# **Regression Analysis with AI Models using NumPy & scikit-learn**

This project demonstrates how **AI regression models** can be used to predict a continuous target variable using structured, normalized data. The focus is on understanding and comparing different regression techniques like **Linear Regression, K-NN Regression, Decision Trees, Random Forest, and Support Vector Regression (SVR)** — all evaluated using **cross-validation** and **Mean Squared Error (MSE)**.

Developed as part of an academic assignment, this project offers a practical introduction to **machine learning for regression tasks**, while showcasing how **NumPy and scikit-learn** are used for building and tuning models.

---

## **📁 Files**

- `Assignment_3.py` — Python script containing model training, evaluation, and comparison logic for multiple regression algorithms.
- `train_dataset.csv` — Dataset with **10 normalized features** and a continuous **target variable** for regression.
- `Regression Report.docx` — Detailed analysis, performance results, best parameter configurations, and recommendations based on the evaluation.

---

## **🎯 Project Objective**

The primary task is **to predict a target value** using a dataset with 10 normalized input features through various **supervised regression algorithms**. The project highlights:

- Use of **cross-validation** to ensure robust model evaluation
- Parameter tuning using **GridSearchCV**
- Comparison of multiple algorithms based on **Mean Squared Error (MSE)**
- Selection of the best model for prediction on unseen data

---

## **🤖 AI Models Implemented**

### ✅ 1. **Linear Regression**
- Acts as a baseline model.
- Simple and interpretable, but may not capture non-linear patterns.
- Average MSE: **5.0970**

---

### ✅ 2. **K-Nearest Neighbors (K-NN) Regression**
- Parameters: `n_neighbors = 10`, `p = 2`, `weights = "distance"`
- Distance-weighted regression based on similar training samples.
- Captures **local non-linear trends** in the data.
- Average MSE: **5.0135**

---

### ✅ 3. **Decision Tree Regression**
- Parameters: `max_depth = 5`, `min_samples_split = 2`
- Tends to **overfit** with shallow depth or insufficient splits.
- Least effective in this dataset.
- Average MSE: **5.5925**

---

### ✅ 4. **Random Forest Regression (Best Performer)**
- Parameters: `n_estimators = 200`, `max_depth = 10`, `min_samples_split = 10`
- Powerful ensemble model that captures **complex non-linear relationships**.
- Best performing algorithm in terms of **lowest MSE**.
- Average MSE: **4.5930**

---

### ✅ 5. **Support Vector Regression (SVR)**
- Parameters: `kernel = "rbf"`, `C = 10.0`, `gamma = "scale"`
- Models complex functions using kernel tricks.
- Performs well with balanced regularization.
- Average MSE: **4.7775**

---

## **📊 Performance Summary**

| Model               | Avg MSE | Best MSE | Worst MSE |
|--------------------|---------|----------|-----------|
| **Random Forest**   | 4.5930  | 4.4918   | 5.1374    |
| **SVR**             | 4.7775  | 5.2497   | 5.8093    |
| **K-NN**            | 5.0135  | 4.8696   | 5.9289    |
| **Linear Regression** | 5.0970 | 4.6023   | 6.0836    |
| **Decision Tree**   | 5.5925  | 5.0756   | 6.0423    |

---

## **📌 Key AI and NumPy Concepts Used**

- **Regression Modeling** with `scikit-learn`
- **Cross-validation** (`KFold` and `cross_val_score`)
- **GridSearchCV** for hyperparameter tuning
- **MSE Calculation** for model evaluation
- **Data Normalization** using `MinMaxScaler`
- **NumPy** for manual matrix operations, reshaping, and vectorized math

---

## **📌 Sample Code Snippet**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Initialize model with best parameters
model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10)

# Cross-validation with MSE as scoring
mse_scores = -1 * cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

# Output average MSE
print("Average MSE:", mse_scores.mean())
