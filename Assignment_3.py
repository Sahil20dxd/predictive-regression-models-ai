"""Regression Analysis Script for COMP10200.

This script performs regression analysis on a dataset using various machine learning algorithms.
It includes cross-validation, hyperparameter tuning using GridSearchCV, and evaluation of models
based on Mean Squared Error (MSE).

Author: [Your Name]
Student Number: [Your Student Number]
Date: [Date]
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Load and shuffle the data
data = pd.read_csv("train_dataset.csv")
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Output the size of the dataset and number of features
print("Dataset Size:", data.shape[0])
print("Number of Features:", X.shape[1])

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, model_name, X, y, cv=5):
    """Evaluate a regression model using cross-validation and print the results.

    Parameters:
    - model: The regression model to evaluate.
    - model_name: A string representing the name of the model.
    - X: The feature matrix.
    - y: The target vector.
    - cv: The number of folds for cross-validation (default is 5).

    Output:
    - Prints the average, maximum, and minimum MSE for the model.
    """
    results = cross_validate(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    mse_scores = -results["test_score"]
    print(f"{model_name} - Average MSE: {mse_scores.mean():.4f}, Max MSE: {mse_scores.max():.4f}, Min MSE: {mse_scores.min():.4f}")

## Linear Regression
linear_reg = LinearRegression()
evaluate_model(linear_reg, "Linear Regression", X_train, y_train)

## K-Nearest Neighbors (K-NN)
knn_reg = KNeighborsRegressor(n_neighbors=5, weights="uniform", p=2)
evaluate_model(knn_reg, "K-Nearest Neighbors (K-NN)", X_train, y_train)

## Decision Tree
tree_reg = DecisionTreeRegressor(max_depth=5, min_samples_split=2, random_state=42)
evaluate_model(tree_reg, "Decision Tree", X_train, y_train)

## Random Forest
forest_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
evaluate_model(forest_reg, "Random Forest", X_train, y_train)

## Support Vector Regression (SVR)
svr_reg = SVR(kernel="rbf", C=1.0, gamma="scale")
evaluate_model(svr_reg, "Support Vector Regression (SVR)", X_train, y_train)

## Grid Search for K-NN
params_knn = {
    "n_neighbors": [3, 5, 7, 10],
    "weights": ["uniform", "distance"],
    "p": [1, 2]
}
grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid=params_knn, scoring="neg_mean_squared_error", cv=5)
grid_knn.fit(X_train, y_train)
print("\nBest Parameters for K-NN:", grid_knn.best_params_)
print("Best MSE for K-NN:", -grid_knn.best_score_)

## Grid Search for Random Forest
params_rf = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10]
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid=params_rf, scoring="neg_mean_squared_error", cv=5)
grid_rf.fit(X_train, y_train)
print("\nBest Parameters for Random Forest:", grid_rf.best_params_)
print("Best MSE for Random Forest:", -grid_rf.best_score_)

## Grid Search for SVR
params_svr = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1.0, 10.0],
    "gamma": ["scale", "auto"]
}
grid_svr = GridSearchCV(SVR(), param_grid=params_svr, scoring="neg_mean_squared_error", cv=5)
grid_svr.fit(X_train, y_train)
print("\nBest Parameters for SVR:", grid_svr.best_params_)
print("Best MSE for SVR:", -grid_svr.best_score_)

print("\n\n***** End of Regression Analysis Script *****\n\n")