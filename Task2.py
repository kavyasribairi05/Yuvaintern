
# YuvaIntern Internship Task
# Project: Regression and Classification using Scikit-learn

import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Store all evaluation results
results = []

# ==================================================
# PART 1: REGRESSION MODELS
# ==================================================

print("\n========== REGRESSION MODELS ==========")

# Load the sample diabetes regression dataset
reg_data = load_diabetes(as_frame=True)

X_reg = reg_data.data
y_reg = reg_data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# Regression algorithms
regression_models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(
        max_depth=4, random_state=42
    ),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

for name, model in regression_models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Predict test-set values
    predictions = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Task": "Regression",
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2_Score": round(r2, 3),
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1_Score": np.nan
    })

    print(f"\nModel: {name}")
    print("Sample Predictions:", np.round(predictions[:5], 2))
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))


# ==================================================
# PART 2: CLASSIFICATION MODELS
# ==================================================

print("\n========== CLASSIFICATION MODELS ==========")

# Load the sample breast cancer classification dataset
cls_data = load_breast_cancer(as_frame=True)

X_cls = cls_data.data
y_cls = cls_data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_cls,
    y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

# Classification algorithms
classification_models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)
    ),
    "Decision Tree Classifier": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "K-Nearest Neighbors": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )
}

for name, model in classification_models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Predict test-set classes
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Task": "Classification",
        "Model": name,
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2_Score": np.nan,
        "Accuracy": round(accuracy, 3),
        "Precision": round(precision, 3),
        "Recall": round(recall, 3),
        "F1_Score": round(f1, 3)
    })

    print(f"\nModel: {name}")
    print("Sample Predictions:", predictions[:5])
    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("F1 Score:", round(f1, 3))


# ==================================================
# PART 3: COMPARE AND SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# Save evaluation metrics to a CSV file
results_df.to_csv("model_comparison.csv", index=False)

print("\nResults saved to model_comparison.csv")

# Display the best model in each task using R2 or F1
reg_results = results_df[
    results_df["Task"] == "Regression"
]
best_reg = reg_results.loc[
    reg_results["R2_Score"].idxmax()
]

cls_results = results_df[
    results_df["Task"] == "Classification"
]
best_cls = cls_results.loc[
    cls_results["F1_Score"].idxmax()
]

print("\nBest regression model by R2 Score:")
print(best_reg["Model"], "-", best_reg["R2_Score"])

print("\nBest classification model by F1 Score:")
print(best_cls["Model"], "-", best_cls["F1_Score"])

print("\nAll models trained and evaluated successfully!")
