
# TASK 3: Clustering Techniques and Model Evaluation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

# ==========================================
# PART 1: CLUSTERING USING K-MEANS
# ==========================================

print("\n===== CLUSTERING =====")

iris = load_iris()
X_cluster = iris.data

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_score = silhouette_score(X_scaled, kmeans_labels)

print("\nK-Means Silhouette Score:", round(kmeans_score, 3))

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X_scaled)

hierarchical_score = silhouette_score(
    X_scaled, hierarchical_labels
)

print("Hierarchical Clustering Silhouette Score:",
      round(hierarchical_score, 3))

# Save clustering comparison
clustering_results = pd.DataFrame({
    "Algorithm": ["K-Means", "Hierarchical Clustering"],
    "Silhouette_Score": [kmeans_score, hierarchical_score]
})

clustering_results.to_csv("clustering_results.csv", index=False)

print("\nClustering Comparison:")
print(clustering_results.round(3).to_string(index=False))

# ==========================================
# PART 2: DIMENSIONALITY REDUCTION USING PCA
# ==========================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA Explained Variance Ratio:",
      np.round(pca.explained_variance_ratio_, 3))

print("Total variance retained:",
      round(sum(pca.explained_variance_ratio_), 3))

plt.figure(figsize=(7, 5))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="viridis"
)
plt.title("K-Means Clusters Visualized Using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("clustering_pca.png")
plt.show()

# ==========================================
# PART 3: CLASSIFICATION AND CROSS-VALIDATION
# ==========================================

print("\n===== CLASSIFICATION EVALUATION =====")

cancer = load_breast_cancer()
X = cancer.data

# Make malignant cases the positive class (1)
y = (cancer.target == 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Pipeline scales data and trains Random Forest
# Scaling is included here for a consistent workflow.
pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)

# Cross-validation on training data
cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross-validation accuracy scores:",
      np.round(cv_scores, 3))
print("Mean cross-validation accuracy:",
      round(cv_scores.mean(), 3))

# ==========================================
# PART 4: HYPERPARAMETER TUNING
# ==========================================

param_grid = {
    "randomforestclassifier__n_estimators": [100, 200],
    "randomforestclassifier__max_depth": [None, 5, 10]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("Best Cross-validation ROC-AUC:",
      round(grid_search.best_score_, 3))

# ==========================================
# PART 5: TEST SET EVALUATION
# ==========================================

predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Benign", "Malignant"],
    zero_division=0
))

print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1, 3))
print("ROC-AUC:", round(roc_auc, 3))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["Benign", "Malignant"])
plt.yticks([0, 1], ["Benign", "Malignant"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Save evaluation metrics
evaluation = pd.DataFrame([{
    "Model": "Tuned Random Forest",
    "CV_Accuracy_Mean": cv_scores.mean(),
    "Best_CV_ROC_AUC": grid_search.best_score_,
    "Test_Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1,
    "ROC_AUC": roc_auc
}])

evaluation.to_csv("classification_evaluation.csv", index=False)

print("\nEvaluation Results:")
print(evaluation.round(3).to_string(index=False))

print("\nAll tasks completed successfully!")
print("Files saved:")
print("1. clustering_results.csv")
print("2. classification_evaluation.csv")
print("3. clustering_pca.png")
print("4. confusion_matrix.png")
