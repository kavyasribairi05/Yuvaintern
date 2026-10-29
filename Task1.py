
# YuvaIntern Internship Task
# Project: Data Preprocessing using Python, NumPy and Pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression

# STEP 1: Create a sample dataset
data = {
    "Age": [20, 21, 19, np.nan, 22, 20, 21, 19, 23, 20],
    "Gender": [
        "Female", "Male", "Female", "Male", np.nan,
        "Female", "Male", "Female", "Male", "Female"
    ],
    "Study_Hours": [4, 6, 3, 5, 7, np.nan, 6, 2, 8, 4],
    "Attendance": [85, 90, 75, 80, 95, 88, np.nan, 70, 96, 85],
    "Previous_Marks": [78, 85, 65, 72, 90, 80, 84, np.nan, 92, 78],
    "Final_Marks": [80, 88, 68, 75, 93, 82, 86, 65, 95, 80]
}

df = pd.DataFrame(data)

# STEP 2: Display the original dataset
print("\n--- Original Dataset ---")
print(df)

# STEP 3: Explore the dataset
print("\n--- Dataset Information ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# STEP 4: Remove duplicate records
df = df.drop_duplicates().copy()

print("\n--- Dataset After Removing Duplicates ---")
print(df)

# STEP 5: Handle missing values
# Fill numerical columns with their median
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with their mode
categorical_cols = df.select_dtypes(
    include=["object"]
).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n--- Dataset After Handling Missing Values ---")
print(df)

print("\nRemaining Missing Values:")
print(df.isnull().sum())

# STEP 6: Encode categorical variables
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

print("\n--- Dataset After Encoding Gender ---")
print(df)

# STEP 7: Exploratory Data Analysis (EDA)
print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

# Visualize the distribution of final marks
plt.figure(figsize=(7, 4))
sns.histplot(df["Final_Marks"], bins=5, kde=True)
plt.title("Distribution of Final Marks")
plt.xlabel("Final Marks")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# Visualize study hours vs final marks
plt.figure(figsize=(7, 4))
sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Final_Marks",
    hue="Gender"
)
plt.title("Study Hours vs Final Marks")
plt.tight_layout()
plt.show()

# STEP 8: Feature selection
X = df.drop(columns=["Final_Marks"])
y = df["Final_Marks"]

selector = SelectKBest(score_func=f_regression, k=3)
selector.fit(X, y)

selected_features = X.columns[selector.get_support()]

print("\n--- Selected Features ---")
print(selected_features.tolist())

# STEP 9: Normalize numerical features
# Keep encoded Gender unchanged
features_to_scale = [
    "Age",
    "Study_Hours",
    "Attendance",
    "Previous_Marks"
]

scaler = MinMaxScaler()
df[features_to_scale] = scaler.fit_transform(
    df[features_to_scale]
)

print("\n--- Normalized Dataset ---")
print(df.round(3))

# STEP 10: Save the cleaned dataset
df.to_csv("cleaned_student_dataset.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("Final dataset shape:", df.shape)
print("Preprocessing completed successfully.")
