import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------------
# LOAD DATA
# -----------------------------------------

df = pd.read_csv("data/churn.csv")

print("data loaded\n")
print(df.head())

# ------------------------------------------
# SPLIT FEATURES
# ------------------------------------------

x = df.drop("churn", axis=1)
y = df["churn"]

# ------------------------------------------
# COLUMN TYPES
# ------------------------------------------

num_features = ["age", "tenure", "monthly_charges", "total_spent"]
cat_features = ["contract_type", "internet_service"]

# ------------------------------------------
# PREPROCESSING
# ------------------------------------------

preprocessor = ColumnTransformer(transformers = [("num", StandardScaler(), num_features), ("cat", OneHotEncoder(), cat_features)])

# -------------------------------------------
# PIPELINE MODELS
# -------------------------------------------

lr_pipeline = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression())])

rf_pipeline = Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier())])

# --------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

# --------------------------------------------
# TRAIN LOGISTIC
# --------------------------------------------

lr_pipeline.fit(x_train, y_train)
lr_pred = lr_pipeline.predict(x_test)

print("\nlogistic regression accuracy:", accuracy_score(y_test, lr_pred))

# ---------------------------------------------
# TRAIN RANDOM FOREST
# ---------------------------------------------

rf_pipeline.fit(x_train, y_train)
rf_pred = rf_pipeline.predict(x_test)

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# ----------------------------------------------
# FINAL REPORT 
# ----------------------------------------------

print("\nClassification report:\n")
print(classification_report(y_test, rf_pred))

# -----------------------------------------------
# ROC CURVE
# -----------------------------------------------

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# get probabilities
y_props = rf_pipeline.predict_proba(x_test)[:,1]

# ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_props)
auc_score = roc_auc_score(y_test, y_props)

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("\nAUC Score:", auc_score)

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------

import numpy as np

# extract trainded RF model
rf_model = rf_pipeline.named_steps["model"]

# get features names after encoding
ohe = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
encoded_features = ohe.get_feature_names_out(["contract_type", "internet_service"])

all_features = num_features + list(encoded_features)

importances = rf_model.feature_importances_

# sort features
indices = np.argsort(importances)[::-1]

print("\nFeature Importance:\n")
for i in indices:
    print(f"{all_features[i]}: {importances[i]:.4f}")