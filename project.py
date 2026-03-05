import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("dataset.csv", header=1, index_col="ID")

# ==============================
# DATA INFO
# ==============================

st.title("Credit Risk Prediction System")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Info")
st.write(df.describe())

st.subheader("Missing Values")
st.write(df.isnull().sum())

# ==============================
# VISUALIZATION
# ==============================

st.subheader("Correlation Heatmap")

fig1 = plt.figure(figsize=(12,6))
sns.heatmap(df.corr(), cmap="coolwarm")
st.pyplot(fig1)

st.subheader("Default Distribution")

fig2 = plt.figure()
sns.countplot(x="default payment next month", data=df)
st.pyplot(fig2)

# ==============================
# FEATURES / TARGET
# ==============================

X = df.drop("default payment next month", axis=1)
y = df["default payment next month"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# SCALING
# ==============================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# SMOTE BALANCING
# ==============================

sm = SMOTE(random_state=42)

X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

# ==============================
# MODELS
# ==============================

lg = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
ada = AdaBoostClassifier()
xgb = XGBClassifier()
nb = GaussianNB()
svm = SVC(probability=True)

models = [lg, knn, rf, gb, ada, xgb, nb, svm]

results = []

for model in models:

    model.fit(X_train_smote, y_train_smote)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    results.append([model.__class__.__name__, accuracy, f1, roc_auc, recall, precision])

result_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "F1 Score", "ROC AUC", "Recall", "Precision"]
)

st.subheader("Model Comparison")
st.dataframe(result_df.sort_values(by="Accuracy", ascending=False))

# ==============================
# BEST MODEL
# ==============================

best_model_row = result_df.sort_values(by="Accuracy", ascending=False).iloc[0]
best_model_name = best_model_row["Model"]

model_mapping = {
    lg.__class__.__name__: lg,
    knn.__class__.__name__: knn,
    rf.__class__.__name__: rf,
    gb.__class__.__name__: gb,
    ada.__class__.__name__: ada,
    xgb.__class__.__name__: xgb,
    nb.__class__.__name__: nb,
    svm.__class__.__name__: svm,
}

best_model = model_mapping[best_model_name]

# SAVE MODEL
joblib.dump(best_model, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ==============================
# PREDICTION APP
# ==============================

st.subheader("Predict Credit Default")

age = st.number_input("Age", min_value=18, max_value=100)
income = st.number_input("Annual Income")
loan_amount = st.number_input("Loan Amount")
credit_utilization = st.number_input("Credit Utilization (%)")
payment_history = st.number_input("Number of Late Payments")

if st.button("Predict Default Risk"):

    input_data = np.array([[age, income, loan_amount, credit_utilization, payment_history]])

    input_scaled = scaler.transform(input_data)

    probability = best_model.predict_proba(input_scaled)[0][1]

    threshold = 0.4

    st.write(f"Default Probability: {probability:.2f}")

    if probability >= threshold:
        st.error(f"High Risk of Default ({probability:.2f})")
    else:
        st.success(f"Low Risk of Default ({probability:.2f})")