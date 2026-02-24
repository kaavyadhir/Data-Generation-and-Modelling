import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def train_models(df):

    # Create results folder
    os.makedirs("../results", exist_ok=True)

    # ==========================
    # Use ONLY initial parameters (No Data Leakage)
    # ==========================
    X = df[["beta", "gamma", "initial_infected"]]
    y = df["severity"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features (for some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================
    # Define 8 Classification Models
    # ==========================
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB()
    }

    results = []

    # ==========================
    # Train & Evaluate
    # ==========================
    for name, model in models.items():

        start_time = time.time()

        # Models requiring scaling
        if name in ["SVM (RBF)", "KNN (k=5)", "Logistic Regression"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        train_time = time.time() - start_time

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(roc, 4),
            "Train_Time_sec": round(train_time, 3)
        })

    # ==========================
    # Create Comparison Table
    # ==========================
    results_df = pd.DataFrame(results)

    # Sort by F1 Score (descending)
    results_df = results_df.sort_values("F1_Score", ascending=False).reset_index(drop=True)

    # Assign integer ranks (1,2,3...)
    results_df["Rank"] = range(1, len(results_df) + 1)

    # Save full comparison table
    results_df.to_csv("../results/model_comparison.csv", index=False)

    # Save best model separately
    best_model = results_df.iloc[0]
    best_model.to_frame().T.to_csv("../results/best_model.csv", index=False)

    return results_df