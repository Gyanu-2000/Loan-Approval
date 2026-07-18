"""
Loan approval model pipeline.

Loads loan.csv, cleans it, trains a Logistic Regression model, and exposes
everything the web app needs: metadata for building the form (categories,
numeric ranges), evaluation metrics, and a predict() function that mirrors
the math so the UI can show per-feature contributions.
"""

import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "loan.csv")
TARGET_COL = "Loan_Approved"
RANDOM_STATE = 42
TEST_SIZE = 0.2

ONE_HOT_COLS = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
    "Education_Level",
    TARGET_COL,
]


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


class LoanApprovalModel:
    """Trains once on startup and answers /api/model + /api/predict requests."""

    def __init__(self, path: str = DATA_PATH):
        self.path = path
        self._fit()

    def _fit(self):
        df = pd.read_csv(self.path)
        df = df.drop("Applicant_ID", axis=1)

        categorical_cols = df.select_dtypes(include=["object"]).columns
        numerical_cols = df.select_dtypes(include=["number"]).columns

        # --- impute missing values ---
        num_imp = SimpleImputer(strategy="mean")
        df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])
        cat_imp = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

        # --- metadata for the UI (captured before encoding) ---
        self.category_options = {
            col: sorted(df[col].unique().tolist())
            for col in categorical_cols
            if col != TARGET_COL
        }
        self.numerical_stats = {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
            }
            for col in numerical_cols
        }
        self.numerical_cols = list(numerical_cols)
        self.categorical_cols = [c for c in categorical_cols if c != TARGET_COL]
        self.n_rows = int(len(df))
        self.approved_pct = float((df[TARGET_COL] == "Yes").mean() * 100)

        # --- encode features (label encode target only, one-hot encode the rest
        #     from their original text categories so factor labels stay readable) ---
        le = LabelEncoder()
        df[TARGET_COL] = le.fit_transform(df[TARGET_COL])

        ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        encoded = ohe.fit_transform(df[ONE_HOT_COLS])
        encoded_df = pd.DataFrame(
            encoded, columns=ohe.get_feature_names_out(ONE_HOT_COLS), index=df.index
        )
        df_enc = pd.concat([df.drop(columns=ONE_HOT_COLS), encoded_df], axis=1)

        target_col_encoded = f"{TARGET_COL}_1"
        X = df_enc.drop([target_col_encoded], axis=1)
        y = df_enc[target_col_encoded]

        self.feature_names = list(X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
        }
        self.confusion_matrix = confusion_matrix(y_test, y_pred).tolist()

        self.scaler_mean = scaler.mean_.tolist()
        self.scaler_scale = scaler.scale_.tolist()
        self.coef = model.coef_[0].tolist()
        self.intercept = float(model.intercept_[0])

    def friendly_one_hot(self, name: str):
        """'Employment_Status_Self-employed' -> ('Employment_Status', 'Self-employed')."""
        for col in self.categorical_cols:
            if name.startswith(col + "_"):
                return col, name[len(col) + 1 :]
        return None, name

    def as_artifact(self) -> dict:
        """Everything the frontend needs to build the form and show global stats."""
        return {
            "feature_names": self.feature_names,
            "coef": self.coef,
            "intercept": self.intercept,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "metrics": self.metrics,
            "confusion_matrix": self.confusion_matrix,
            "categorical_cols": self.categorical_cols,
            "numerical_cols": self.numerical_cols,
            "category_options": self.category_options,
            "numerical_stats": self.numerical_stats,
            "ohe_categories": self.category_options,
            "n_rows": self.n_rows,
            "approved_pct": self.approved_pct,
        }

    def predict(self, state: dict) -> dict:
        """state: {feature_or_category_col: value}. Returns probability + contributions."""
        raw = []
        for fname in self.feature_names:
            if fname in self.numerical_cols:
                raw.append(float(state.get(fname, 0)))
            else:
                col, cat = self.friendly_one_hot(fname)
                raw.append(1.0 if state.get(col) == cat else 0.0)

        z = self.intercept
        contributions = []
        for i, v in enumerate(raw):
            zscore = (v - self.scaler_mean[i]) / self.scaler_scale[i]
            contrib = zscore * self.coef[i]
            z += contrib
            contributions.append({"name": self.feature_names[i], "contrib": contrib})

        p = sigmoid(z)
        return {"probability": p, "contributions": contributions}
