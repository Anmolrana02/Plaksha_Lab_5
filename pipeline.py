"""Lab 5 — shared pipeline building blocks.

Built across Day 1 (Steps 5-6: the split itself is done in the notebook, not
here) and Day 2 (Steps 1-3: preprocessing, the full pipeline, feature
engineering).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(num_cols, cat_cols):
    """ColumnTransformer: numeric -> impute(median) + scale;
    categorical -> impute(most_frequent) + one-hot. handle_unknown='ignore'
    on the encoder so a category seen only at predict time doesn't crash."""
    numeric_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])
    return preprocessor


def build_pipeline(num_cols, cat_cols, model):
    """Preprocessing + a model as one fittable Pipeline, so every fitted step
    gets refit from scratch on each CV fold's training portion only."""
    preprocessor = build_preprocessor(num_cols, cat_cols)
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])


def extract_title(name):
    """'Smith, Mr. John' -> 'Mr'. Returns 'Unknown' if no title is found."""
    if pd.isna(name):
        return "Unknown"
    parts = name.split(",")
    if len(parts) < 2:
        return "Unknown"
    title_part = parts[1].split(".")[0].strip()
    return title_part if title_part else "Unknown"


def engineer(df):
    """Add engineered features to a copy of df and return the copy.

    - family_size = SibSp + Parch + 1 (self); is_alone = family_size == 1.
      Hypothesis: solo travelers and very large families survived at
      different rates than small families (small families could help each
      other on the way to a boat; solo travelers had no one, and very large
      families were slower to move together).
    - title, extracted from Name.
      Hypothesis: title carries social status and age/gender information
      beyond what Pclass/Sex/Age capture alone (e.g. 'Master' flags young
      boys, who were prioritized, even in the ~20% of rows where Age is
      missing).
    """
    df = df.copy()
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["title"] = df["Name"].apply(extract_title)
    # Collapse rare titles so one-hot encoding doesn't create many
    # near-empty columns.
    common = {"Mr", "Miss", "Mrs", "Master"}
    df["title"] = df["title"].where(df["title"].isin(common), "Rare")
    return df