

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, classification_report, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from metrics_utils import evaluate_gnn
from metrics_utils import evaluate_gnn, set_seed
set_seed(42)

FILE_PATH = "datasets/SAML-D.csv"
N_ROWS = 500000


def load_and_prepare(n_rows=N_ROWS):
    print(f"Loading {n_rows} rows from SAML-D...")
    df = pd.read_csv(FILE_PATH, nrows=n_rows)
    print("ستون‌های موجود:", list(df.columns))


    cat_cols = [
        "Payment_currency", "Received_currency",
        "Sender_bank_location", "Receiver_bank_location", "Payment_type",
    ]
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    if "Payment_currency" in df.columns and "Received_currency" in df.columns:
        df["cross_currency"] = (df["Payment_currency"] != df["Received_currency"]).astype(int)
    if "Sender_bank_location" in df.columns and "Receiver_bank_location" in df.columns:
        df["cross_border"] = (df["Sender_bank_location"] != df["Receiver_bank_location"]).astype(int)

    return df


def get_feature_cols(df):
    candidates = [
        "Amount", "Payment_currency", "Received_currency",
        "Sender_bank_location", "Receiver_bank_location",
        "Payment_type", "cross_currency", "cross_border",
    ]
    return [c for c in candidates if c in df.columns]


def evaluate_binary(name, y_true, y_pred, y_prob):
    print(f"\n--- {name} ---")
    print(f"AUC        : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"PR-AUC     : {average_precision_score(y_true, y_prob):.4f}")
    print(f"F1         : {f1_score(y_true, y_pred):.4f}")
    print(f"Precision  : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall     : {recall_score(y_true, y_pred):.4f}")
    print(f"MCC        : {matthews_corrcoef(y_true, y_pred):.4f}")


def task_a_binary(df):
    print("\n" + "=" * 60)
    print("Task A - Binary: Is Laundering")
    print("=" * 60)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = df["Is_laundering"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos
    print(f"scale_pos_weight (Nneg/Npos): {scale_pos_weight:.2f}")

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    evaluate_binary("XGBoost - Is Laundering", y_test, y_pred, y_prob)


def task_b_type(df):
    print("\n" + "=" * 60)
    print("Task B - Multi-class: Laundering Type (فقط روی نمونه‌های illicit)")
    print("=" * 60)

    df_illicit = df[df["Is_laundering"] == 1].copy()
    print(f"تعداد نمونه‌های واقعا پول‌شویی: {len(df_illicit)}")

    feature_cols = get_feature_cols(df_illicit)

    le_target = LabelEncoder()
    y = le_target.fit_transform(df_illicit["Laundering_type"])
    X = df_illicit[feature_cols]

    print(f"تعداد انواع پول‌شویی: {len(le_target.classes_)}")
    print("انواع:", list(le_target.classes_))


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    model = XGBClassifier(
        n_estimators=100,
        objective="multi:softprob",
        num_class=len(le_target.classes_),
        tree_method="hist",
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(
        y_test, y_pred,
        labels=np.arange(len(le_target.classes_)),
        target_names=le_target.classes_,
        zero_division=0,
    ))


if __name__ == "__main__":
    df = load_and_prepare()
    task_a_binary(df)
    task_b_type(df)