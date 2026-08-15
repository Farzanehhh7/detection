import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from metrics_utils import evaluate_gnn, set_seed
set_seed(42)

FILE_PATH = "datasets/SAML-D.csv"


def explore_and_build_graph(n_rows=200000):
    print(f"--- Phase 1: Exploration (First {n_rows} rows) ---")
    df = pd.read_csv(FILE_PATH, nrows=n_rows)

    target_col = 'Is_laundering'
    type_col = 'Laundering_type'
    sender_col = 'Sender_account'
    receiver_col = 'Receiver_account'

    print("\nTop 10 Money Laundering Typologies:")
    print(df[type_col].value_counts().head(10))

    print("\nLaundering vs Normal Transactions:")
    print(df[target_col].value_counts())

    print("\n--- Phase 2: Graph Stats ---")
    all_accounts = np.unique(df[[sender_col, receiver_col]].values)

    src_indices = df[sender_col].factorize()[0]

    print(f"Total Nodes (Accounts): {len(all_accounts)}")
    print(f"Total Edges (Transactions): {len(df)}")

    le = LabelEncoder()
    df[type_col] = df[type_col].fillna('Normal')
    labels = le.fit_transform(df[type_col])
    print(f"Number of unique laundering patterns: {len(le.classes_)}")
    print("Sample of classes:", le.classes_[:5])

    return True


if __name__ == "__main__":
    explore_and_build_graph()