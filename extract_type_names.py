"""
extract_type_names.py

Recovers the REAL laundering-type names (e.g. "Smurfing", "Layering") for
the numeric type ids used in samld_processed_v3.pt and in step25's output.

WHY A SEPARATE SCRIPT INSTEAD OF RE-RUNNING step18v3:
  The account-level type encoding in step18v3 (illicit_tx -> type_by_account
  -> LabelEncoder) depends ONLY on the raw CSV's Sender_account,
  Receiver_account, Is_laundering, and Laundering_type columns -- nothing
  about the graph, features, or splits. So it can be reproduced completely
  independently, without touching the already-built samld_processed_v3.pt
  or any of your 5 trained checkpoints. Re-running step18v3 itself would
  rebuild the entire graph from scratch and risk (however slim) being
  slightly out of sync with the checkpoints you already trained.

This script does NOT modify samld_processed_v3.pt. It only prints a
lookup table you can read alongside step25's output. If you'd rather have
it wired directly into step25's printed reports, that's a small follow-up
once you confirm this mapping looks right.

IMPORTANT: this must load the exact same first N_ROWS of the SAME CSV file
step18v3 used, or the mapping will not match. Keep N_ROWS = 1_000_000 as-is
unless you changed it in step18v3.
"""

import torch
from sklearn.preprocessing import LabelEncoder

FILE_PATH = "datasets/SAML-D.csv"
N_ROWS = 1_000_000  # must match step18v3 exactly
DATA_PATH = "samld_processed_v3.pt"
MIN_TRAIN_SAMPLES_PER_TYPE = 10  # must match step25's merge_rare_types default


def recover_type_names():
    """Exact replica of step18v3's type-encoding logic -- nothing else."""
    import pandas as pd

    print(f"Loading first {N_ROWS} rows of {FILE_PATH} (same as step18v3)...")
    df = pd.read_csv(FILE_PATH, nrows=N_ROWS)

    SENDER_COL, RECEIVER_COL = "Sender_account", "Receiver_account"
    LABEL_COL, TYPE_COL = "Is_laundering", "Laundering_type"

    illicit_tx = df[df[LABEL_COL] == 1]
    illicit_accounts = set(illicit_tx[SENDER_COL]).union(set(illicit_tx[RECEIVER_COL]))

    type_by_account = {}
    for acc in illicit_accounts:
        mask = (illicit_tx[SENDER_COL] == acc) | (illicit_tx[RECEIVER_COL] == acc)
        types = illicit_tx.loc[mask, TYPE_COL]
        if len(types) > 0:
            type_by_account[acc] = types.mode().iloc[0]

    le_type = LabelEncoder()
    all_types = sorted(set(type_by_account.values()))
    le_type.fit(all_types)

    id_to_name = {i: name for i, name in enumerate(le_type.classes_)}
    print(f"\nRecovered {len(id_to_name)} type names (should match num_types in the .pt file).")
    return id_to_name


def merge_rare_types(y_type, train_mask, min_train_samples=MIN_TRAIN_SAMPLES_PER_TYPE):
    """Identical to the version in step25 -- reproduced here so this script
    is self-contained and doesn't require importing step25."""
    from collections import Counter

    train_types = y_type[train_mask]
    train_types = train_types[train_types != -1].tolist()
    counts = Counter(train_types)

    all_type_ids = sorted(set(y_type[y_type != -1].tolist()))
    kept_type_ids = [t for t in all_type_ids if counts.get(t, 0) >= min_train_samples]
    rare_type_ids = [t for t in all_type_ids if t not in kept_type_ids]

    remap = {t: i for i, t in enumerate(kept_type_ids)}
    other_id = len(kept_type_ids)
    if rare_type_ids:
        for t in rare_type_ids:
            remap[t] = other_id
    return remap, kept_type_ids, rare_type_ids


if __name__ == "__main__":
    id_to_name = recover_type_names()

    print("\n=== Original type id -> real name ===")
    for i in sorted(id_to_name):
        print(f"  {i:2d}: {id_to_name[i]}")

    print("\nLoading samld_processed_v3.pt to compute the merge used by step25...")
    data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    y_type = data_dict["y_type"]
    train_mask = data_dict["train_mask"]

    remap, kept_type_ids, rare_type_ids = merge_rare_types(y_type, train_mask)

    print(f"\n=== Final classifier class id -> real name (what step25 reports as class N) ===")
    for old_id in kept_type_ids:
        new_id = remap[old_id]
        print(f"  class {new_id} = {id_to_name.get(old_id, '?')}")

    other_names = [id_to_name.get(t, "?") for t in rare_type_ids]
    other_class_id = len(kept_type_ids)
    print(f"  class {other_class_id} = Other  (merged: {', '.join(other_names)})")

    print("\nCopy this mapping next to step25's classification_report output when writing it up.")