"""
step31_samld_coarse_family_classification.py

Coarser-grained typology classification: instead of distinguishing all
17 (merged to 10) individual types, group them into 3 semantically
meaningful families with far more samples each, and classify family
membership. Reuses the existing 5 binary checkpoints (no retraining)
and the structural features from step26.

FAMILY DEFINITIONS (based on the precise typology definitions researched
from Oztas et al. 2023):

  0 = structural/multi-hop  (topology of the flow matters most)
      Fan_In, Fan_Out, Layered_Fan_In, Layered_Fan_Out,
      Gather-Scatter, Scatter-Gather, Bipartite, Stacked Bipartite, Cycle

  1 = volume/threshold  (transaction size/count matters most)
      Structuring, Smurfing, Single_large, Cash_Withdrawal, Over-Invoicing

  2 = behavioral/temporal  (deviation from an account's own established
      pattern matters most)
      Behavioural_Change_1, Behavioural_Change_2, Deposit-Send

CAVEAT stated honestly: Deposit-Send's exact definition wasn't directly
confirmed in the source paper during research (unlike Layered_Fan_In and
Behavioural_Change, which were). It's placed in family 2 based on its
name ("deposit then send" = a behavioral/temporal pass-through pattern),
but this is an inference, not a confirmed citation -- worth revisiting
if a more precise definition turns up.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from metrics_utils import load_checkpoint
from step25_samld_type_classification import StructuralOnlyGraphSAGE, extract_embeddings
from step26_samld_structural_features import compute_structural_features

DATA_PATH = "samld_processed_v3.pt"
CHECKPOINT_PATTERN = "samld_seed_{seed}.pt"
SEEDS = (42, 1, 7, 123, 2024)

# original type id -> family id, using the id_to_name mapping recovered by extract_type_names.py:
# 0 Behavioural_Change_1, 1 Behavioural_Change_2, 2 Bipartite, 3 Cash_Withdrawal, 4 Cycle,
# 5 Deposit-Send, 6 Fan_In, 7 Fan_Out, 8 Gather-Scatter, 9 Layered_Fan_In, 10 Layered_Fan_Out,
# 11 Over-Invoicing, 12 Scatter-Gather, 13 Single_large, 14 Smurfing, 15 Stacked Bipartite, 16 Structuring
FAMILY_MAP = {
    2: 0, 4: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 12: 0, 15: 0,   # structural/multi-hop
    3: 1, 11: 1, 13: 1, 14: 1, 16: 1,                          # volume/threshold
    0: 2, 1: 2, 5: 2,                                          # behavioral/temporal
}
FAMILY_NAMES = {0: "Structural/Multi-hop", 1: "Volume/Threshold", 2: "Behavioral/Temporal"}


def build_family_labels(y_type):
    y_family = torch.full_like(y_type, -1)
    for orig_id, fam_id in FAMILY_MAP.items():
        y_family[y_type == orig_id] = fam_id
    return y_family


def fit_and_evaluate(X, y_family, train_mask, test_mask, seed_label=None, verbose=True):
    illicit_train = train_mask & (y_family != -1)
    illicit_test = test_mask & (y_family != -1)

    X_train, y_train = X[illicit_train.numpy()], y_family[illicit_train].numpy()
    X_test, y_test = X[illicit_test.numpy()], y_family[illicit_test].numpy()

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if verbose:
        labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
        target_names = [FAMILY_NAMES[i] for i in labels_present]
        print(f"\n--- seed={seed_label} ---")
        print(classification_report(y_test, y_pred, labels=labels_present,
                                     target_names=target_names, zero_division=0))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


if __name__ == "__main__":
    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y_type = data_dict["y_type"]
    train_mask = data_dict["train_mask"]
    test_mask = data_dict["test_mask"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_family = build_family_labels(y_type)
    print("\nFamily sample counts (train):")
    for fam_id, fam_name in FAMILY_NAMES.items():
        n = ((y_family == fam_id) & train_mask).sum().item()
        print(f"  {fam_name:25s}: {n}")
    print("Family sample counts (test):")
    for fam_id, fam_name in FAMILY_NAMES.items():
        n = ((y_family == fam_id) & test_mask).sum().item()
        print(f"  {fam_name:25s}: {n}")

    struct_features_raw, _ = compute_structural_features(edge_index, x.shape[0])
    illicit_train_np = (train_mask & (y_family != -1)).numpy()
    scaler = StandardScaler().fit(struct_features_raw[illicit_train_np])
    struct_features_scaled = scaler.transform(struct_features_raw)

    all_metrics = []
    for seed in SEEDS:
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
        load_checkpoint(model, CHECKPOINT_PATTERN.format(seed=seed), map_location=device)

        h2 = extract_embeddings(model, x, edge_index, device).numpy()
        X = np.concatenate([h2, struct_features_scaled], axis=1)

        m = fit_and_evaluate(X, y_family, train_mask, test_mask, seed_label=seed, verbose=True)
        all_metrics.append(m)
        print(f"  accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    print(f"\n\n{'=' * 60}\n=== Coarse family classifier -- summary across {len(SEEDS)} seeds ===\n{'=' * 60}")
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        vals = np.array([m[key] for m in all_metrics])
        print(f"  {key:12s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")

    print(f"\nFor comparison, the fine-grained (10-class) result was: macro_f1 = 0.1350 +/- 0.0203 (step26)")
