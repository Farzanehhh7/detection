

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

from metrics_utils import load_checkpoint, paired_significance_test
from step25_samld_type_classification import (
    StructuralOnlyGraphSAGE, merge_rare_types, extract_embeddings, load_type_names,
)
from step26_samld_structural_features import compute_structural_features
from sklearn.preprocessing import StandardScaler

DATA_PATH = "samld_processed_v3.pt"
BEHAVIOURAL_PATH = "samld_behavioural_features.pt"
CHECKPOINT_PATTERN = "samld_seed_{seed}.pt"
SEEDS = (42, 1, 7, 123, 2024)

# original type ids for the two classes we actually care about here
BEHAVIOURAL_CHANGE_1_ID = 0
BEHAVIOURAL_CHANGE_2_ID = 1


def fit_and_evaluate(X, y_type_remapped, train_mask, test_mask, remap, seed_label=None, verbose=True):
    illicit_train = train_mask & (y_type_remapped != -1)
    illicit_test = test_mask & (y_type_remapped != -1)

    X_train, y_train = X[illicit_train.numpy()], y_type_remapped[illicit_train].numpy()
    X_test, y_test = X[illicit_test.numpy()], y_type_remapped[illicit_test].numpy()

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # per-class F1 for BC1/BC2 specifically, via their REMAPPED class ids
    bc1_new_id = remap.get(BEHAVIOURAL_CHANGE_1_ID)
    bc2_new_id = remap.get(BEHAVIOURAL_CHANGE_2_ID)
    per_class = f1_score(y_test, y_pred, average=None, zero_division=0,
                          labels=sorted(set(y_test.tolist()) | set(y_pred.tolist())))
    labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    label_to_f1 = dict(zip(labels_present, per_class))

    bc1_f1 = label_to_f1.get(bc1_new_id, None)
    bc2_f1 = label_to_f1.get(bc2_new_id, None)

    if verbose:
        print(f"  seed={seed_label}: macro_f1={macro_f1:.4f}  "
              f"BC1_f1={bc1_f1}  BC2_f1={bc2_f1}")

    return {"macro_f1": macro_f1, "bc1_f1": bc1_f1, "bc2_f1": bc2_f1}


if __name__ == "__main__":
    print("Loading data + new behavioral features...")
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data["x"]
    edge_index = data["edge_index"]
    y_type = data["y_type"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]

    behavioural = torch.load(BEHAVIOURAL_PATH, map_location="cpu", weights_only=False)
    behavioural_features = behavioural["features_scaled"].numpy()
    print(f"Behavioral features shape: {behavioural_features.shape}  "
          f"names: {behavioural['feature_names']}")

    device = torch.device("cpu")
    remap, num_new_types, _ = merge_rare_types(y_type, train_mask)
    y_type_remapped = y_type.clone()
    for old_id, new_id in remap.items():
        y_type_remapped[y_type == old_id] = new_id

    struct_raw, _ = compute_structural_features(edge_index, x.shape[0])
    illicit_train_np = (train_mask & (y_type_remapped != -1)).numpy()
    struct_scaler = StandardScaler().fit(struct_raw[illicit_train_np])
    struct_scaled = struct_scaler.transform(struct_raw)

    baseline_metrics, enhanced_metrics = [], []
    for seed in SEEDS:
        print(f"\nSeed {seed}:")
        model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
        load_checkpoint(model, CHECKPOINT_PATTERN.format(seed=seed), map_location=device)
        h2 = extract_embeddings(model, x, edge_index, device).numpy()

        X_baseline = np.concatenate([h2, struct_scaled], axis=1)
        X_enhanced = np.concatenate([h2, struct_scaled, behavioural_features], axis=1)

        print(" baseline (embedding + structural, no behavioral):")
        m_base = fit_and_evaluate(X_baseline, y_type_remapped, train_mask, test_mask, remap, seed_label=seed)
        baseline_metrics.append(m_base)

        print(" enhanced (+ behavioral features):")
        m_enh = fit_and_evaluate(X_enhanced, y_type_remapped, train_mask, test_mask, remap, seed_label=seed)
        enhanced_metrics.append(m_enh)

    print(f"\n\n{'=' * 70}\n=== Summary across {len(SEEDS)} seeds ===\n{'=' * 70}")
    for label, metrics in [("BASELINE", baseline_metrics), ("ENHANCED (+behavioral)", enhanced_metrics)]:
        macro = np.array([m["macro_f1"] for m in metrics])
        bc1 = np.array([m["bc1_f1"] for m in metrics if m["bc1_f1"] is not None])
        bc2 = np.array([m["bc2_f1"] for m in metrics if m["bc2_f1"] is not None])
        print(f"\n{label}:")
        print(f"  macro_f1: {macro.mean():.4f} +/- {macro.std(ddof=1):.4f}")
        print(f"  Behavioural_Change_1 F1: {bc1.mean():.4f} +/- {bc1.std(ddof=1):.4f}" if len(bc1)
              else "  Behavioural_Change_1: not present as its own class this run")
        print(f"  Behavioural_Change_2 F1: {bc2.mean():.4f} +/- {bc2.std(ddof=1):.4f}" if len(bc2)
              else "  Behavioural_Change_2: not present as its own class this run")

    # ------------------------------------------------------------------
    # Added: paired significance test, baseline vs enhanced, same 5 seeds.
    # Without this, "enhanced looks a bit higher" has no statistical backing.
    # ------------------------------------------------------------------
    print(f"\n\n{'=' * 70}\n=== Paired significance test: baseline vs enhanced ({len(SEEDS)} seeds) ===\n{'=' * 70}")

    macro_base = [m["macro_f1"] for m in baseline_metrics]
    macro_enh = [m["macro_f1"] for m in enhanced_metrics]
    paired_significance_test(macro_enh, macro_base, "Enhanced (macro-F1)", "Baseline (macro-F1)")

    bc1_pairs = [(b["bc1_f1"], e["bc1_f1"]) for b, e in zip(baseline_metrics, enhanced_metrics)
                 if b["bc1_f1"] is not None and e["bc1_f1"] is not None]
    if len(bc1_pairs) >= 2:
        bc1_base_vals, bc1_enh_vals = zip(*[(b, e) for b, e in bc1_pairs])
        paired_significance_test(list(bc1_enh_vals), list(bc1_base_vals),
                                  "Enhanced (BC1 F1)", "Baseline (BC1 F1)")
    else:
        print("\nNot enough seeds where Behavioural_Change_1 appeared in both baseline and "
              "enhanced test predictions to run a paired test (need >=2).")

    bc2_pairs = [(b["bc2_f1"], e["bc2_f1"]) for b, e in zip(baseline_metrics, enhanced_metrics)
                 if b["bc2_f1"] is not None and e["bc2_f1"] is not None]
    if len(bc2_pairs) >= 2:
        bc2_base_vals, bc2_enh_vals = zip(*[(b, e) for b, e in bc2_pairs])
        paired_significance_test(list(bc2_enh_vals), list(bc2_base_vals),
                                  "Enhanced (BC2 F1)", "Baseline (BC2 F1)")
    else:
        print("\nNot enough seeds where Behavioural_Change_2 appeared in both baseline and "
              "enhanced test predictions to run a paired test (need >=2).")