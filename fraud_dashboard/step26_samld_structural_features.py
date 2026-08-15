"""
step26_samld_structural_features.py

Adds explicit graph-structural features -- each one tied directly to a
real typology definition -- and compares type-classification accuracy
against the embedding-only baseline from step25, head-to-head, on the
same 5 seeds.

WHY THESE SPECIFIC FEATURES (see the typology reference table):
  in_degree            -> Fan_In strength (many distinct senders)
  out_degree           -> Fan_Out strength (many distinct receivers)
  gather_scatter_score -> min(in_degree, out_degree): high when an account
                           both gathers AND scatters funds
  degree_ratio         -> out_degree / (in_degree+1): skew toward
                           fan-out-heavy vs fan-in-heavy behavior
  reciprocal_count     -> number of neighbors with an edge in BOTH
                           directions: a cheap, real proxy for Cycle
                           involvement (a full k-hop cycle search is much
                           more expensive; this catches the 2-hop case)
  is_pass_through      -> binary flag: in_degree>=2 AND out_degree>=2,
                           reinforcing the Gather-Scatter / Stacked-
                           Bipartite signal

LIMITATION, stated honestly: a true Structuring/Smurfing signal needs
per-transaction amounts relative to a reporting threshold, which isn't
available at this account-aggregate level -- only sent_amount_mean/count
proxies already in the original 8 features. Not added here; flagged as a
possible future feature if per-transaction data gets pulled in later.

This script does NOT retrain the binary GNN. It reuses the same 5
checkpoints from step22 and the same merge_rare_types()/embedding
extraction machinery from step25.
"""

import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from metrics_utils import load_checkpoint
from step25_samld_type_classification import (
    StructuralOnlyGraphSAGE, merge_rare_types, extract_embeddings, load_type_names,
)

DATA_PATH = "samld_processed_v3.pt"
CHECKPOINT_PATTERN = "samld_seed_{seed}.pt"
SEEDS = (42, 1, 7, 123, 2024)


# ---------------------------------------------------------------------------
# The structural features themselves
# ---------------------------------------------------------------------------

def compute_structural_features(edge_index, num_nodes):
    t0 = time.time()
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()

    in_degree = np.bincount(dst, minlength=num_nodes).astype(np.float32)
    out_degree = np.bincount(src, minlength=num_nodes).astype(np.float32)

    edge_set = set(zip(src.tolist(), dst.tolist()))
    reciprocal_mask = np.fromiter(
        ((d, s) in edge_set for s, d in zip(src.tolist(), dst.tolist())),
        dtype=bool, count=len(src),
    )
    recip_src = src[reciprocal_mask]
    recip_dst = dst[reciprocal_mask]
    reciprocal_count = (np.bincount(recip_src, minlength=num_nodes) +
                         np.bincount(recip_dst, minlength=num_nodes)).astype(np.float32)

    gather_scatter_score = np.minimum(in_degree, out_degree)
    degree_ratio = out_degree / (in_degree + 1.0)
    is_pass_through = ((in_degree >= 2) & (out_degree >= 2)).astype(np.float32)

    features = np.stack(
        [in_degree, out_degree, gather_scatter_score, degree_ratio, reciprocal_count, is_pass_through],
        axis=1,
    )
    print(f"Structural features computed in {time.time() - t0:.2f}s   shape={features.shape}")
    return features, [
        "in_degree", "out_degree", "gather_scatter_score",
        "degree_ratio", "reciprocal_count", "is_pass_through",
    ]


# ---------------------------------------------------------------------------
# Fit + evaluate helper, shared for both the baseline and enhanced runs
# ---------------------------------------------------------------------------

def fit_and_evaluate(X, y_type_remapped, train_mask, test_mask, type_names=None,
                      seed_label=None, run_label="", verbose=True):
    illicit_train = train_mask & (y_type_remapped != -1)
    illicit_test = test_mask & (y_type_remapped != -1)

    X_train, y_train = X[illicit_train.numpy()], y_type_remapped[illicit_train].numpy()
    X_test, y_test = X[illicit_test.numpy()], y_type_remapped[illicit_test].numpy()

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if verbose:
        labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
        target_names = [type_names.get(i, f"type_{i}") for i in labels_present] if type_names else None
        print(f"\n--- {run_label} (seed={seed_label}) ---")
        print(classification_report(y_test, y_pred, labels=labels_present,
                                     target_names=target_names, zero_division=0))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


def summarize(all_metrics, label):
    print(f"\n{label}:")
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        vals = np.array([m[key] for m in all_metrics])
        print(f"  {key:12s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")
    return {key: (np.mean([m[key] for m in all_metrics]), np.std([m[key] for m in all_metrics], ddof=1))
            for key in ("accuracy", "macro_f1", "weighted_f1")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y_type = data_dict["y_type"]
    train_mask = data_dict["train_mask"]
    test_mask = data_dict["test_mask"]
    num_nodes = x.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    remap, num_new_types, kept_type_ids = merge_rare_types(y_type, train_mask)
    y_type_remapped = y_type.clone()
    for old_id, new_id in remap.items():
        y_type_remapped[y_type == old_id] = new_id

    type_names = load_type_names()
    if type_names:
        print(f"Loaded {len(type_names)} real type names.")

    struct_features_raw, struct_feature_names = compute_structural_features(edge_index, num_nodes)
    print("Structural feature columns:", struct_feature_names)

    # scale structural features once (train-illicit-only fit, applied to all)
    illicit_train_np = (train_mask & (y_type_remapped != -1)).numpy()
    scaler = StandardScaler().fit(struct_features_raw[illicit_train_np])
    struct_features_scaled = scaler.transform(struct_features_raw)

    baseline_metrics, enhanced_metrics = [], []

    for seed in SEEDS:
        ckpt_path = CHECKPOINT_PATTERN.format(seed=seed)
        print(f"\n{'=' * 60}\nSeed {seed}: loading {ckpt_path}\n{'=' * 60}")

        model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
        load_checkpoint(model, ckpt_path, map_location=device)

        h2 = extract_embeddings(model, x, edge_index, device).numpy()

        # baseline: embedding only (exactly step25)
        m_base = fit_and_evaluate(
            h2, y_type_remapped, train_mask, test_mask, type_names=type_names,
            seed_label=seed, run_label="BASELINE (embedding only)", verbose=False,
        )
        baseline_metrics.append(m_base)

        # enhanced: embedding + structural features
        X_enhanced = np.concatenate([h2, struct_features_scaled], axis=1)
        m_enh = fit_and_evaluate(
            X_enhanced, y_type_remapped, train_mask, test_mask, type_names=type_names,
            seed_label=seed, run_label="ENHANCED (embedding + structural features)", verbose=True,
        )
        enhanced_metrics.append(m_enh)

        print(f"  baseline : acc={m_base['accuracy']:.4f}  macro_f1={m_base['macro_f1']:.4f}")
        print(f"  enhanced : acc={m_enh['accuracy']:.4f}  macro_f1={m_enh['macro_f1']:.4f}")

    print(f"\n\n{'=' * 60}\n=== FINAL COMPARISON across {len(SEEDS)} seeds ===\n{'=' * 60}")
    base_summary = summarize(baseline_metrics, "BASELINE (embedding only, step25)")
    enh_summary = summarize(enhanced_metrics, "ENHANCED (embedding + structural features)")

    print("\n--- Delta (enhanced - baseline) ---")
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        delta = enh_summary[key][0] - base_summary[key][0]
        print(f"  {key:12s}: {delta:+.4f}")

    # quick paired significance check on macro_f1 across the 5 seeds
    from scipy import stats
    base_vals = [m["macro_f1"] for m in baseline_metrics]
    enh_vals = [m["macro_f1"] for m in enhanced_metrics]
    t_stat, p_val = stats.ttest_rel(enh_vals, base_vals)
    print(f"\nPaired t-test on macro_f1 (enhanced vs baseline), df={len(SEEDS)-1}: "
          f"t={t_stat:.3f}  p={p_val:.4f}")
    if p_val < 0.05:
        print("-> statistically significant at the usual 0.05 level.")
    else:
        print("-> NOT statistically significant with only 5 seeds -- treat the improvement "
              "(if any) as suggestive, not conclusive, exactly like the lr/hidden search lesson.")
