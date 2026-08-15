"""
step28_samld_multitask_fixed.py

Fixed, properly-tested version of the multi-task joint-training idea:
one shared GraphSAGE backbone, two heads (binary illicit/licit + type),
trained together instead of the two-stage approach in step25/step26.

THREE BUGS FOUND IN REVIEW, ALL FIXED HERE:

1. NaN logging bug. `total_type_loss += loss_type.item()` accumulated
   every batch, including "all-ignored" batches (no illicit account
   happened to land in that batch -- likely with only 275 illicit
   accounts spread across 174k train nodes). CrossEntropyLoss(ignore_index=-1)
   returns nan for the scalar loss VALUE on an all-ignored batch (0/0),
   even though its backward-pass gradient is genuinely zero for every
   parameter (verified directly with a reproduction script -- the shared
   backbone and binary head are NOT corrupted). Fixed here by only
   accumulating the type-loss average over batches that had >=1 valid
   (non -1) target, using a separate counter.

2. Merge-rule inconsistency. The original script merged only types 11
   and 12 (chosen by <5 TEST samples). This script uses the exact same
   train<10 rule as step23/step25 (merge_rare_types), so results are
   directly comparable to the two-stage baseline -- same 10 final classes.

3. Missing class weighting on the type head. type_criterion had no
   weight parameter despite the same severe per-type imbalance the
   binary head is weighted for. Fixed with sklearn's balanced class
   weights computed on the merged type distribution (train only),
   capped the same conservative way the binary weight is.

ALSO: runs the full 5-seed protocol (42, 1, 7, 123, 2024), not the
original single seed=42, so the comparison against the two-stage
baseline (step25/step26) is apples-to-apples: mean +/- std both sides.
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from metrics_utils import set_seed, find_best_threshold, report_at_percentile_thresholds
from step25_samld_type_classification import merge_rare_types, load_type_names

DATA_PATH = "samld_processed_v3.pt"
S1, S2 = 25, 10
BATCH_SIZE = 256
WEIGHT_CAP = 30.0
TYPE_WEIGHT_CAP = 10.0
EPOCHS = 20
SEEDS = (42, 1, 7, 123, 2024)
TYPE_LOSS_WEIGHT = 0.5


class MultiTaskGraphSAGE(nn.Module):
    """Shared backbone, two heads: binary and type."""

    def __init__(self, in_channels, hidden_channels, num_type_classes):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.binary_head = nn.Linear(hidden_channels, 2)
        self.type_head = nn.Linear(hidden_channels, num_type_classes)

    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), p=0.3, training=self.training)
        return self.binary_head(h), self.type_head(h)


def sample_k(neighbors, k, rng):
    return neighbors if len(neighbors) <= k else rng.sample(neighbors, k)


def build_batch_subgraph(seed_nodes, adjacency, x, rng, s1=S1, s2=S2):
    hop1_of, all_hop1 = {}, set()
    for s in seed_nodes:
        neighs = sample_k(adjacency.get(s, []), s1, rng)
        hop1_of[s] = neighs
        all_hop1.update(neighs)
    hop2_of, all_hop2 = {}, set()
    for n in all_hop1:
        neighs = sample_k(adjacency.get(n, []), s2, rng)
        hop2_of[n] = neighs
        all_hop2.update(neighs)
    all_nodes = list(set(seed_nodes) | all_hop1 | all_hop2)
    local_id = {n: i for i, n in enumerate(all_nodes)}
    edges_src, edges_dst = [], []
    for s in seed_nodes:
        for n in hop1_of[s]:
            edges_src.append(local_id[n]); edges_dst.append(local_id[s])
    for n in all_hop1:
        for n2 in hop2_of[n]:
            edges_src.append(local_id[n2]); edges_dst.append(local_id[n])
    node_idx_tensor = torch.tensor(all_nodes, dtype=torch.long)
    sub_x = x[node_idx_tensor]
    sub_edge_index = (torch.zeros((2, 0), dtype=torch.long) if not edges_src
                       else torch.tensor([edges_src, edges_dst], dtype=torch.long))
    seed_local_idx = torch.tensor([local_id[s] for s in seed_nodes], dtype=torch.long)
    return sub_x, sub_edge_index, seed_local_idx


def compute_type_class_weights(y_type_remapped, train_mask, num_classes, cap=TYPE_WEIGHT_CAP):
    illicit_train_types = y_type_remapped[train_mask]
    illicit_train_types = illicit_train_types[illicit_train_types != -1].numpy()
    present_classes = np.unique(illicit_train_types)
    balanced = compute_class_weight("balanced", classes=present_classes, y=illicit_train_types)
    weights = np.ones(num_classes, dtype=np.float32)
    for cls, w in zip(present_classes, balanced):
        weights[cls] = min(w, cap)
    return torch.tensor(weights, dtype=torch.float)


def run_one_seed(seed, x, edge_index, y_binary, y_type_remapped, train_mask, val_mask, test_mask,
                  adjacency, num_type_classes, class_weight_binary, class_weight_type, device,
                  epochs=EPOCHS, verbose_epochs=False):
    set_seed(seed)
    rng = random.Random(seed)

    model = MultiTaskGraphSAGE(in_channels=x.shape[1], hidden_channels=64,
                                num_type_classes=num_type_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    binary_criterion = nn.CrossEntropyLoss(weight=class_weight_binary.to(device))
    type_criterion = nn.CrossEntropyLoss(weight=class_weight_type.to(device), ignore_index=-1)

    train_node_ids = train_mask.nonzero(as_tuple=True)[0].tolist()

    for epoch in range(1, epochs + 1):
        rng.shuffle(train_node_ids)
        model.train()
        total_bin_loss, n_bin_batches = 0.0, 0
        total_type_loss, n_type_batches = 0.0, 0  # FIX 1: separate counter, only valid batches

        for i in range(0, len(train_node_ids), BATCH_SIZE):
            seed_nodes = train_node_ids[i:i + BATCH_SIZE]
            sub_x, sub_edge_index, seed_local_idx = build_batch_subgraph(seed_nodes, adjacency, x, rng)
            sub_x, sub_edge_index = sub_x.to(device), sub_edge_index.to(device)
            seed_local_idx = seed_local_idx.to(device)
            batch_y_binary = y_binary[seed_nodes].to(device)
            batch_y_type = y_type_remapped[seed_nodes].to(device)

            optimizer.zero_grad()
            out_binary, out_type = model(sub_x, sub_edge_index)
            loss_binary = binary_criterion(out_binary[seed_local_idx], batch_y_binary)

            has_valid_type = (batch_y_type != -1).any()
            if has_valid_type:
                loss_type = type_criterion(out_type[seed_local_idx], batch_y_type)
                loss = loss_binary + TYPE_LOSS_WEIGHT * loss_type
                total_type_loss += loss_type.item()
                n_type_batches += 1
            else:
                loss = loss_binary  # FIX 1: skip the type term entirely, no nan enters the graph

            loss.backward()
            optimizer.step()

            total_bin_loss += loss_binary.item()
            n_bin_batches += 1

        if verbose_epochs and epoch % 5 == 0:
            avg_type = total_type_loss / max(n_type_batches, 1)
            print(f"  seed={seed} epoch={epoch:03d}  Binary={total_bin_loss/n_bin_batches:.4f}  "
                  f"Type={avg_type:.4f}  (valid type batches: {n_type_batches}/{n_bin_batches})")

    model.eval()
    with torch.no_grad():
        out_binary, out_type = model(x.to(device), edge_index.to(device))
        probs_binary = F.softmax(out_binary, dim=1)[:, 1].cpu().numpy()
        probs_type = F.softmax(out_type, dim=1).cpu().numpy()
        preds_type = probs_type.argmax(axis=1)

    # binary: same val-tuned-threshold discipline as everywhere else
    y_val_b = y_binary[val_mask].numpy()
    probs_val_b = probs_binary[val_mask.numpy()]
    best_t, _ = find_best_threshold(y_val_b, probs_val_b)

    y_test_b = y_binary[test_mask].numpy()
    probs_test_b = probs_binary[test_mask.numpy()]

    # note: type head evaluated only on real illicit test accounts
    y_test_type = y_type_remapped[test_mask].numpy()
    preds_test_type = preds_type[test_mask.numpy()]
    has_type = y_test_type != -1

    type_metrics = {
        "accuracy": accuracy_score(y_test_type[has_type], preds_test_type[has_type]),
        "macro_f1": f1_score(y_test_type[has_type], preds_test_type[has_type], average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test_type[has_type], preds_test_type[has_type], average="weighted", zero_division=0),
    }

    return {
        "seed": seed,
        "binary_threshold": best_t,
        "probs_test_binary": probs_test_b,
        "y_test_binary": y_test_b,
        "type_metrics": type_metrics,
        "y_test_type": y_test_type[has_type],
        "preds_test_type": preds_test_type[has_type],
    }


if __name__ == "__main__":
    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y_binary = data_dict["y_binary"]
    y_type = data_dict["y_type"]
    train_mask = data_dict["train_mask"]
    val_mask = data_dict["val_mask"]
    test_mask = data_dict["test_mask"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # FIX 2: same merge rule as step23/step25
    remap, num_new_types, kept_type_ids = merge_rare_types(y_type, train_mask)
    y_type_remapped = y_type.clone()
    for old_id, new_id in remap.items():
        y_type_remapped[y_type == old_id] = new_id
    print(f"Type classes after merge: {num_new_types}")

    type_names = load_type_names()

    n_pos = (y_binary[train_mask] == 1).sum().item()
    n_neg = (y_binary[train_mask] == 0).sum().item()
    class_weight_binary = torch.tensor([1.0, min(n_neg / max(n_pos, 1), WEIGHT_CAP)])

    # FIX 3: real class weights for the type head
    class_weight_type = compute_type_class_weights(y_type_remapped, train_mask, num_new_types)
    print("Type class weights:", class_weight_type.tolist())

    adjacency = defaultdict(list)
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adjacency[d].append(s)

    all_results = []
    for seed in SEEDS:
        print(f"\n{'=' * 60}\nSeed {seed}\n{'=' * 60}")
        result = run_one_seed(
            seed, x, edge_index, y_binary, y_type_remapped, train_mask, val_mask, test_mask,
            adjacency, num_new_types, class_weight_binary, class_weight_type, device,
            verbose_epochs=True,
        )
        all_results.append(result)
        print(f"  type macro_f1 this seed: {result['type_metrics']['macro_f1']:.4f}")

    print(f"\n\n{'=' * 60}\n=== Binary head summary across {len(SEEDS)} seeds ===\n{'=' * 60}")
    for r in all_results:
        print(f"\nseed={r['seed']}:")
        report_at_percentile_thresholds(r["y_test_binary"], r["probs_test_binary"], percentiles=(90, 95, 99))

    print(f"\n\n{'=' * 60}\n=== Type head summary across {len(SEEDS)} seeds ===\n{'=' * 60}")
    for key in ("accuracy", "macro_f1", "weighted_f1"):
        vals = np.array([r["type_metrics"][key] for r in all_results])
        print(f"  {key:12s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")

    print("\nPer-class classification_report, last seed (illustrative):")
    last = all_results[-1]
    labels_present = sorted(set(last["y_test_type"].tolist()) | set(last["preds_test_type"].tolist()))
    target_names = [type_names.get(i, f"type_{i}") for i in labels_present] if type_names else None
    print(classification_report(last["y_test_type"], last["preds_test_type"],
                                 labels=labels_present, target_names=target_names, zero_division=0))