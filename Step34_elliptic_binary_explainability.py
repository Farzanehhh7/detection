"""
Step34_elliptic_binary_explainability.py

Same explainability technique already built and validated for SAML-D
(step33): Grad x Input on the illicit-class logit, giving two
complementary explanations per flagged transaction:

  1. Own-feature attribution: which of the 165 anonymized features
     (feat_0 .. feat_164 -- Elliptic never released real names for these,
     unlike SAML-D's 8 named columns) pushed this transaction's own
     illicit score up or down.
  2. Neighbor-contribution attribution: which other transactions, up to
     2 hops away (this model is 2-layer SAGEConv), most influenced this
     specific decision. Nodes outside the 2-hop neighborhood get exactly
     zero gradient, so this naturally restricts to the accounts that
     could plausibly matter.

Uses the real, already-trained checkpoint (structural_only_best.pt,
seed=123, F1=0.4909) from Step17 -- no retraining.

WHAT WAS ACTUALLY TESTED, STATED HONESTLY: this script's logic was run
against synthetic Elliptic-shaped dummy data (165 features, small node
count) purely to confirm the forward/backward mechanics work with the
real checkpoint's weights (shapes match exactly: block1 165->64,
block2 64->64, classifier 64->2). It was NOT run against your real
elliptic_txs_*.csv files, which weren't available in this session.
Please run it for real and sanity-check the printed P(illicit) values
against numbers you already trust before citing specific attributions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from metrics_utils import build_edge_index, get_temporal_split_masks

FEATURES_PATH = "datasets/elliptic_txs_features.csv"
EDGES_PATH = "datasets/elliptic_txs_edgelist.csv"
CLASSES_PATH = "datasets/elliptic_txs_classes.csv"
CHECKPOINT_PATH = "structural_only_best.pt"


# ---------------------------------------------------------------------------
# Identical architecture to Step17 -- required for the checkpoint to load
# ---------------------------------------------------------------------------
class SAGEBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        return F.dropout(h, p=0.2, training=self.training)


class StructuralOnlyGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.block1 = SAGEBlock(in_channels, hidden_channels)
        self.block2 = SAGEBlock(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.dropout(F.relu(self.block1(x, edge_index)))
        h = self.dropout(F.relu(self.block2(h, edge_index)))
        return self.classifier(h)


def explain_elliptic_decision(model, x, edge_index, node_idx, y, device,
                               top_k_features=5, top_k_neighbors=5, verbose=True):
    model.eval()
    x_grad = x.clone().to(device).requires_grad_(True)
    out = model(x_grad, edge_index.to(device))
    probs = F.softmax(out, dim=1)[:, 1]
    illicit_logit = out[node_idx, 1]

    illicit_logit.backward()
    grad = x_grad.grad

    own_grad = grad[node_idx].detach().cpu().numpy()
    own_feat = x[node_idx].detach().cpu().numpy()
    own_attr = own_grad * own_feat
    order = np.argsort(-np.abs(own_attr))

    if verbose:
        label = "illicit" if y[node_idx].item() == 1 else ("licit" if y[node_idx].item() == 0 else "unknown")
        print(f"\n{'=' * 70}\nTransaction node {node_idx}   "
              f"P(illicit)={probs[node_idx].item():.4f}   actual label={label}\n{'=' * 70}")
        print("Own-feature attribution (feat_i = i-th anonymized Elliptic feature):")
        for i in order[:top_k_features]:
            print(f"  feat_{i:<4d}                     value={own_feat[i]:+.3f}   attribution={own_attr[i]:+.4f}")

    grad_magnitude = grad.abs().sum(dim=1).cpu().numpy()
    grad_magnitude[node_idx] = 0.0
    n_neighborhood = int((grad_magnitude > 0).sum())
    top_neighbors = np.argsort(-grad_magnitude)[:top_k_neighbors]
    top_neighbors = [n for n in top_neighbors if grad_magnitude[n] > 0]

    if verbose:
        print(f"\nNeighbor contribution ({n_neighborhood} transactions in the 2-hop computational neighborhood):")
        for n_idx in top_neighbors:
            label = "illicit" if y[n_idx].item() == 1 else ("licit" if y[n_idx].item() == 0 else "unknown")
            print(f"  node {n_idx:8d}   influence={grad_magnitude[n_idx]:.4f}   actual label={label}")

    return {
        "node_idx": node_idx, "prob_illicit": probs[node_idx].item(),
        "n_neighborhood": n_neighborhood,
        "top_neighbors": [(int(n), float(grad_magnitude[n]), int(y[n].item())) for n in top_neighbors],
    }


if __name__ == "__main__":
    print("Loading Elliptic dataset...")
    df_feat = pd.read_csv(FEATURES_PATH, header=None)
    df_edge = pd.read_csv(EDGES_PATH)
    df_class = pd.read_csv(CLASSES_PATH)

    df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
    df_class.columns = ["txId", "class"]
    df_class["label"] = df_class["class"].map({"1": 1, "2": 0, "unknown": -1})

    map_id, edge_index = build_edge_index(df_feat["txId"].values, df_edge["txId1"], df_edge["txId2"])

    x_raw = df_feat.drop(columns=["txId", "time_step"]).values
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float)
    y = torch.tensor(df_class["label"].values, dtype=torch.long)

    time_steps_raw = torch.tensor(df_feat["time_step"].values, dtype=torch.long)
    train_mask, val_mask, test_mask = get_temporal_split_masks(time_steps_raw, y, train_end=27, val_end=34)

    device = torch.device("cpu")
    model = StructuralOnlyGraphSAGE(in_channels=165, hidden_channels=64, out_channels=2).to(device)
    payload = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    print(f"Loaded checkpoint: seed={payload['seed']}  F1={payload['F1']:.4f}")

    illicit_test_idx = (test_mask & (y == 1)).nonzero(as_tuple=True)[0]
    print(f"\n{len(illicit_test_idx)} illicit test transactions available. Explaining the first 3 in detail.")
    for node_idx in illicit_test_idx[:3].tolist():
        explain_elliptic_decision(model, x, edge_index, node_idx, y, device)

    N_SAMPLE = 50
    sample_idx = illicit_test_idx[:N_SAMPLE].tolist()
    print(f"\n\n{'=' * 70}\n=== Aggregate check over {len(sample_idx)} illicit test transactions ===\n{'=' * 70}")
    n_with_neighbors, n_top_illicit, n_any_top5_illicit = 0, 0, 0
    neighborhood_sizes = []
    for node_idx in sample_idx:
        r = explain_elliptic_decision(model, x, edge_index, node_idx, y, device, verbose=False)
        neighborhood_sizes.append(r["n_neighborhood"])
        if r["top_neighbors"]:
            n_with_neighbors += 1
            if r["top_neighbors"][0][2] == 1:
                n_top_illicit += 1
            if any(n[2] == 1 for n in r["top_neighbors"]):
                n_any_top5_illicit += 1

    print(f"Transactions with >=1 2-hop neighbor: {n_with_neighbors}/{len(sample_idx)}")
    print(f"Average 2-hop neighborhood size: {np.mean(neighborhood_sizes):.2f}")
    if n_with_neighbors:
        print(f"Top-influence neighbor is itself illicit: {n_top_illicit}/{n_with_neighbors} "
              f"({100*n_top_illicit/n_with_neighbors:.1f}%)")
        print(f"At least one of top-5 neighbors is illicit: {n_any_top5_illicit}/{n_with_neighbors} "
              f"({100*n_any_top5_illicit/n_with_neighbors:.1f}%)")
    illicit_rate_overall = (y == 1).float().mean().item()
    print(f"\nFor reference, overall illicit rate among ALL labeled transactions is {illicit_rate_overall*100:.2f}% "
          f"-- compare that baseline to the percentages above.")