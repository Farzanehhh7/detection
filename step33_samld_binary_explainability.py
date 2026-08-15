"""
step33_samld_binary_explainability.py

Explains "why is this account flagged as suspicious?" for the BINARY
head -- the piece that was still missing (step25/26/28/31 only explain
"why this type", never "why suspicious at all").

Two complementary explanations, both via Grad x Input on the illicit-
class logit, backpropped through the actual trained binary model:

1. OWN-FEATURE attribution: which of this account's 8 raw features
   (sent/recv amount, count, payment-type diversity) pushed its own
   illicit score up or down. Same method already validated for the type
   classifier.

2. NEIGHBOR-CONTRIBUTION attribution (new): because this is a 2-layer
   message-passing model, an account's final embedding depends on
   accounts up to 2 hops away. Backpropping to the FULL input tensor (not
   just this node's own row) and ranking other nodes by total gradient
   magnitude reveals WHICH neighboring accounts most influenced this
   specific decision -- a genuinely relational explanation, not just a
   tabular one. Nodes outside the 2-hop neighborhood get exactly zero
   gradient (the computation graph never reaches them), so this
   automatically restricts to the accounts that could plausibly matter.

Uses the existing 5 checkpoints from step22 -- no retraining.
"""

import numpy as np
import torch
import torch.nn.functional as F

from metrics_utils import load_checkpoint
from step25_samld_type_classification import StructuralOnlyGraphSAGE

DATA_PATH = "samld_processed_v3.pt"
CHECKPOINT_PATTERN = "samld_seed_{seed}.pt"


def explain_binary_decision(model, x, edge_index, node_idx, feature_names, y_binary, device,
                             top_k_features=5, top_k_neighbors=5, verbose=True):
    model.eval()
    x_grad = x.clone().to(device).requires_grad_(True)
    out = model(x_grad, edge_index.to(device))
    probs = F.softmax(out, dim=1)[:, 1]
    illicit_logit = out[node_idx, 1]

    illicit_logit.backward()
    grad = x_grad.grad  # (N, 8)

    # --- 1. own-feature attribution ---
    own_grad = grad[node_idx].detach().cpu().numpy()
    own_features = x[node_idx].detach().cpu().numpy()
    own_attribution = own_grad * own_features
    order = np.argsort(-np.abs(own_attribution))

    if verbose:
        print(f"\n{'=' * 70}\nAccount node {node_idx}   "
              f"P(illicit)={probs[node_idx].item():.4f}   "
              f"actual label={'illicit' if y_binary[node_idx].item() == 1 else 'licit'}\n{'=' * 70}")
        print("Own-feature attribution (why THIS account's own numbers raised suspicion):")
        for i in order[:top_k_features]:
            print(f"  {feature_names[i]:28s} value={own_features[i]:+.3f}   attribution={own_attribution[i]:+.4f}")

    # --- 2. neighbor-contribution attribution ---
    grad_magnitude = grad.abs().sum(dim=1).cpu().numpy()
    grad_magnitude[node_idx] = 0.0  # exclude self, already covered above
    n_in_computational_neighborhood = int((grad_magnitude > 0).sum())

    top_neighbors = np.argsort(-grad_magnitude)[:top_k_neighbors]
    top_neighbors = [n for n in top_neighbors if grad_magnitude[n] > 0]

    if verbose:
        print(f"\nNeighbor contribution (which OTHER accounts, up to 2 hops away, "
              f"influenced this decision most):")
        print(f"  ({n_in_computational_neighborhood} accounts total in the 2-hop computational neighborhood)")
        for n_idx in top_neighbors:
            label = "illicit" if y_binary[n_idx].item() == 1 else "licit"
            print(f"  node {n_idx:8d}   influence={grad_magnitude[n_idx]:.4f}   actual label={label}")

    return {
        "node_idx": node_idx,
        "prob_illicit": probs[node_idx].item(),
        "own_attribution": dict(zip(feature_names, own_attribution.tolist())),
        "n_neighborhood": n_in_computational_neighborhood,
        "top_neighbors": [(int(n), float(grad_magnitude[n]), int(y_binary[n].item())) for n in top_neighbors],
    }


if __name__ == "__main__":
    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y_binary = data_dict["y_binary"]
    test_mask = data_dict["test_mask"]
    feature_names = data_dict["feature_cols"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEED = 42  # illustrative -- for a real report, repeat across all 5 and note if the
               # top explanatory features/neighbors are consistent across seeds
    model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
    payload = load_checkpoint(model, CHECKPOINT_PATTERN.format(seed=SEED), map_location=device)
    print(f"Loaded seed={SEED} checkpoint, binary F1 on this seed: {payload['F1']:.4f}")

    # pick a few real illicit test accounts to explain in detail
    illicit_test_idx = (test_mask & (y_binary == 1)).nonzero(as_tuple=True)[0]
    print(f"\n{len(illicit_test_idx)} illicit test accounts available. Explaining the first 3 in detail.")

    for node_idx in illicit_test_idx[:3].tolist():
        explain_binary_decision(model, x, edge_index, node_idx, feature_names, y_binary, device)

    # ------------------------------------------------------------------
    # Aggregate statistic over N accounts: how often is the single most
    # influential neighbor itself illicit? (guilt-by-association check)
    # ------------------------------------------------------------------
    N_SAMPLE = 50
    sample_idx = illicit_test_idx[:N_SAMPLE].tolist()
    print(f"\n\n{'=' * 70}\n=== Aggregate check over {len(sample_idx)} illicit test accounts ===\n{'=' * 70}")

    n_with_neighbors = 0
    n_top_neighbor_illicit = 0
    n_any_top5_illicit = 0
    neighborhood_sizes = []

    for node_idx in sample_idx:
        result = explain_binary_decision(model, x, edge_index, node_idx, feature_names, y_binary, device, verbose=False)
        neighborhood_sizes.append(result["n_neighborhood"])
        if result["top_neighbors"]:
            n_with_neighbors += 1
            if result["top_neighbors"][0][2] == 1:  # top neighbor's actual label
                n_top_neighbor_illicit += 1
            if any(n[2] == 1 for n in result["top_neighbors"]):
                n_any_top5_illicit += 1

    print(f"Accounts with at least one 2-hop neighbor: {n_with_neighbors}/{len(sample_idx)}")
    print(f"Average 2-hop computational neighborhood size: {np.mean(neighborhood_sizes):.2f}")
    if n_with_neighbors > 0:
        print(f"Top-influence neighbor is itself illicit: {n_top_neighbor_illicit}/{n_with_neighbors} "
              f"({100*n_top_neighbor_illicit/n_with_neighbors:.1f}%)")
        print(f"At least one of top-5 influential neighbors is illicit: {n_any_top5_illicit}/{n_with_neighbors} "
              f"({100*n_any_top5_illicit/n_with_neighbors:.1f}%)")
    print(f"\nFor reference, overall illicit rate among ALL accounts is 0.28% -- "
          f"compare that tiny baseline rate to the percentages above.")