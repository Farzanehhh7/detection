"""
step29_binary_only_fair_reference.py

The fair 5-seed binary-only reference, using percentile-threshold
reporting (not F1), so it's directly comparable to multi-task's binary
head numbers in step28.

Does NOT retrain anything -- reuses the 5 checkpoints step22 already
saved (samld_seed_<seed>.pt). Those checkpoints are the pure binary
model (no type head at all), so this is the correct "no multi-task cost"
baseline to compare step28's binary numbers against.
"""

import numpy as np
import torch

from metrics_utils import load_checkpoint
from step25_samld_type_classification import StructuralOnlyGraphSAGE

DATA_PATH = "samld_processed_v3.pt"
CHECKPOINT_PATTERN = "samld_seed_{seed}.pt"
SEEDS = (42, 1, 7, 123, 2024)
PERCENTILES = (90, 95, 99)


def report_at_percentile_thresholds_capture(y_true, y_prob, percentiles):
    """Same computation as metrics_utils.report_at_percentile_thresholds,
    but returns the numbers instead of only printing them, so they can be
    averaged across seeds."""
    from sklearn.metrics import precision_score, recall_score
    results = {}
    for p in percentiles:
        threshold = np.percentile(y_prob, p)
        preds = (y_prob >= threshold).astype(int)
        flagged = preds.sum()
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        results[p] = {"threshold": threshold, "flagged": flagged, "precision": precision, "recall": recall}
        print(f"  صدک {p:5.1f} | threshold={threshold:.4f} | flagged={flagged:6d} | "
              f"Precision={precision:.4f} | Recall={recall:.4f}")
    return results


if __name__ == "__main__":
    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y_binary = data_dict["y_binary"]
    test_mask = data_dict["test_mask"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_seed_results = []
    for seed in SEEDS:
        print(f"\n{'=' * 60}\nSeed {seed} (reusing existing checkpoint, no retraining)\n{'=' * 60}")
        model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
        load_checkpoint(model, CHECKPOINT_PATTERN.format(seed=seed), map_location=device)

        model.eval()
        with torch.no_grad():
            out = model(x.to(device), edge_index.to(device))
            probs_all = torch.softmax(out, dim=1)[:, 1].cpu().numpy()

        y_test = y_binary[test_mask].numpy()
        probs_test = probs_all[test_mask.numpy()]

        seed_results = report_at_percentile_thresholds_capture(y_test, probs_test, PERCENTILES)
        all_seed_results.append(seed_results)

    print(f"\n\n{'=' * 60}\n=== FAIR 5-SEED BINARY-ONLY REFERENCE (percentile-based) ===\n{'=' * 60}")
    for p in PERCENTILES:
        precisions = np.array([r[p]["precision"] for r in all_seed_results])
        recalls = np.array([r[p]["recall"] for r in all_seed_results])
        print(f"صدک {p:5.1f}:  Precision = {precisions.mean():.4f} +/- {precisions.std(ddof=1):.4f}   "
              f"Recall = {recalls.mean():.4f} +/- {recalls.std(ddof=1):.4f}")

    print("\nCompare this to step28's multi-task binary numbers at the same percentiles.")