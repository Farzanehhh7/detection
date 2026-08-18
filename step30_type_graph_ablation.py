#
# import numpy as np
# import torch
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.utils.class_weight import compute_class_weight
#
# from step25_samld_type_classification import merge_rare_types
#
# DATA_PATH = "samld_processed_v3.pt"
# SEEDS = (42, 1, 7, 123, 2024)  # used as LogisticRegression random_state for a like-for-like 5-run spread
#
#
# def fit_and_evaluate_no_graph(x, y_type_remapped, train_mask, test_mask, seed):
#     illicit_train = train_mask & (y_type_remapped != -1)
#     illicit_test = test_mask & (y_type_remapped != -1)
#
#     X_train = x[illicit_train].numpy()
#     y_train = y_type_remapped[illicit_train].numpy()
#     X_test = x[illicit_test].numpy()
#     y_test = y_type_remapped[illicit_test].numpy()
#
#     clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#
#     return {
#         "accuracy": accuracy_score(y_test, y_pred),
#         "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
#         "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
#     }
#
#
# if __name__ == "__main__":
#     print("Loading real SAML-D v3 data...")
#     data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
#     x = data_dict["x"]
#     y_type = data_dict["y_type"]
#     train_mask = data_dict["train_mask"]
#     test_mask = data_dict["test_mask"]
#
#     remap, num_new_types, _ = merge_rare_types(y_type, train_mask)
#     y_type_remapped = y_type.clone()
#     for old_id, new_id in remap.items():
#         y_type_remapped[y_type == old_id] = new_id
#
#     print("\n=== NO-GRAPH type classifier: raw 8 account features only ===")
#     no_graph_metrics = []
#     for seed in SEEDS:
#         m = fit_and_evaluate_no_graph(x, y_type_remapped, train_mask, test_mask, seed)
#         no_graph_metrics.append(m)
#         print(f"  seed={seed}  accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")
#
#     print(f"\n{'=' * 70}\n=== FULL LADDER: how much does each graph ingredient add? ===\n{'=' * 70}")
#
#     def summarize(metrics, label):
#         vals = np.array([m["macro_f1"] for m in metrics])
#         print(f"{label:45s} macro_f1 = {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")
#         return vals
#
#     no_graph_vals = summarize(no_graph_metrics, "NO GRAPH (raw 8 features only)")
#
#     # these three are the already-measured real results from step25/26/28 --
#     # hardcoded here from the actual runs so the full ladder prints in one place.
#     embedding_only_vals = np.array([0.1398, 0.1125, 0.1142, 0.0943, 0.1224])       # step25
#     embedding_structural_vals = np.array([0.1681, 0.1148, 0.1237, 0.1369, 0.1317])  # step26 enhanced
#     multitask_vals = np.array([0.1461, 0.1447, 0.1521, 0.2243, 0.1356])             # step28 (this session's run)
#
#     print(f"{'GNN embedding only (step25)':45s} macro_f1 = {embedding_only_vals.mean():.4f} +/- {embedding_only_vals.std(ddof=1):.4f}")
#     print(f"{'GNN embedding + structural feats (step26)':45s} macro_f1 = {embedding_structural_vals.mean():.4f} +/- {embedding_structural_vals.std(ddof=1):.4f}")
#     print(f"{'Multi-task shared backbone (step28)':45s} macro_f1 = {multitask_vals.mean():.4f} +/- {multitask_vals.std(ddof=1):.4f}")
#
#     from scipy import stats
#     print(f"\n--- Paired t-tests against NO GRAPH ---")
#     for label, vals in [("embedding_only", embedding_only_vals),
#                          ("embedding+structural", embedding_structural_vals),
#                          ("multi_task", multitask_vals)]:
#         t, p = stats.ttest_rel(vals, no_graph_vals)
#         delta = vals.mean() - no_graph_vals.mean()
#         sig = "SIGNIFICANT" if p < 0.05 else "not significant"
#         print(f"  {label:25s} delta={delta:+.4f}  t={t:.3f}  p={p:.4f}  ({sig})")


import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from step25_samld_type_classification import merge_rare_types

DATA_PATH = "samld_processed_v3.pt"
SEEDS = (42, 1, 7, 123, 2024)  # used as LogisticRegression random_state for a like-for-like 5-run spread


def fit_and_evaluate_no_graph(x, y_type_remapped, train_mask, test_mask, seed):
    illicit_train = train_mask & (y_type_remapped != -1)
    illicit_test = test_mask & (y_type_remapped != -1)

    X_train = x[illicit_train].numpy()
    y_train = y_type_remapped[illicit_train].numpy()
    X_test = x[illicit_test].numpy()
    y_test = y_type_remapped[illicit_test].numpy()

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }


if __name__ == "__main__":
    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data_dict["x"]
    y_type = data_dict["y_type"]
    train_mask = data_dict["train_mask"]
    test_mask = data_dict["test_mask"]

    remap, num_new_types, _ = merge_rare_types(y_type, train_mask)
    y_type_remapped = y_type.clone()
    for old_id, new_id in remap.items():
        y_type_remapped[y_type == old_id] = new_id

    print("\n=== NO-GRAPH type classifier: raw 8 account features only ===")
    no_graph_metrics = []
    for seed in SEEDS:
        m = fit_and_evaluate_no_graph(x, y_type_remapped, train_mask, test_mask, seed)
        no_graph_metrics.append(m)
        print(f"  seed={seed}  accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    print(f"\n{'=' * 70}\n=== FULL LADDER: how much does each graph ingredient add? ===\n{'=' * 70}")

    def summarize(metrics, label):
        vals = np.array([m["macro_f1"] for m in metrics])
        print(f"{label:45s} macro_f1 = {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")
        return vals

    no_graph_vals = summarize(no_graph_metrics, "NO GRAPH (raw 8 features only)")


    embedding_only_vals = np.array([0.1100, 0.1088, 0.1295, 0.1074, 0.1252])        # step25، seeds 42,1,7,123,2024
    embedding_structural_vals = np.array([0.1573, 0.1245, 0.1592, 0.1089, 0.1183])  # step26 enhanced، همون ترتیب seed
    multitask_vals = np.array([0.2252, 0.1377, 0.1085, 0.1518, 0.1196])             # step28، همون ترتیب seed

    print(f"{'GNN embedding only (step25)':45s} macro_f1 = {embedding_only_vals.mean():.4f} +/- {embedding_only_vals.std(ddof=1):.4f}")
    print(f"{'GNN embedding + structural feats (step26)':45s} macro_f1 = {embedding_structural_vals.mean():.4f} +/- {embedding_structural_vals.std(ddof=1):.4f}")
    print(f"{'Multi-task shared backbone (step28)':45s} macro_f1 = {multitask_vals.mean():.4f} +/- {multitask_vals.std(ddof=1):.4f}")

    from scipy import stats
    print(f"\n--- Paired t-tests against NO GRAPH ---")
    for label, vals in [("embedding_only", embedding_only_vals),
                         ("embedding+structural", embedding_structural_vals),
                         ("multi_task", multitask_vals)]:
        t, p = stats.ttest_rel(vals, no_graph_vals)
        delta = vals.mean() - no_graph_vals.mean()
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {label:25s} delta={delta:+.4f}  t={t:.3f}  p={p:.4f}  ({sig})")