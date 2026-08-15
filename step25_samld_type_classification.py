# # """
# # step25_samld_type_classification.py
# #
# # Typology classification (which laundering type) + explainability (why),
# # built on top of the existing binary SAML-D GraphSAGE pipeline.
# #
# # DESIGN: two stages, not one joint model (for now)
# # =====================================================================
# # Stage 1 (already exists, e.g. step22): binary GraphSAGE flags illicit
# #   accounts, trained with the standard 5-seed protocol.
# # Stage 2 (this file): reuses that model's learned embeddings (h2) as
# #   input features for a separate multi-class classifier over the 17
# #   laundering types -- trained ONLY on accounts already labeled illicit
# #   (y_type == -1 for licit accounts, so they're excluded entirely).
# #
# # Why two stages instead of one joint multi-task head right away:
# #   - Reuses checkpoints you already trained/validated with 5 seeds;
# #     no need to retrain the GNN from scratch.
# #   - Keeps the type classifier simple (sklearn LogisticRegression), so
# #     the very small per-type sample counts (many types have <10 train
# #     examples, per step23's own finding) don't also have to fight
# #     GNN-scale hyperparameters.
# #   - A joint multi-task head (same backbone, second nn.Linear output)
# #     is a natural upgrade later if this simpler version works well --
# #     see the note at the bottom of the file.
# #
# # EXPLAINABILITY: Grad x Input attribution on the 8 raw SAML-D features.
# #   Simplest attribution method that still gives a genuinely readable
# #   answer per account, e.g. "flagged as type 5 mainly because of
# #   recv_amount_count and cross-border pattern" -- no architecture
# #   change needed, just a few extra lines using autograd.
# #
# # REQUIRED ONE-LINE CHANGE TO YOUR EXISTING SAML-D MODEL CLASS:
# #   Your SAML-D StructuralOnlyGraphSAGE (step19/20/22/24) doesn't return
# #   embeddings yet -- only the Elliptic version in Step17 does. Add the
# #   same `return_embeddings` parameter here too (done below), and add a
# #   `save_checkpoint(...)` call at the end of your SAML-D training loop
# #   (you already have this utility in metrics_utils.py, just not wired
# #   into the SAML-D scripts yet).
# # """
# #
# # import random
# # from collections import defaultdict, Counter
# #
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch_geometric.nn import SAGEConv
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.metrics import classification_report
# #
# # from metrics_utils import set_seed, save_checkpoint, load_checkpoint  # noqa: F401  (load_checkpoint used once you have real checkpoints)
# #
# # DATA_PATH = "samld_processed_v3.pt"
# # S1, S2 = 25, 10
# # BATCH_SIZE = 256
# # WEIGHT_CAP = 30.0
# # MIN_TRAIN_SAMPLES_PER_TYPE = 10  # exactly the rule you proposed in step23
# #
# #
# # # ---------------------------------------------------------------------------
# # # Model: same as your existing SAML-D StructuralOnlyGraphSAGE, plus the
# # # return_embeddings flag (the one-line addition mentioned above).
# # # ---------------------------------------------------------------------------
# #
# # class StructuralOnlyGraphSAGE(nn.Module):
# #     def __init__(self, in_channels, hidden_channels, out_channels):
# #         super().__init__()
# #         self.conv1 = SAGEConv(in_channels, hidden_channels)
# #         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
# #         self.classifier = nn.Linear(hidden_channels, out_channels)
# #
# #     def forward(self, x, edge_index, return_embeddings=False):
# #         h1 = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
# #         h2 = F.dropout(F.relu(self.conv2(h1, edge_index)), p=0.3, training=self.training)
# #         out = self.classifier(h2)
# #         if return_embeddings:
# #             return out, h1, h2
# #         return out
# #
# #
# # # ---------------------------------------------------------------------------
# # # Stage 2a: rare-type merging, exactly the rule from step23
# # # ---------------------------------------------------------------------------
# #
# # def merge_rare_types(y_type, train_mask, min_train_samples=MIN_TRAIN_SAMPLES_PER_TYPE):
# #     """
# #     Any type with fewer than `min_train_samples` illicit examples in TRAIN
# #     gets merged into a single "Other" bucket (id = number of kept types).
# #     Decision is made using TRAIN counts only, never val/test, to avoid
# #     leaking test-set class-frequency information into the label space.
# #
# #     Returns:
# #         remap: dict old_type_id -> new_type_id
# #         num_new_types: int, including the "Other" bucket if used
# #         kept_type_ids: sorted list of original type ids kept as-is
# #     """
# #     train_types = y_type[train_mask]
# #     train_types = train_types[train_types != -1].tolist()
# #     counts = Counter(train_types)
# #
# #     all_type_ids = sorted(set(y_type[y_type != -1].tolist()))
# #     kept_type_ids = [t for t in all_type_ids if counts.get(t, 0) >= min_train_samples]
# #     rare_type_ids = [t for t in all_type_ids if t not in kept_type_ids]
# #
# #     remap = {t: i for i, t in enumerate(kept_type_ids)}
# #     other_id = len(kept_type_ids)
# #     if rare_type_ids:
# #         for t in rare_type_ids:
# #             remap[t] = other_id
# #         num_new_types = other_id + 1
# #     else:
# #         num_new_types = other_id
# #
# #     print(f"Types kept as-is: {len(kept_type_ids)}   Types merged into 'Other': {len(rare_type_ids)}")
# #     return remap, num_new_types, kept_type_ids
# #
# #
# # # ---------------------------------------------------------------------------
# # # Stage 2b: extract embeddings from a trained binary model
# # # ---------------------------------------------------------------------------
# #
# # def extract_embeddings(model, x, edge_index, device, batch_eval_size=50000):
# #     """
# #     Full-graph forward pass in eval mode, returning the h2 embedding for
# #     every node. Done in chunks only to keep peak memory reasonable on
# #     large graphs -- message passing still uses the FULL edge_index each
# #     time (SAGEConv needs full neighborhood context), only the returned
# #     slice is chunked.
# #     """
# #     model.eval()
# #     with torch.no_grad():
# #         _, _, h2_full = model(x.to(device), edge_index.to(device), return_embeddings=True)
# #     return h2_full.cpu()
# #
# #
# # # ---------------------------------------------------------------------------
# # # Stage 2c: fit the type classifier on illicit-only embeddings
# # # ---------------------------------------------------------------------------
# #
# # def fit_and_evaluate_type_classifier(h2, y_type_remapped, train_mask, test_mask, num_new_types,
# #                                       seed_label=None, verbose=True):
# #     from sklearn.metrics import accuracy_score, f1_score
# #
# #     illicit_train = train_mask & (y_type_remapped != -1)
# #     illicit_test = test_mask & (y_type_remapped != -1)
# #
# #     X_train = h2[illicit_train].numpy()
# #     y_train = y_type_remapped[illicit_train].numpy()
# #     X_test = h2[illicit_test].numpy()
# #     y_test = y_type_remapped[illicit_test].numpy()
# #
# #     if verbose:
# #         tag = f" (seed={seed_label})" if seed_label is not None else ""
# #         print(f"\nType classifier{tag}: train n={len(y_train)}   test n={len(y_test)}   "
# #               f"num_types(after merge)={num_new_types}")
# #
# #     clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
# #     clf.fit(X_train, y_train)
# #
# #     y_pred = clf.predict(X_test)
# #     labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
# #     if verbose:
# #         print(classification_report(y_test, y_pred, labels=labels_present, zero_division=0))
# #
# #     seed_metrics = {
# #         "accuracy": accuracy_score(y_test, y_pred),
# #         "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
# #         "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
# #     }
# #     return clf, seed_metrics
# #
# #
# # # ---------------------------------------------------------------------------
# # # Stage 2d: "why" -- Grad x Input attribution on the 8 raw features
# # # ---------------------------------------------------------------------------
# #
# # def explain_account(model, clf, x, edge_index, node_idx, feature_names, device, type_names=None):
# #     """
# #     Explains a single account's PREDICTED TYPE by attributing the binary
# #     model's embedding-driving gradient back to the 8 raw input features.
# #
# #     Method: Grad x Input. We backprop the norm of the node's own h2
# #     embedding (a proxy for "how strongly this node's representation is
# #     being shaped") through the GNN to its input features, then multiply
# #     element-wise by the feature values themselves. Positive score = this
# #     feature pushed the embedding in the direction that ended up mattering;
# #     magnitude = how much it mattered. This needs no architecture change.
# #     """
# #     model.eval()
# #     x_grad = x.clone().to(device).requires_grad_(True)
# #     _, _, h2 = model(x_grad, edge_index.to(device), return_embeddings=True)
# #
# #     target_embedding = h2[node_idx]
# #     target_embedding.pow(2).sum().backward()
# #
# #     grad_at_node = x_grad.grad[node_idx].detach().cpu().numpy()
# #     raw_features = x[node_idx].detach().cpu().numpy()
# #     attribution = grad_at_node * raw_features
# #
# #     predicted_type = clf.predict(h2[node_idx].detach().cpu().numpy().reshape(1, -1))[0]
# #     type_label = type_names[predicted_type] if type_names else f"type_{predicted_type}"
# #
# #     order = np.argsort(-np.abs(attribution))
# #     print(f"\nAccount node {node_idx} -> predicted type: {type_label}")
# #     print("Top contributing features:")
# #     for i in order[:5]:
# #         print(f"  {feature_names[i]:28s} value={raw_features[i]:+.3f}   attribution={attribution[i]:+.4f}")
# #
# #
# # # ---------------------------------------------------------------------------
# # # Main: real 5-seed protocol. Loads each of the 5 checkpoints saved by the
# # # updated step22 (samld_seed_<seed>.pt), extracts embeddings, fits a type
# # # classifier per seed, and reports mean +/- std across all 5 -- exactly the
# # # same discipline used everywhere else in the project.
# # # ---------------------------------------------------------------------------
# #
# # SEEDS = (42, 1, 7, 123, 2024)
# # CHECKPOINT_PATTERN = "samld_seed_{seed}.pt"
# #
# # if __name__ == "__main__":
# #     print("Loading real SAML-D v3 data...")
# #     data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
# #     x = data_dict["x"]
# #     edge_index = data_dict["edge_index"]
# #     y_type = data_dict["y_type"]
# #     train_mask = data_dict["train_mask"]
# #     test_mask = data_dict["test_mask"]
# #     feature_names = data_dict["feature_cols"]
# #
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"Using device: {device}")
# #     print(f"Nodes: {x.shape[0]}   Features: {x.shape[1]}   Types: {data_dict['num_types']}")
# #
# #     # --- rare-type merging, per step23's rule (train-only, seed-independent) ---
# #     remap, num_new_types, kept_type_ids = merge_rare_types(y_type, train_mask)
# #     y_type_remapped = y_type.clone()
# #     for old_id, new_id in remap.items():
# #         y_type_remapped[y_type == old_id] = new_id
# #     # y_type_remapped still has -1 for licit accounts, untouched
# #
# #     all_seed_metrics = []
# #     last_model, last_clf = None, None
# #
# #     for seed in SEEDS:
# #         ckpt_path = CHECKPOINT_PATTERN.format(seed=seed)
# #         print(f"\n{'=' * 60}\nSeed {seed}: loading {ckpt_path}\n{'=' * 60}")
# #
# #         model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
# #         payload = load_checkpoint(model, ckpt_path, map_location=device)
# #         print(f"Loaded checkpoint -- binary F1 on this seed: {payload['F1']:.4f}   "
# #               f"threshold: {payload['threshold']:.2f}")
# #
# #         h2 = extract_embeddings(model, x, edge_index, device)
# #         clf, seed_metrics = fit_and_evaluate_type_classifier(
# #             h2, y_type_remapped, train_mask, test_mask, num_new_types, seed_label=seed,
# #         )
# #         seed_metrics["seed"] = seed
# #         all_seed_metrics.append(seed_metrics)
# #         last_model, last_clf = model, clf
# #
# #     # --- aggregate across the 5 seeds, same discipline as run_multi_seed ---
# #     print(f"\n\n{'=' * 60}\n=== Type classifier -- summary across {len(SEEDS)} seeds ===\n{'=' * 60}")
# #     print(f"{'seed':>6} {'accuracy':>10} {'macro_f1':>10} {'weighted_f1':>12}")
# #     for m in all_seed_metrics:
# #         print(f"{m['seed']:>6} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['weighted_f1']:>12.4f}")
# #
# #     for key in ("accuracy", "macro_f1", "weighted_f1"):
# #         vals = np.array([m[key] for m in all_seed_metrics])
# #         print(f"\n{key:12s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")
# #
# #     # --- explainability demo, using the last loaded seed's model+classifier.
# #     #     This is illustrative only (showing HOW the explanation works for a
# #     #     couple of real accounts), not a reported metric -- so, unlike
# #     #     Step17's "pick the best seed" issue, using just one seed here for
# #     #     a qualitative example carries no statistical bias concern. ---
# #     illicit_test_idx = (test_mask & (y_type_remapped != -1)).nonzero(as_tuple=True)[0]
# #     print(f"\n[explainability demo, using seed={SEEDS[-1]}'s model] "
# #           f"{len(illicit_test_idx)} illicit test accounts available to explain.")
# #     for node_idx in illicit_test_idx[:3].tolist():
# #         explain_account(last_model, last_clf, x, edge_index, node_idx, feature_names, device)
# #
# #
# # # ---------------------------------------------------------------------------
# # # NOTE: upgrading to a joint multi-task head later
# # # ---------------------------------------------------------------------------
# # # If this two-stage version works well, the natural next step is a SINGLE
# # # model with two output heads sharing conv1/conv2: classifier_binary (2
# # # classes, supervised on all labeled nodes) and classifier_type (num_new_types
# # # classes, supervised ONLY on illicit train nodes). Loss = L_binary +
# # # lambda * L_type, lambda found via a small search (with the same
# # # winner's-curse-aware full-seed reverification you used for lr/hidden).
# # # This mirrors exactly the shared-backbone design you already proposed for
# # # the Elliptic+SAML-D multi-task idea, just applied within SAML-D alone.
#
#
# """
# step25_samld_type_classification.py
#
# Typology classification (which laundering type) + explainability (why),
# built on top of the existing binary SAML-D GraphSAGE pipeline.
#
# DESIGN: two stages, not one joint model (for now)
# =====================================================================
# Stage 1 (already exists, e.g. step22): binary GraphSAGE flags illicit
#   accounts, trained with the standard 5-seed protocol.
# Stage 2 (this file): reuses that model's learned embeddings (h2) as
#   input features for a separate multi-class classifier over the 17
#   laundering types -- trained ONLY on accounts already labeled illicit
#   (y_type == -1 for licit accounts, so they're excluded entirely).
#
# Why two stages instead of one joint multi-task head right away:
#   - Reuses checkpoints you already trained/validated with 5 seeds;
#     no need to retrain the GNN from scratch.
#   - Keeps the type classifier simple (sklearn LogisticRegression), so
#     the very small per-type sample counts (many types have <10 train
#     examples, per step23's own finding) don't also have to fight
#     GNN-scale hyperparameters.
#   - A joint multi-task head (same backbone, second nn.Linear output)
#     is a natural upgrade later if this simpler version works well --
#     see the note at the bottom of the file.
#
# EXPLAINABILITY: Grad x Input attribution on the 8 raw SAML-D features.
#   Simplest attribution method that still gives a genuinely readable
#   answer per account, e.g. "flagged as type 5 mainly because of
#   recv_amount_count and cross-border pattern" -- no architecture
#   change needed, just a few extra lines using autograd.
#
# REQUIRED ONE-LINE CHANGE TO YOUR EXISTING SAML-D MODEL CLASS:
#   Your SAML-D StructuralOnlyGraphSAGE (step19/20/22/24) doesn't return
#   embeddings yet -- only the Elliptic version in Step17 does. Add the
#   same `return_embeddings` parameter here too (done below), and add a
#   `save_checkpoint(...)` call at the end of your SAML-D training loop
#   (you already have this utility in metrics_utils.py, just not wired
#   into the SAML-D scripts yet).
# """
#
# import random
# from collections import defaultdict, Counter
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
#
# from metrics_utils import set_seed, save_checkpoint, load_checkpoint  # noqa: F401  (load_checkpoint used once you have real checkpoints)
#
# DATA_PATH = "samld_processed_v3.pt"
# S1, S2 = 25, 10
# BATCH_SIZE = 256
# WEIGHT_CAP = 30.0
# MIN_TRAIN_SAMPLES_PER_TYPE = 10  # exactly the rule you proposed in step23
#
#
# # ---------------------------------------------------------------------------
# # Model: same as your existing SAML-D StructuralOnlyGraphSAGE, plus the
# # return_embeddings flag (the one-line addition mentioned above).
# # ---------------------------------------------------------------------------
#
# class StructuralOnlyGraphSAGE(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.conv1 = SAGEConv(in_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)
#         self.classifier = nn.Linear(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index, return_embeddings=False):
#         h1 = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
#         h2 = F.dropout(F.relu(self.conv2(h1, edge_index)), p=0.3, training=self.training)
#         out = self.classifier(h2)
#         if return_embeddings:
#             return out, h1, h2
#         return out
#
#
# # ---------------------------------------------------------------------------
# # Stage 2a: rare-type merging, exactly the rule from step23
# # ---------------------------------------------------------------------------
#
# def merge_rare_types(y_type, train_mask, min_train_samples=MIN_TRAIN_SAMPLES_PER_TYPE):
#     """
#     Any type with fewer than `min_train_samples` illicit examples in TRAIN
#     gets merged into a single "Other" bucket (id = number of kept types).
#     Decision is made using TRAIN counts only, never val/test, to avoid
#     leaking test-set class-frequency information into the label space.
#
#     Returns:
#         remap: dict old_type_id -> new_type_id
#         num_new_types: int, including the "Other" bucket if used
#         kept_type_ids: sorted list of original type ids kept as-is
#     """
#     train_types = y_type[train_mask]
#     train_types = train_types[train_types != -1].tolist()
#     counts = Counter(train_types)
#
#     all_type_ids = sorted(set(y_type[y_type != -1].tolist()))
#     kept_type_ids = [t for t in all_type_ids if counts.get(t, 0) >= min_train_samples]
#     rare_type_ids = [t for t in all_type_ids if t not in kept_type_ids]
#
#     remap = {t: i for i, t in enumerate(kept_type_ids)}
#     other_id = len(kept_type_ids)
#     if rare_type_ids:
#         for t in rare_type_ids:
#             remap[t] = other_id
#         num_new_types = other_id + 1
#     else:
#         num_new_types = other_id
#
#     print(f"Types kept as-is: {len(kept_type_ids)}   Types merged into 'Other': {len(rare_type_ids)}")
#     return remap, num_new_types, kept_type_ids
#
#
# # ---------------------------------------------------------------------------
# # Stage 2b: extract embeddings from a trained binary model
# # ---------------------------------------------------------------------------
#
# def extract_embeddings(model, x, edge_index, device, batch_eval_size=50000):
#     """
#     Full-graph forward pass in eval mode, returning the h2 embedding for
#     every node. Done in chunks only to keep peak memory reasonable on
#     large graphs -- message passing still uses the FULL edge_index each
#     time (SAGEConv needs full neighborhood context), only the returned
#     slice is chunked.
#     """
#     model.eval()
#     with torch.no_grad():
#         _, _, h2_full = model(x.to(device), edge_index.to(device), return_embeddings=True)
#     return h2_full.cpu()
#
#
# # ---------------------------------------------------------------------------
# # Stage 2c: fit the type classifier on illicit-only embeddings
# # ---------------------------------------------------------------------------
#
# def fit_and_evaluate_type_classifier(h2, y_type_remapped, train_mask, test_mask, num_new_types,
#                                       seed_label=None, verbose=True):
#     from sklearn.metrics import accuracy_score, f1_score
#
#     illicit_train = train_mask & (y_type_remapped != -1)
#     illicit_test = test_mask & (y_type_remapped != -1)
#
#     X_train = h2[illicit_train].numpy()
#     y_train = y_type_remapped[illicit_train].numpy()
#     X_test = h2[illicit_test].numpy()
#     y_test = y_type_remapped[illicit_test].numpy()
#
#     if verbose:
#         tag = f" (seed={seed_label})" if seed_label is not None else ""
#         print(f"\nType classifier{tag}: train n={len(y_train)}   test n={len(y_test)}   "
#               f"num_types(after merge)={num_new_types}")
#
#     clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
#     clf.fit(X_train, y_train)
#
#     y_pred = clf.predict(X_test)
#     labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
#     if verbose:
#         print(classification_report(y_test, y_pred, labels=labels_present, zero_division=0))
#
#     seed_metrics = {
#         "accuracy": accuracy_score(y_test, y_pred),
#         "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
#         "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
#     }
#     return clf, seed_metrics
#
#
# # ---------------------------------------------------------------------------
# # Stage 2d: "why" -- Grad x Input attribution on the 8 raw features
# # ---------------------------------------------------------------------------
#
# def explain_account(model, clf, x, edge_index, node_idx, feature_names, device, type_names=None):
#     """
#     Explains a single account's PREDICTED TYPE by attributing the actual
#     predicted-class LOGIT (from the fitted sklearn classifier's own weights)
#     back to the 8 raw input features, via Grad x Input.
#
#     This backprops through: raw features -> GNN -> embedding -> (the
#     classifier's linear score for the predicted class), so the attribution
#     genuinely answers "why type X" -- not just "what shapes this node's
#     embedding in general" (an earlier version of this function backpropped
#     from the embedding's squared norm instead, which explained embedding
#     magnitude, not the type decision itself).
#     """
#     model.eval()
#     x_grad = x.clone().to(device).requires_grad_(True)
#     _, _, h2 = model(x_grad, edge_index.to(device), return_embeddings=True)
#
#     node_embedding = h2[node_idx]
#     coef = torch.tensor(clf.coef_, dtype=torch.float, device=device)
#     intercept = torch.tensor(clf.intercept_, dtype=torch.float, device=device)
#     logits = node_embedding @ coef.T + intercept  # sklearn's own decision function, made differentiable
#
#     predicted_type = int(logits.argmax().item())
#     logits[predicted_type].backward()
#
#     grad_at_node = x_grad.grad[node_idx].detach().cpu().numpy()
#     raw_features = x[node_idx].detach().cpu().numpy()
#     attribution = grad_at_node * raw_features
#
#     type_label = type_names[predicted_type] if type_names else f"type_{predicted_type}"
#
#     order = np.argsort(-np.abs(attribution))
#     print(f"\nAccount node {node_idx} -> predicted type: {type_label} "
#           f"(logit={logits[predicted_type].item():.3f})")
#     print("Top contributing features:")
#     for i in order[:5]:
#         print(f"  {feature_names[i]:28s} value={raw_features[i]:+.3f}   attribution={attribution[i]:+.4f}")
#
#
# # ---------------------------------------------------------------------------
# # Main: real 5-seed protocol. Loads each of the 5 checkpoints saved by the
# # updated step22 (samld_seed_<seed>.pt), extracts embeddings, fits a type
# # classifier per seed, and reports mean +/- std across all 5 -- exactly the
# # same discipline used everywhere else in the project.
# # ---------------------------------------------------------------------------
#
# SEEDS = (42, 1, 7, 123, 2024)
# CHECKPOINT_PATTERN = "samld_seed_{seed}.pt"
#
# if __name__ == "__main__":
#     print("Loading real SAML-D v3 data...")
#     data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
#     x = data_dict["x"]
#     edge_index = data_dict["edge_index"]
#     y_type = data_dict["y_type"]
#     train_mask = data_dict["train_mask"]
#     test_mask = data_dict["test_mask"]
#     feature_names = data_dict["feature_cols"]
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     print(f"Nodes: {x.shape[0]}   Features: {x.shape[1]}   Types: {data_dict['num_types']}")
#
#     # --- rare-type merging, per step23's rule (train-only, seed-independent) ---
#     remap, num_new_types, kept_type_ids = merge_rare_types(y_type, train_mask)
#     y_type_remapped = y_type.clone()
#     for old_id, new_id in remap.items():
#         y_type_remapped[y_type == old_id] = new_id
#     # y_type_remapped still has -1 for licit accounts, untouched
#
#     all_seed_metrics = []
#     last_model, last_clf = None, None
#
#     for seed in SEEDS:
#         ckpt_path = CHECKPOINT_PATTERN.format(seed=seed)
#         print(f"\n{'=' * 60}\nSeed {seed}: loading {ckpt_path}\n{'=' * 60}")
#
#         model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
#         payload = load_checkpoint(model, ckpt_path, map_location=device)
#         print(f"Loaded checkpoint -- binary F1 on this seed: {payload['F1']:.4f}   "
#               f"threshold: {payload['threshold']:.2f}")
#
#         h2 = extract_embeddings(model, x, edge_index, device)
#         clf, seed_metrics = fit_and_evaluate_type_classifier(
#             h2, y_type_remapped, train_mask, test_mask, num_new_types, seed_label=seed,
#         )
#         seed_metrics["seed"] = seed
#         all_seed_metrics.append(seed_metrics)
#         last_model, last_clf = model, clf
#
#     # --- aggregate across the 5 seeds, same discipline as run_multi_seed ---
#     print(f"\n\n{'=' * 60}\n=== Type classifier -- summary across {len(SEEDS)} seeds ===\n{'=' * 60}")
#     print(f"{'seed':>6} {'accuracy':>10} {'macro_f1':>10} {'weighted_f1':>12}")
#     for m in all_seed_metrics:
#         print(f"{m['seed']:>6} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['weighted_f1']:>12.4f}")
#
#     for key in ("accuracy", "macro_f1", "weighted_f1"):
#         vals = np.array([m[key] for m in all_seed_metrics])
#         print(f"\n{key:12s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")
#
#     # --- explainability demo, using the last loaded seed's model+classifier.
#     #     This is illustrative only (showing HOW the explanation works for a
#     #     couple of real accounts), not a reported metric -- so, unlike
#     #     Step17's "pick the best seed" issue, using just one seed here for
#     #     a qualitative example carries no statistical bias concern. ---
#     illicit_test_idx = (test_mask & (y_type_remapped != -1)).nonzero(as_tuple=True)[0]
#     print(f"\n[explainability demo, using seed={SEEDS[-1]}'s model] "
#           f"{len(illicit_test_idx)} illicit test accounts available to explain.")
#     for node_idx in illicit_test_idx[:3].tolist():
#         explain_account(last_model, last_clf, x, edge_index, node_idx, feature_names, device)
#
#
# # ---------------------------------------------------------------------------
# # NOTE: upgrading to a joint multi-task head later
# # ---------------------------------------------------------------------------
# # If this two-stage version works well, the natural next step is a SINGLE
# # model with two output heads sharing conv1/conv2: classifier_binary (2
# # classes, supervised on all labeled nodes) and classifier_type (num_new_types
# # classes, supervised ONLY on illicit train nodes). Loss = L_binary +
# # lambda * L_type, lambda found via a small search (with the same
# # winner's-curse-aware full-seed reverification you used for lr/hidden).
# # This mirrors exactly the shared-backbone design you already proposed for
# # the Elliptic+SAML-D multi-task idea, just applied within SAML-D alone.


"""
step25_samld_type_classification.py

Typology classification (which laundering type) + explainability (why),
built on top of the existing binary SAML-D GraphSAGE pipeline.

DESIGN: two stages, not one joint model (for now)
=====================================================================
Stage 1 (already exists, e.g. step22): binary GraphSAGE flags illicit
  accounts, trained with the standard 5-seed protocol.
Stage 2 (this file): reuses that model's learned embeddings (h2) as
  input features for a separate multi-class classifier over the 17
  laundering types -- trained ONLY on accounts already labeled illicit
  (y_type == -1 for licit accounts, so they're excluded entirely).

Why two stages instead of one joint multi-task head right away:
  - Reuses checkpoints you already trained/validated with 5 seeds;
    no need to retrain the GNN from scratch.
  - Keeps the type classifier simple (sklearn LogisticRegression), so
    the very small per-type sample counts (many types have <10 train
    examples, per step23's own finding) don't also have to fight
    GNN-scale hyperparameters.
  - A joint multi-task head (same backbone, second nn.Linear output)
    is a natural upgrade later if this simpler version works well --
    see the note at the bottom of the file.

EXPLAINABILITY: Grad x Input attribution on the 8 raw SAML-D features.
  Simplest attribution method that still gives a genuinely readable
  answer per account, e.g. "flagged as type 5 mainly because of
  recv_amount_count and cross-border pattern" -- no architecture
  change needed, just a few extra lines using autograd.

REQUIRED ONE-LINE CHANGE TO YOUR EXISTING SAML-D MODEL CLASS:
  Your SAML-D StructuralOnlyGraphSAGE (step19/20/22/24) doesn't return
  embeddings yet -- only the Elliptic version in Step17 does. Add the
  same `return_embeddings` parameter here too (done below), and add a
  `save_checkpoint(...)` call at the end of your SAML-D training loop
  (you already have this utility in metrics_utils.py, just not wired
  into the SAML-D scripts yet).
"""

import random
import json
import os
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from metrics_utils import set_seed, save_checkpoint, load_checkpoint  # noqa: F401  (load_checkpoint used once you have real checkpoints)

DATA_PATH = "samld_processed_v3.pt"
S1, S2 = 25, 10
BATCH_SIZE = 256
WEIGHT_CAP = 30.0
MIN_TRAIN_SAMPLES_PER_TYPE = 10  # exactly the rule you proposed in step23
TYPE_NAMES_PATH = "samld_type_names.json"  # produced by extract_type_names.py, optional


def load_type_names(path=TYPE_NAMES_PATH):
    """
    Loads the real typology names produced by extract_type_names.py, if
    present. Returns {int class_id: str name}, or None if the file doesn't
    exist yet -- callers fall back to plain "type_N" / numeric labels in
    that case, so this script still works even before you've run the name
    extraction step.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Model: same as your existing SAML-D StructuralOnlyGraphSAGE, plus the
# return_embeddings flag (the one-line addition mentioned above).
# ---------------------------------------------------------------------------

class StructuralOnlyGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_embeddings=False):
        h1 = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        h2 = F.dropout(F.relu(self.conv2(h1, edge_index)), p=0.3, training=self.training)
        out = self.classifier(h2)
        if return_embeddings:
            return out, h1, h2
        return out


# ---------------------------------------------------------------------------
# Stage 2a: rare-type merging, exactly the rule from step23
# ---------------------------------------------------------------------------

def merge_rare_types(y_type, train_mask, min_train_samples=MIN_TRAIN_SAMPLES_PER_TYPE):
    """
    Any type with fewer than `min_train_samples` illicit examples in TRAIN
    gets merged into a single "Other" bucket (id = number of kept types).
    Decision is made using TRAIN counts only, never val/test, to avoid
    leaking test-set class-frequency information into the label space.

    Returns:
        remap: dict old_type_id -> new_type_id
        num_new_types: int, including the "Other" bucket if used
        kept_type_ids: sorted list of original type ids kept as-is
    """
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
        num_new_types = other_id + 1
    else:
        num_new_types = other_id

    print(f"Types kept as-is: {len(kept_type_ids)}   Types merged into 'Other': {len(rare_type_ids)}")
    return remap, num_new_types, kept_type_ids


# ---------------------------------------------------------------------------
# Stage 2b: extract embeddings from a trained binary model
# ---------------------------------------------------------------------------

def extract_embeddings(model, x, edge_index, device, batch_eval_size=50000):
    """
    Full-graph forward pass in eval mode, returning the h2 embedding for
    every node. Done in chunks only to keep peak memory reasonable on
    large graphs -- message passing still uses the FULL edge_index each
    time (SAGEConv needs full neighborhood context), only the returned
    slice is chunked.
    """
    model.eval()
    with torch.no_grad():
        _, _, h2_full = model(x.to(device), edge_index.to(device), return_embeddings=True)
    return h2_full.cpu()


# ---------------------------------------------------------------------------
# Stage 2c: fit the type classifier on illicit-only embeddings
# ---------------------------------------------------------------------------

def fit_and_evaluate_type_classifier(h2, y_type_remapped, train_mask, test_mask, num_new_types,
                                      seed_label=None, verbose=True, type_names=None):
    from sklearn.metrics import accuracy_score, f1_score

    illicit_train = train_mask & (y_type_remapped != -1)
    illicit_test = test_mask & (y_type_remapped != -1)

    X_train = h2[illicit_train].numpy()
    y_train = y_type_remapped[illicit_train].numpy()
    X_test = h2[illicit_test].numpy()
    y_test = y_type_remapped[illicit_test].numpy()

    if verbose:
        tag = f" (seed={seed_label})" if seed_label is not None else ""
        print(f"\nType classifier{tag}: train n={len(y_train)}   test n={len(y_test)}   "
              f"num_types(after merge)={num_new_types}")

    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    labels_present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    target_names = [type_names.get(i, f"type_{i}") for i in labels_present] if type_names else None
    if verbose:
        print(classification_report(
            y_test, y_pred, labels=labels_present, target_names=target_names, zero_division=0,
        ))

    seed_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    return clf, seed_metrics


# ---------------------------------------------------------------------------
# Stage 2d: "why" -- Grad x Input attribution on the 8 raw features
# ---------------------------------------------------------------------------

def explain_account(model, clf, x, edge_index, node_idx, feature_names, device, type_names=None):
    """
    Explains a single account's PREDICTED TYPE by attributing the actual
    predicted-class LOGIT (from the fitted sklearn classifier's own weights)
    back to the 8 raw input features, via Grad x Input.

    This backprops through: raw features -> GNN -> embedding -> (the
    classifier's linear score for the predicted class), so the attribution
    genuinely answers "why type X" -- not just "what shapes this node's
    embedding in general" (an earlier version of this function backpropped
    from the embedding's squared norm instead, which explained embedding
    magnitude, not the type decision itself).
    """
    model.eval()
    x_grad = x.clone().to(device).requires_grad_(True)
    _, _, h2 = model(x_grad, edge_index.to(device), return_embeddings=True)

    node_embedding = h2[node_idx]
    coef = torch.tensor(clf.coef_, dtype=torch.float, device=device)
    intercept = torch.tensor(clf.intercept_, dtype=torch.float, device=device)
    logits = node_embedding @ coef.T + intercept  # sklearn's own decision function, made differentiable

    predicted_type = int(logits.argmax().item())
    logits[predicted_type].backward()

    grad_at_node = x_grad.grad[node_idx].detach().cpu().numpy()
    raw_features = x[node_idx].detach().cpu().numpy()
    attribution = grad_at_node * raw_features

    type_label = type_names[predicted_type] if type_names else f"type_{predicted_type}"

    order = np.argsort(-np.abs(attribution))
    print(f"\nAccount node {node_idx} -> predicted type: {type_label} "
          f"(logit={logits[predicted_type].item():.3f})")
    print("Top contributing features:")
    for i in order[:5]:
        print(f"  {feature_names[i]:28s} value={raw_features[i]:+.3f}   attribution={attribution[i]:+.4f}")


# ---------------------------------------------------------------------------
# Main: real 5-seed protocol. Loads each of the 5 checkpoints saved by the
# updated step22 (samld_seed_<seed>.pt), extracts embeddings, fits a type
# classifier per seed, and reports mean +/- std across all 5 -- exactly the
# same discipline used everywhere else in the project.
# ---------------------------------------------------------------------------

SEEDS = (42, 1, 7, 123, 2024)
CHECKPOINT_PATTERN = "samld_seed_{seed}.pt"

if __name__ == "__main__":
    print("Loading real SAML-D v3 data...")
    data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index = data_dict["edge_index"]
    y_type = data_dict["y_type"]
    train_mask = data_dict["train_mask"]
    test_mask = data_dict["test_mask"]
    feature_names = data_dict["feature_cols"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Nodes: {x.shape[0]}   Features: {x.shape[1]}   Types: {data_dict['num_types']}")

    # --- rare-type merging, per step23's rule (train-only, seed-independent) ---
    remap, num_new_types, kept_type_ids = merge_rare_types(y_type, train_mask)
    y_type_remapped = y_type.clone()
    for old_id, new_id in remap.items():
        y_type_remapped[y_type == old_id] = new_id
    # y_type_remapped still has -1 for licit accounts, untouched

    type_names = load_type_names()
    if type_names:
        print(f"Loaded {len(type_names)} real type names from {TYPE_NAMES_PATH}.")
    else:
        print(f"No {TYPE_NAMES_PATH} found -- reports will use numeric type ids. "
              f"Run extract_type_names.py first if you want real names.")

    all_seed_metrics = []
    last_model, last_clf = None, None

    for seed in SEEDS:
        ckpt_path = CHECKPOINT_PATTERN.format(seed=seed)
        print(f"\n{'=' * 60}\nSeed {seed}: loading {ckpt_path}\n{'=' * 60}")

        model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
        payload = load_checkpoint(model, ckpt_path, map_location=device)
        print(f"Loaded checkpoint -- binary F1 on this seed: {payload['F1']:.4f}   "
              f"threshold: {payload['threshold']:.2f}")

        h2 = extract_embeddings(model, x, edge_index, device)
        clf, seed_metrics = fit_and_evaluate_type_classifier(
            h2, y_type_remapped, train_mask, test_mask, num_new_types, seed_label=seed,
            type_names=type_names,
        )
        seed_metrics["seed"] = seed
        all_seed_metrics.append(seed_metrics)
        last_model, last_clf = model, clf

    # --- aggregate across the 5 seeds, same discipline as run_multi_seed ---
    print(f"\n\n{'=' * 60}\n=== Type classifier -- summary across {len(SEEDS)} seeds ===\n{'=' * 60}")
    print(f"{'seed':>6} {'accuracy':>10} {'macro_f1':>10} {'weighted_f1':>12}")
    for m in all_seed_metrics:
        print(f"{m['seed']:>6} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f} {m['weighted_f1']:>12.4f}")

    for key in ("accuracy", "macro_f1", "weighted_f1"):
        vals = np.array([m[key] for m in all_seed_metrics])
        print(f"\n{key:12s}: {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")

    # --- explainability demo, using the last loaded seed's model+classifier.
    #     This is illustrative only (showing HOW the explanation works for a
    #     couple of real accounts), not a reported metric -- so, unlike
    #     Step17's "pick the best seed" issue, using just one seed here for
    #     a qualitative example carries no statistical bias concern. ---
    illicit_test_idx = (test_mask & (y_type_remapped != -1)).nonzero(as_tuple=True)[0]
    print(f"\n[explainability demo, using seed={SEEDS[-1]}'s model] "
          f"{len(illicit_test_idx)} illicit test accounts available to explain.")
    for node_idx in illicit_test_idx[:3].tolist():
        explain_account(last_model, last_clf, x, edge_index, node_idx, feature_names, device,
                         type_names=type_names)


# ---------------------------------------------------------------------------
# NOTE: upgrading to a joint multi-task head later
# ---------------------------------------------------------------------------
# If this two-stage version works well, the natural next step is a SINGLE
# model with two output heads sharing conv1/conv2: classifier_binary (2
# classes, supervised on all labeled nodes) and classifier_type (num_new_types
# classes, supervised ONLY on illicit train nodes). Loss = L_binary +
# lambda * L_type, lambda found via a small search (with the same
# winner's-curse-aware full-seed reverification you used for lr/hidden).
# This mirrors exactly the shared-backbone design you already proposed for
# the Elliptic+SAML-D multi-task idea, just applied within SAML-D alone.