"""
manage.py import_saml_results

Populates the dashboard with REAL results from the actual trained
pipeline -- no fake/demo data. Reuses:
  - the seed=42 binary checkpoint from step22
  - the type classifier logic from step25/step26
  - the family classifier logic from step31
  - the explainability logic from step33

To keep the import fast for a first working version, this imports:
  - all real illicit test accounts (the ones the model actually flags as
    suspicious are drawn from here)
  - a random sample of licit test accounts, for contrast/browsing
  - the real transaction edges among all imported accounts, for the
    graph visualization
"""

import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from django.core.management.base import BaseCommand
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from detector.models import AnalysisRun, Account, FeatureAttribution, NeighborInfluence, GraphEdge


class Command(BaseCommand):
    help = "Import real SAML-D pipeline results into the dashboard"

    def add_arguments(self, parser):
        parser.add_argument("--n-licit-sample", type=int, default=1500)
        parser.add_argument("--n-explain", type=int, default=100,
                             help="how many top-flagged accounts get full own-feature + neighbor explanations")

    def handle(self, *args, **options):
        import sys
        sys.path.insert(0, ".")
        from metrics_utils import load_checkpoint
        from step25_samld_type_classification import (
            StructuralOnlyGraphSAGE, merge_rare_types, extract_embeddings, load_type_names,
        )
        from step26_samld_structural_features import compute_structural_features
        from step31_samld_coarse_family_classification import build_family_labels, FAMILY_NAMES

        self.stdout.write("Loading real SAML-D v3 data...")
        data = torch.load("samld_processed_v3.pt", map_location="cpu", weights_only=False)
        x = data["x"]
        edge_index = data["edge_index"]
        y_binary = data["y_binary"]
        y_type = data["y_type"]
        test_mask = data["test_mask"]
        train_mask = data["train_mask"]
        feature_names = data["feature_cols"]

        device = torch.device("cpu")
        model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
        load_checkpoint(model, "samld_seed_42.pt", map_location=device)
        model.eval()

        self.stdout.write("Running inference on the full graph (one forward pass)...")
        with torch.no_grad():
            out = model(x.to(device), edge_index.to(device))
            probs_all = F.softmax(out, dim=1)[:, 1].numpy()

        # --- fit type + family classifiers on illicit train accounts (same as step25/26/31) ---
        self.stdout.write("Fitting type and family classifiers...")
        h2 = extract_embeddings(model, x, edge_index, device).numpy()
        struct_raw, _ = compute_structural_features(edge_index, x.shape[0])
        illicit_train_np = (train_mask & (y_type != -1)).numpy()
        scaler = StandardScaler().fit(struct_raw[illicit_train_np])
        struct_scaled = scaler.transform(struct_raw)
        X_combined = np.concatenate([h2, struct_scaled], axis=1)

        remap, num_new_types, _ = merge_rare_types(y_type, train_mask)
        y_type_remapped = y_type.clone()
        for old_id, new_id in remap.items():
            y_type_remapped[y_type == old_id] = new_id
        type_names = load_type_names() or {}

        type_illicit_train = (train_mask & (y_type_remapped != -1)).numpy()
        type_clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        type_clf.fit(X_combined[type_illicit_train], y_type_remapped[train_mask & (y_type_remapped != -1)].numpy())

        y_family = build_family_labels(y_type)
        fam_illicit_train = (train_mask & (y_family != -1)).numpy()
        family_clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        family_clf.fit(X_combined[fam_illicit_train], y_family[train_mask & (y_family != -1)].numpy())

        import pickle
        with open("reference_classifiers.pkl", "wb") as f:
            pickle.dump({"type_clf": type_clf, "family_clf": family_clf}, f)
        self.stdout.write(self.style.SUCCESS("Saved reference_classifiers.pkl for reuse on new uploads."))

        # --- choose which accounts to import ---
        illicit_test_idx = (test_mask & (y_binary == 1)).nonzero(as_tuple=True)[0].tolist()
        licit_test_idx_all = (test_mask & (y_binary == 0)).nonzero(as_tuple=True)[0].tolist()
        random.seed(42)
        licit_sample = random.sample(licit_test_idx_all, min(options["n_licit_sample"], len(licit_test_idx_all)))
        import_ids = set(illicit_test_idx) | set(licit_sample)
        self.stdout.write(f"Importing {len(import_ids)} accounts "
                           f"({len(illicit_test_idx)} illicit test + {len(licit_sample)} licit sample)...")

        run = AnalysisRun.objects.create(
            name="SAML-D v3, seed=42 checkpoint (real pipeline)",
            status="done",
            notes="Imported via import_saml_results management command.",
        )

        node_id_to_account = {}
        accounts_to_create = []
        for node_id in import_ids:
            feats = x[node_id].numpy()
            prob = float(probs_all[node_id])
            is_flagged = prob >= 0.5  # simple default; percentile-based flagging can be layered in the UI later

            pred_type, pred_family = "", ""
            if is_flagged:
                feat_vec = X_combined[node_id:node_id + 1]
                type_id = int(type_clf.predict(feat_vec)[0])
                pred_type = type_names.get(type_id, f"type_{type_id}")
                fam_id = int(family_clf.predict(feat_vec)[0])
                pred_family = FAMILY_NAMES.get(fam_id, f"family_{fam_id}")

            accounts_to_create.append(Account(
                run=run, node_id=node_id,
                sent_amount_sum=float(feats[0]), sent_amount_mean=float(feats[1]),
                sent_amount_count=float(feats[2]), sent_payment_type_nunique=float(feats[3]),
                recv_amount_sum=float(feats[4]), recv_amount_mean=float(feats[5]),
                recv_amount_count=float(feats[6]), recv_payment_type_nunique=float(feats[7]),
                prob_illicit=prob, is_flagged=is_flagged,
                actual_label=int(y_binary[node_id].item()),
                predicted_type=pred_type, predicted_family=pred_family,
            ))
        Account.objects.bulk_create(accounts_to_create, batch_size=1000)
        for acc in Account.objects.filter(run=run):
            node_id_to_account[acc.node_id] = acc
        self.stdout.write(self.style.SUCCESS(f"Created {len(accounts_to_create)} accounts."))

        # --- real transaction edges among imported accounts ---
        self.stdout.write("Importing real edges among selected accounts...")
        src_all, dst_all = edge_index[0].tolist(), edge_index[1].tolist()
        edges_to_create = []
        for s, d in zip(src_all, dst_all):
            if s in import_ids and d in import_ids:
                edges_to_create.append(GraphEdge(run=run, source_node_id=s, target_node_id=d))
        GraphEdge.objects.bulk_create(edges_to_create, batch_size=2000)
        self.stdout.write(self.style.SUCCESS(f"Created {len(edges_to_create)} edges."))

        # --- explainability for the top-N flagged accounts (own-feature + neighbor influence) ---
        n_explain = options["n_explain"]
        top_flagged = sorted(illicit_test_idx, key=lambda i: -probs_all[i])[:n_explain]
        self.stdout.write(f"Computing explanations for the top {len(top_flagged)} flagged accounts...")

        attributions_to_create, neighbor_infos_to_create = [], []
        for node_id in top_flagged:
            x_grad = x.clone().requires_grad_(True)
            out = model(x_grad, edge_index)
            out[node_id, 1].backward()
            grad = x_grad.grad

            own_grad = grad[node_id].numpy()
            own_feat = x[node_id].numpy()
            own_attr = own_grad * own_feat
            order = np.argsort(-np.abs(own_attr))
            acc = node_id_to_account[node_id]
            for rank, i in enumerate(order[:5]):
                attributions_to_create.append(FeatureAttribution(
                    account=acc, feature_name=feature_names[i],
                    value=float(own_feat[i]), attribution=float(own_attr[i]), rank=rank,
                ))

            grad_mag = grad.abs().sum(dim=1).numpy()
            grad_mag[node_id] = 0.0
            top_neighbors = np.argsort(-grad_mag)[:5]
            for rank, n_idx in enumerate(top_neighbors):
                if grad_mag[n_idx] <= 0:
                    continue
                neighbor_infos_to_create.append(NeighborInfluence(
                    account=acc, neighbor_node_id=int(n_idx), influence=float(grad_mag[n_idx]),
                    neighbor_is_flagged=bool(y_binary[n_idx].item() == 1), rank=rank,
                ))

        FeatureAttribution.objects.bulk_create(attributions_to_create, batch_size=1000)
        NeighborInfluence.objects.bulk_create(neighbor_infos_to_create, batch_size=1000)
        self.stdout.write(self.style.SUCCESS(
            f"Done. {len(attributions_to_create)} feature attributions, "
            f"{len(neighbor_infos_to_create)} neighbor influences."
        ))
