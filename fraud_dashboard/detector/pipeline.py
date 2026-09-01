"""
detector/pipeline.py

The Phase C pipeline: takes a newly-uploaded transaction CSV, runs it
through the full trained system, and updates AnalysisRun.status /
current_step / progress_pct at every stage so the frontend can poll and
show real progress -- not a fake spinner.

Scaler note: if step18v3 has been updated to save its StandardScaler via
`joblib.dump(scaler, "samld_scaler_v3.pkl")` and that file is placed next
to this project, this pipeline automatically loads and reuses it -- giving
correctly-calibrated predictions on new uploads. If that file isn't
present, it falls back to fitting a fresh scaler on the upload alone
(only approximate), and flags this clearly in the run's notes.
"""

import sys
import os
import threading
import traceback

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")


REQUIRED_COLUMNS = ["Sender_account", "Receiver_account", "Amount", "Payment_type"]


def _update(run, step, pct, status=None):
    run.current_step = step
    run.progress_pct = pct
    fields = ["current_step", "progress_pct"]
    if status is not None:
        run.status = status
        fields.append("status")
    run.save(update_fields=fields)


def process_upload(run_id, csv_path):
    """Runs the full pipeline for one uploaded CSV. Meant to be called in
    a background thread from the upload view."""
    from detector.models import AnalysisRun, Account, FeatureAttribution, NeighborInfluence, GraphEdge
    from metrics_utils import load_checkpoint
    from step25_samld_type_classification import (
        StructuralOnlyGraphSAGE, merge_rare_types, extract_embeddings, load_type_names,
    )
    from step26_samld_structural_features import compute_structural_features
    from step31_samld_coarse_family_classification import build_family_labels, FAMILY_NAMES

    run = AnalysisRun.objects.get(id=run_id)
    try:
        _update(run, "در حال خواندن فایل آپلودشده...", 5, status="running")

        df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"ستون‌های لازم موجود نیست: {missing}")

        _update(run, "در حال ساخت گراف حساب‌ها از تراکنش‌ها...", 15)
        all_accounts = pd.unique(df[["Sender_account", "Receiver_account"]].values.ravel())
        acc_to_idx = {acc: i for i, acc in enumerate(all_accounts)}
        num_nodes = len(all_accounts)

        src = df["Sender_account"].map(acc_to_idx).values
        dst = df["Receiver_account"].map(acc_to_idx).values
        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

        _update(run, "در حال محاسبه‌ی فیچرهای هر حساب...", 30)
        ptype_col = "Payment_type" if "Payment_type" in df.columns else None
        sent_stats = df.groupby("Sender_account")["Amount"].agg(["sum", "mean", "count"])
        sent_stats.columns = ["sent_amount_sum", "sent_amount_mean", "sent_amount_count"]
        recv_stats = df.groupby("Receiver_account")["Amount"].agg(["sum", "mean", "count"])
        recv_stats.columns = ["recv_amount_sum", "recv_amount_mean", "recv_amount_count"]

        if ptype_col:
            sent_ptype = df.groupby("Sender_account")[ptype_col].nunique().rename("sent_payment_type_nunique")
            recv_ptype = df.groupby("Receiver_account")[ptype_col].nunique().rename("recv_payment_type_nunique")
        else:
            sent_ptype = pd.Series(1.0, index=all_accounts, name="sent_payment_type_nunique")
            recv_ptype = pd.Series(1.0, index=all_accounts, name="recv_payment_type_nunique")

        feat_df = pd.concat([sent_stats, sent_ptype, recv_stats, recv_ptype], axis=1)
        feat_df = feat_df.reindex(all_accounts).fillna(0.0)
        feature_cols = list(feat_df.columns)

        # Use the REAL training-time scaler if step18v3 has saved one
        # (samld_scaler_v3.pkl); otherwise fall back to a fresh fit on this
        # upload alone, which is only approximate -- see module docstring.
        scaler_path = "samld_scaler_v3.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            x = torch.tensor(scaler.transform(feat_df.values), dtype=torch.float32)
            run.notes = "از scaler واقعی آموزش (samld_scaler_v3.pkl) استفاده شد."
        else:
            scaler = StandardScaler()
            x = torch.tensor(scaler.fit_transform(feat_df.values), dtype=torch.float32)
            run.notes = ("هشدار: samld_scaler_v3.pkl پیدا نشد -- یک scaler تقریبی روی همین "
                          "فایل آپلودی fit شد، نه scaler اصلی آموزش. نتایج را تقریبی در نظر بگیر.")
        run.save(update_fields=["notes"])

        _update(run, "در حال اجرای مدل GNN آموزش‌دیده روی گراف جدید...", 50)
        device = torch.device("cpu")
        model = StructuralOnlyGraphSAGE(in_channels=8, hidden_channels=64, out_channels=2).to(device)
        load_checkpoint(model, "samld_seed_42.pt", map_location=device)
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
            probs_all = F.softmax(out, dim=1)[:, 1].numpy()

        _update(run, "در حال طبقه‌بندی نوع پول‌شویی برای حساب‌های پرچم‌خورده...", 70)
        h2 = extract_embeddings(model, x, edge_index, device).numpy()
        struct_raw, _ = compute_structural_features(edge_index, num_nodes)
        struct_scaled = StandardScaler().fit_transform(struct_raw)
        X_combined = np.concatenate([h2, struct_scaled], axis=1)

        type_names = load_type_names() or {}
        flagged_idx = np.where(probs_all >= 0.5)[0]

        # NOTE: without real illicit-labeled examples in a fresh upload, we
        # reuse the classifiers already fit during import_saml_results if
        # available; otherwise type/family are left blank for this run.
        type_clf, family_clf = _load_reference_classifiers()

        _update(run, "در حال ذخیره‌ی نتایج...", 85)
        Account.objects.filter(run=run).delete()
        accounts_to_create = []
        for node_id in range(num_nodes):
            feats = feat_df.values[node_id]
            prob = float(probs_all[node_id])
            is_flagged = bool(prob >= 0.5)
            pred_type, pred_family = "", ""
            if is_flagged and type_clf is not None:
                vec = X_combined[node_id:node_id + 1]
                pred_type = type_names.get(int(type_clf.predict(vec)[0]), "")
                pred_family = FAMILY_NAMES.get(int(family_clf.predict(vec)[0]), "")
            accounts_to_create.append(Account(
                run=run, node_id=node_id,
                sent_amount_sum=float(feats[0]), sent_amount_mean=float(feats[1]),
                sent_amount_count=float(feats[2]), sent_payment_type_nunique=float(feats[3]),
                recv_amount_sum=float(feats[4]), recv_amount_mean=float(feats[5]),
                recv_amount_count=float(feats[6]), recv_payment_type_nunique=float(feats[7]),
                prob_illicit=prob, is_flagged=is_flagged,
                predicted_type=pred_type, predicted_family=pred_family,
            ))
        Account.objects.bulk_create(accounts_to_create, batch_size=1000)

        GraphEdge.objects.filter(run=run).delete()
        GraphEdge.objects.bulk_create([
            GraphEdge(run=run, source_node_id=int(s), target_node_id=int(d))
            for s, d in zip(src.tolist(), dst.tolist())
        ], batch_size=2000)

        _update(run, "در حال محاسبه‌ی توضیح برای حساب‌های برتر...", 92)
        node_lookup = {a.node_id: a for a in Account.objects.filter(run=run)}
        top_flagged = sorted(flagged_idx.tolist(), key=lambda i: -probs_all[i])[:50]
        attributions_to_create, neighbor_infos_to_create = [], []
        for node_id in top_flagged:
            x_grad = x.clone().requires_grad_(True)
            out2 = model(x_grad, edge_index)
            out2[node_id, 1].backward()
            grad = x_grad.grad

            own_grad = grad[node_id].numpy()
            own_feat = x[node_id].numpy()
            own_attr = own_grad * own_feat
            order = np.argsort(-np.abs(own_attr))
            acc = node_lookup[node_id]
            for rank, i in enumerate(order[:5]):
                attributions_to_create.append(FeatureAttribution(
                    account=acc, feature_name=feature_cols[i],
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
                    neighbor_is_flagged=bool(probs_all[n_idx] >= 0.5), rank=rank,
                ))
        FeatureAttribution.objects.bulk_create(attributions_to_create, batch_size=1000)
        NeighborInfluence.objects.bulk_create(neighbor_infos_to_create, batch_size=1000)

        _update(run, f"تمام شد — {len(flagged_idx)} حساب از {num_nodes} پرچم خورد.", 100, status="done")

    except Exception as e:
        run.status = "failed"
        run.current_step = f"خطا: {e}"
        run.notes = traceback.format_exc()
        run.save(update_fields=["status", "current_step", "notes"])


def _load_reference_classifiers():
    """Reuses the type/family classifiers already fit on the reference
    pretrained run, if one exists, so a fresh upload doesn't need its own
    (nonexistent) labeled illicit examples to train a classifier from."""
    import pickle
    import os
    if os.path.exists("reference_classifiers.pkl"):
        with open("reference_classifiers.pkl", "rb") as f:
            data = pickle.load(f)
        return data["type_clf"], data["family_clf"]
    return None, None


def process_upload_async(run_id, csv_path):
    thread = threading.Thread(target=process_upload, args=(run_id, csv_path), daemon=True)
    thread.start()
