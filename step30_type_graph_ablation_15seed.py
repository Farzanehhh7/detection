"""
step30_type_graph_ablation_15seed.py

بازبینی نهایی سوال «آیا گراف SAML-D به تشخیص نوع کمک می‌کنه؟» با ۱۵ seed
به‌جای ۵ -- دقیقاً همون کاری که برای Elliptic (Step32) و برای همین سوال
(step25/26/28 نسخه‌ی ۱۵-seed) انجام شد.

اعداد embedding_only/embedding_structural/multi_task زیر، دقیقاً همون
per-seed macro_f1 واقعی‌ان که از اجرای واقعی step25_15seed،
step26_15seed، و step28_15seed به‌دست اومدن (ترتیب seed یکسان:
42, 1, 7, 123, 2024, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12).

NO GRAPH هم روی همون ۱۵ seed دوباره اجرا می‌شه (برخلاف نسخه‌ی قبلی که
فقط ۵ seed داشت) -- حتی اگه LogisticRegression با این solver نسبت به
random_state تغییر نکنه (چیزی که توی نسخه‌ی ۵-seed هم دیدیم، همه‌شون
دقیقاً ۰.۱۰۶۸ بودن)، حداقل صریح و دوباره تایید می‌شه، نه فرض گرفته می‌شه.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from step25_samld_type_classification import merge_rare_types

DATA_PATH = "samld_processed_v3.pt"
SEEDS = (42, 1, 7, 123, 2024, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)


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

    print(f"\n=== NO-GRAPH type classifier: raw 8 account features only ({len(SEEDS)} seeds) ===")
    no_graph_metrics = []
    for seed in SEEDS:
        m = fit_and_evaluate_no_graph(x, y_type_remapped, train_mask, test_mask, seed)
        no_graph_metrics.append(m)
        print(f"  seed={seed}  accuracy={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    print(f"\n{'=' * 70}\n=== FULL LADDER (15 seeds): how much does each graph ingredient add? ===\n{'=' * 70}")

    def summarize(vals, label):
        vals = np.array(vals)
        print(f"{label:45s} macro_f1 = {vals.mean():.4f} +/- {vals.std(ddof=1):.4f}")
        return vals

    no_graph_vals = summarize([m["macro_f1"] for m in no_graph_metrics], "NO GRAPH (raw 8 features only)")

    # اعداد واقعی از اجرای step25_15seed / step26_15seed / step28_15seed --
    # همون ترتیب seed بالا: 42, 1, 7, 123, 2024, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12
    embedding_only_vals = np.array([
        0.1100, 0.1088, 0.1295, 0.1074, 0.1252,
        0.1243, 0.1160, 0.0965, 0.0906, 0.1046,
        0.1154, 0.1049, 0.1103, 0.0953, 0.1454,
    ])
    embedding_structural_vals = np.array([
        0.1573, 0.1245, 0.1592, 0.1089, 0.1183,
        0.1088, 0.1363, 0.1096, 0.1375, 0.1190,
        0.1364, 0.1268, 0.1231, 0.1172, 0.1612,
    ])
    multitask_vals = np.array([
        0.2252, 0.1377, 0.1085, 0.1518, 0.1196,
        0.1797, 0.1168, 0.1311, 0.2036, 0.1665,
        0.1479, 0.1381, 0.1706, 0.1501, 0.1254,
    ])

    print(f"{'GNN embedding only (step25, 15-seed)':45s} macro_f1 = {embedding_only_vals.mean():.4f} +/- {embedding_only_vals.std(ddof=1):.4f}")
    print(f"{'GNN embedding + structural (step26, 15-seed)':45s} macro_f1 = {embedding_structural_vals.mean():.4f} +/- {embedding_structural_vals.std(ddof=1):.4f}")
    print(f"{'Multi-task shared backbone (step28, 15-seed)':45s} macro_f1 = {multitask_vals.mean():.4f} +/- {multitask_vals.std(ddof=1):.4f}")

    from scipy import stats
    print(f"\n--- Paired t-tests against NO GRAPH (df={len(SEEDS)-1}) ---")
    for label, vals in [("embedding_only", embedding_only_vals),
                         ("embedding+structural", embedding_structural_vals),
                         ("multi_task", multitask_vals)]:
        t, p = stats.ttest_rel(vals, no_graph_vals)
        delta = vals.mean() - no_graph_vals.mean()
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {label:25s} delta={delta:+.4f}  t={t:.3f}  p={p:.4f}  ({sig})")
