"""
experiment_full_learnable_adjacency_model.py

نسخه‌ی کامل و واقعی از آزمایش Learnable Adjacency -- برخلاف سه تست ارزون
قبلی (که فقط سیگنال nearest-neighbor خام رو سنجیدن)، این‌جا دقیقاً همون
سوالی که مهمه رو تست می‌کنیم: آیا یال‌های استنتاجی (behavioral similarity)
دقیقاً همون‌جایی که گراف تراکنش واقعی چیزی نداره بگه (حساب‌های
low-degree) کمک می‌کنن؟ این دقیقاً همون معیاریه که evaluate_by_degree_bucket
توی learnable_adjacency.py طراحی شده بسنجه.

نسخه‌ی دوم (بازبینی): NeighborLoader استاندارد PyG به pyg-lib یا
torch-sparse نیاز داشت که روی سیستم کاربر نصب نبود و نصبش شکننده‌ست.
این‌جا برگشتیم به همون sampling دستی dict-based که step22 استفاده کرد
(بدون هیچ dependency جدید)، فقط گسترش‌یافته که edge_attr (پرچم
واقعی/استنتاجی + وزن شباهت) هم از بین دو hop رد بشه، نه فقط node id.

دو مدل، هر دو با معماری یکسان (EdgeAwareGraphSAGEPlus)، تا فرق بین‌شون
فقط "آیا یال استنتاجی دارن یا نه" باشه:

  BASELINE  -- فقط یال‌های واقعی تراکنش (edge_attr همیشه [0, 1])
  ENHANCED  -- یال‌های واقعی + یال‌های استنتاجی (behavioral top-k)

هر دو روی همون ۵ seed استاندارد پروژه train می‌شن، بعد با
evaluate_by_degree_bucket جدا برای حساب‌های low-degree (<=1 یال واقعی)
و بقیه سنجیده می‌شن.

هشدار صادقانه: این اولین اجرای واقعی این معماریه. اگه نتیجه مثبت بود،
باید حتماً یک بار دیگه با seed بیشتر قطعی بشه، نه فقط ۵ seed.
"""

import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from metrics_utils import set_seed, find_best_threshold, paired_significance_test
from learnable_adjacency import (
    top_k_inferred_edges, merge_real_and_inferred_edges,
    EdgeAwareGraphSAGEPlus, evaluate_by_degree_bucket,
)

DATA_PATH = "samld_processed_v3.pt"
BEHAVIOURAL_PATH = "samld_behavioural_features.pt"
SEEDS = (42, 1, 7, 123, 2024)
EPOCHS = 15
S1, S2 = 15, 10  # تعداد نمونه‌ی همسایه در هاپ اول و دوم، دقیقاً مثل step22
BATCH_SIZE = 256
TOP_K_INFERRED = 10
SIM_THRESHOLD = 0.5
LOW_DEGREE_THRESHOLD = 1


def build_adjacency(edge_index, edge_attr):
    """adjacency[dst] = [(src, is_inferred, weight), ...] -- دقیقاً مثل
    الگوی step22 (adjacency[d].append(s))، فقط هر ورودی edge_attr هم حمل می‌کنه."""
    adjacency = defaultdict(list)
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    is_inf = edge_attr[:, 0].tolist()
    w = edge_attr[:, 1].tolist()
    for s, d, i, wt in zip(src, dst, is_inf, w):
        adjacency[d].append((s, i, wt))
    return adjacency


def sample_k(neighbors, k):
    return neighbors if len(neighbors) <= k else random.sample(neighbors, k)


def build_batch_subgraph(seed_nodes, adjacency):
    hop1_of, all_hop1 = {}, set()
    for s in seed_nodes:
        neighs = sample_k(adjacency.get(s, []), S1)
        hop1_of[s] = neighs
        all_hop1.update(n[0] for n in neighs)

    hop2_of, all_hop2 = {}, set()
    for n in all_hop1:
        neighs2 = sample_k(adjacency.get(n, []), S2)
        hop2_of[n] = neighs2
        all_hop2.update(n2[0] for n2 in neighs2)

    all_nodes = list(set(seed_nodes) | all_hop1 | all_hop2)
    local_id = {n: i for i, n in enumerate(all_nodes)}

    edges_src, edges_dst, edge_is_inf, edge_w = [], [], [], []
    for s in seed_nodes:
        for (src_node, is_inf, w) in hop1_of[s]:
            edges_src.append(local_id[src_node]); edges_dst.append(local_id[s])
            edge_is_inf.append(is_inf); edge_w.append(w)
    for n in all_hop1:
        for (src_node2, is_inf2, w2) in hop2_of[n]:
            edges_src.append(local_id[src_node2]); edges_dst.append(local_id[n])
            edge_is_inf.append(is_inf2); edge_w.append(w2)

    node_idx_tensor = torch.tensor(all_nodes, dtype=torch.long)
    if edges_src:
        sub_edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        sub_edge_attr = torch.tensor(list(zip(edge_is_inf, edge_w)), dtype=torch.float)
    else:
        sub_edge_index = torch.zeros((2, 0), dtype=torch.long)
        sub_edge_attr = torch.zeros((0, 2), dtype=torch.float)
    seed_local_idx = torch.tensor([local_id[s] for s in seed_nodes], dtype=torch.long)
    return node_idx_tensor, sub_edge_index, sub_edge_attr, seed_local_idx


def train_one_seed(x, adjacency, y, train_node_ids_master, class_weight, seed, in_channels, device):
    set_seed(seed)
    random.seed(seed)
    train_node_ids = list(train_node_ids_master)

    model = EdgeAwareGraphSAGEPlus(in_channels=in_channels, hidden_channels=32,
                                    out_channels=2, edge_dim=2, heads=4, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight.to(device))

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train_node_ids)
        model.train()
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(train_node_ids), BATCH_SIZE):
            seed_nodes = train_node_ids[i:i + BATCH_SIZE]
            node_idx, sub_edge_index, sub_edge_attr, seed_local_idx = build_batch_subgraph(seed_nodes, adjacency)
            sub_x = x[node_idx].to(device)
            sub_edge_index = sub_edge_index.to(device)
            sub_edge_attr = sub_edge_attr.to(device)
            seed_local_idx = seed_local_idx.to(device)
            batch_y = y[seed_nodes].to(device)

            optimizer.zero_grad()
            out = model(sub_x, sub_edge_index, sub_edge_attr)
            loss = criterion(out[seed_local_idx], batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 5 == 0:
            print(f"    epoch={epoch:02d}  avg_loss={total_loss / max(n_batches, 1):.4f}")

    return model


@torch.no_grad()
def predict_nodes(model, x, adjacency, node_ids, device, batch_size=1024):
    model.eval()
    all_probs = np.zeros(len(node_ids), dtype=np.float32)
    node_ids_list = list(node_ids)
    for i in range(0, len(node_ids_list), batch_size):
        batch_nodes = node_ids_list[i:i + batch_size]
        node_idx, sub_edge_index, sub_edge_attr, seed_local_idx = build_batch_subgraph(batch_nodes, adjacency)
        sub_x = x[node_idx].to(device)
        sub_edge_index = sub_edge_index.to(device)
        sub_edge_attr = sub_edge_attr.to(device)
        seed_local_idx = seed_local_idx.to(device)
        out = model(sub_x, sub_edge_index, sub_edge_attr)
        probs = F.softmax(out[seed_local_idx], dim=1)[:, 1].cpu().numpy()
        all_probs[i:i + len(batch_nodes)] = probs
    return all_probs


def run_condition(label, x, edge_index, edge_attr, y, real_degree,
                   train_mask_np, val_mask_np, test_mask_np, in_channels, device):
    print(f"\n{'=' * 70}\n=== {label} ===\n{'=' * 70}")
    print("  در حال ساخت adjacency dict (یک بار، برای هر ۵ seed استفاده می‌شه)...")
    adjacency = build_adjacency(edge_index, edge_attr)

    y_np = y.numpy()
    train_node_ids_master = np.where(train_mask_np)[0].tolist()
    val_node_ids = np.where(val_mask_np)[0].tolist()
    test_node_ids = np.where(test_mask_np)[0].tolist()

    n_pos = (y_np[train_mask_np] == 1).sum()
    n_neg = (y_np[train_mask_np] == 0).sum()
    class_weight = torch.tensor([1.0, min(n_neg / max(n_pos, 1), 30.0)])

    low_degree_results, rest_results = [], []
    for seed in SEEDS:
        print(f"  seed={seed}")
        model = train_one_seed(x, adjacency, y, train_node_ids_master, class_weight,
                                seed, in_channels, device)

        val_probs = predict_nodes(model, x, adjacency, val_node_ids, device)
        best_t, _ = find_best_threshold(y_np[val_node_ids], val_probs)

        test_probs = predict_nodes(model, x, adjacency, test_node_ids, device)
        test_ids_arr = np.array(test_node_ids)

        report = evaluate_by_degree_bucket(
            y_np[test_ids_arr], test_probs, real_degree[test_ids_arr],
            low_degree_threshold=LOW_DEGREE_THRESHOLD,
        )
        low = report.get("low_degree")
        rest = report.get("rest")
        print(f"    low-degree (n={low['n'] if low else 0}): {low}")
        print(f"    rest       (n={rest['n'] if rest else 0}): {rest}")
        if low and "auc" in low:
            low_degree_results.append(low)
        if rest and "auc" in rest:
            rest_results.append(rest)

    return low_degree_results, rest_results


def summarize(results, label):
    if not results:
        print(f"{label}: هیچ seed معتبری نداشت")
        return None
    aucs = np.array([r["auc"] for r in results])
    pr_aucs = np.array([r["pr_auc"] for r in results])
    std_auc = aucs.std(ddof=1) if len(aucs) > 1 else 0.0
    std_pr = pr_aucs.std(ddof=1) if len(pr_aucs) > 1 else 0.0
    print(f"{label}: AUC={aucs.mean():.4f}±{std_auc:.4f}   "
          f"PR-AUC={pr_aucs.mean():.4f}±{std_pr:.4f}   n_seeds={len(results)}")
    return aucs, pr_aucs


if __name__ == "__main__":
    device = torch.device("cpu")
    print("در حال بارگذاری داده...")
    data_dict = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data_dict["x"]
    edge_index_real = data_dict["edge_index"]
    y = data_dict["y_binary"]
    train_mask = data_dict["train_mask"]
    val_mask = data_dict["val_mask"]
    test_mask = data_dict["test_mask"]

    behavioural = torch.load(BEHAVIOURAL_PATH, map_location="cpu", weights_only=False)
    behavior_vec = torch.cat([x, behavioural["features_scaled"]], dim=1).numpy().astype(np.float32)

    n_nodes = x.shape[0]
    real_degree = np.bincount(edge_index_real.numpy().flatten(), minlength=n_nodes)
    print(f"تعداد گره: {n_nodes}   میانگین درجه‌ی واقعی: {real_degree.mean():.2f}   "
          f"درصد گره‌های low-degree (<={LOW_DEGREE_THRESHOLD}): "
          f"{100 * (real_degree <= LOW_DEGREE_THRESHOLD).mean():.1f}%")

    print(f"\nدر حال ساخت یال‌های استنتاجی (top-{TOP_K_INFERRED}، آستانه شباهت={SIM_THRESHOLD})...")
    edge_index_inferred, edge_sim = top_k_inferred_edges(
        behavior_vec, k=TOP_K_INFERRED, sim_threshold=SIM_THRESHOLD,
    )
    print(f"{edge_index_inferred.shape[1]} یال استنتاجی ساخته شد")

    edge_index_real_only, edge_attr_real_only = merge_real_and_inferred_edges(
        edge_index_real, torch.zeros((2, 0), dtype=torch.long), torch.zeros(0),
    )
    edge_index_merged, edge_attr_merged = merge_real_and_inferred_edges(
        edge_index_real, edge_index_inferred, edge_sim,
    )

    train_mask_np, val_mask_np, test_mask_np = train_mask.numpy(), val_mask.numpy(), test_mask.numpy()
    in_channels = x.shape[1]

    baseline_low, baseline_rest = run_condition(
        "BASELINE -- فقط یال‌های واقعی (GATv2 edge-aware)",
        x, edge_index_real_only, edge_attr_real_only, y, real_degree,
        train_mask_np, val_mask_np, test_mask_np, in_channels, device,
    )
    enhanced_low, enhanced_rest = run_condition(
        "ENHANCED -- یال‌های واقعی + استنتاجی (Learnable Adjacency)",
        x, edge_index_merged, edge_attr_merged, y, real_degree,
        train_mask_np, val_mask_np, test_mask_np, in_channels, device,
    )

    print(f"\n\n{'=' * 70}\n=== نتیجه‌ی نهایی: دقیقاً روی حساب‌های low-degree ===\n{'=' * 70}")
    b_low = summarize(baseline_low, "BASELINE  (فقط یال واقعی)  low-degree")
    e_low = summarize(enhanced_low, "ENHANCED  (+ یال استنتاجی)  low-degree")

    print(f"\n--- برای مقایسه، همون دو مدل روی بقیه‌ی حساب‌ها ---")
    summarize(baseline_rest, "BASELINE  رست")
    summarize(enhanced_rest, "ENHANCED  رست")

    if b_low is not None and e_low is not None and len(baseline_low) == len(enhanced_low) == len(SEEDS):
        print(f"\n=== آزمون معناداری paired، فقط روی زیرمجموعه‌ی low-degree، AUC ===")
        paired_significance_test(list(e_low[0]), list(b_low[0]),
                                  "ENHANCED (low-degree AUC)", "BASELINE (low-degree AUC)")
    else:
        print("\nتعداد seedهای معتبر برابر نبود -- آزمون معناداری paired قابل‌اجرا نیست.")