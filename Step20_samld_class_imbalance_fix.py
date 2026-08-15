"""
فاز سه — رفع ناپایداری class weight=330
=====================================================================
سه رویکرد رو روی همون گراف جهت‌دار مقایسه می‌کنه: وزن خام فعلی که
احتمالاً باعث F1=0 در آزمایش undirected شده، وزن با سقف محدود، و
focal loss که اصلاً از weight ثابت استفاده نمی‌کنه و به‌جاش نمونه‌های
سخت رو خودش پیدا می‌کنه. هر سه فقط یک seed، برای مقایسه ارزون قبل از
پروتکل کامل.
"""

import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from metrics_utils import evaluate_binary, find_best_threshold, set_seed, run_multi_seed

# DATA_PATH = "samld_processed.pt"
DATA_PATH = "samld_processed_v3.pt"
S1, S2 = 25, 10
BATCH_SIZE = 256
RAW_WEIGHT_CAP = 30.0  # سقف پیشنهادی؛ ۳۳۰ خام در برابر این سقف

set_seed(42)
random.seed(42)

print("در حال بارگذاری داده پردازش‌شده SAML-D...")
data_dict = torch.load(DATA_PATH, weights_only=False)
x = data_dict["x"]
edge_index = data_dict["edge_index"]
y = data_dict["y_binary"]
train_mask = data_dict["train_mask"]
val_mask = data_dict["val_mask"]
test_mask = data_dict["test_mask"]

n_pos = (y[train_mask] == 1).sum().item()
n_neg = (y[train_mask] == 0).sum().item()
raw_weight = n_neg / max(n_pos, 1)
capped_weight = min(raw_weight, RAW_WEIGHT_CAP)
print(f"وزن خام: {raw_weight:.2f}   وزن با سقف: {capped_weight:.2f}")

adjacency = defaultdict(list)
for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
    adjacency[d].append(s)


def sample_k(neighbors, k):
    return neighbors if len(neighbors) <= k else random.sample(neighbors, k)


def build_batch_subgraph(seed_nodes):
    hop1_of, all_hop1 = {}, set()
    for s in seed_nodes:
        neighs = sample_k(adjacency.get(s, []), S1)
        hop1_of[s] = neighs
        all_hop1.update(neighs)

    hop2_of, all_hop2 = {}, set()
    for n in all_hop1:
        neighs = sample_k(adjacency.get(n, []), S2)
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


class StructuralOnlyGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), p=0.3, training=self.training)
        return self.classifier(h)


class FocalLoss(nn.Module):
    """گاما بالاتر یعنی تمرکز بیشتر روی نمونه‌های سخت‌اشتباه؛ آلفا وزن نسبی هر کلاس، ولی بدون انفجار عددی weight خام."""
    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.alpha[targets]
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
train_node_ids_master = train_mask.nonzero(as_tuple=True)[0].tolist()


def run_one_seed(seed, criterion_name):
    set_seed(seed)
    random.seed(seed)
    train_node_ids = list(train_node_ids_master)

    model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)

    if criterion_name == "raw_weight":
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, raw_weight]).to(device))
    elif criterion_name == "capped_weight":
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, capped_weight]).to(device))
    else:  # focal
        criterion = FocalLoss(alpha=torch.tensor([1.0, capped_weight]).to(device), gamma=2.0)

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(train_node_ids)
        model.train()
        for i in range(0, len(train_node_ids), BATCH_SIZE):
            seed_nodes = train_node_ids[i:i + BATCH_SIZE]
            sub_x, sub_edge_index, seed_local_idx = build_batch_subgraph(seed_nodes)
            sub_x, sub_edge_index = sub_x.to(device), sub_edge_index.to(device)
            seed_local_idx = seed_local_idx.to(device)
            batch_y = y[seed_nodes].to(device)

            optimizer.zero_grad()
            out = model(sub_x, sub_edge_index)
            loss = criterion(out[seed_local_idx], batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x.to(device), edge_index.to(device))
        probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    y_val = y[val_mask].numpy()
    probs_val = probs_all[val_mask.numpy()]
    best_t, _ = find_best_threshold(y_val, probs_val)

    y_test = y[test_mask].numpy()
    probs_test = probs_all[test_mask.numpy()]
    preds_test = (probs_test >= best_t).astype(int)

    metrics = evaluate_binary(criterion_name, y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


print("\n=== مقایسه سه رویکرد عدم توازن، هرکدام یک seed ===")
for name in ["raw_weight", "capped_weight", "focal"]:
    m = run_one_seed(42, name)
    print(f"{name:16s} F1={m['F1']:.4f}  Precision={m['Precision']:.4f}  "
          f"Recall={m['Recall']:.4f}  PR-AUC={m['PR-AUC']:.4f}  threshold={m['threshold']:.2f}")