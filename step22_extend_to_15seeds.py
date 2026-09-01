"""
step22_extend_to_15seeds.py

بازبینی سوال «آیا گراف SAML-D به تشخیص نوع پول‌شویی کمک می‌کنه؟» با ۱۵
seed به‌جای ۵ -- دقیقاً همون کاری که برای سوال مشابه روی Elliptic
(Step32) انجام شد، وقتی ۵ seed نتیجه‌ی مبهم داد (p=0.5576) و ۱۵ seed
قطعی‌اش کرد.

این اسکریپت فقط ۱۰ seed جدید رو train می‌کنه (۲، ۳، ۴، ۵، ۶، ۸، ۹، ۱۰،
۱۱، ۱۲ -- عین همون seedهای اضافه‌ی Step32) و checkpoint هرکدوم رو
جداگانه ذخیره می‌کنه. به پنج checkpoint موجود (samld_seed_42/1/7/123/2024.pt)
دست نمی‌زنه.

بعد از این اسکریپت، باید step25/step26/step28 رو با نسخه‌ی ۱۵-seed‌شون
(SEEDS شامل هر ۱۵ تا) دوباره اجرا کنی تا خود macro-F1ها رو با ۱۵ عدد
به‌دست بیاری، بعد step30 رو با اون اعداد تازه به‌روزرسانی می‌کنیم --
دقیقاً همون ترتیبی که برای Elliptic طی شد.
"""

import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from metrics_utils import evaluate_binary, find_best_threshold, run_multi_seed, set_seed, save_checkpoint

DATA_PATH = "samld_processed_v3.pt"
S1, S2 = 25, 10
BATCH_SIZE = 256
WEIGHT_CAP = 30.0
EPOCHS = 20
NEW_SEEDS = (2, 3, 4, 5, 6, 8, 9, 10, 11, 12)  # ده seed جدید، عین Step32 برای Elliptic
CHECKPOINT_PATTERN = "samld_seed_{seed}.pt"

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
class_weight = torch.tensor([1.0, min(raw_weight, WEIGHT_CAP)])
print(f"Train: {train_mask.sum().item()}   Val: {val_mask.sum().item()}   Test: {test_mask.sum().item()}")
print(f"illicit در test: {y[test_mask].sum().item()}   وزن اعمال‌شده: {class_weight[1].item():.2f}")

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

    def forward(self, x, edge_index, return_embeddings=False):
        h1 = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        h2 = F.dropout(F.relu(self.conv2(h1, edge_index)), p=0.3, training=self.training)
        out = self.classifier(h2)
        if return_embeddings:
            return out, h1, h2
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
train_node_ids_master = train_mask.nonzero(as_tuple=True)[0].tolist()


def run_one_seed(seed):
    set_seed(seed)
    random.seed(seed)
    train_node_ids = list(train_node_ids_master)

    model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

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

    metrics = evaluate_binary("SAML-D GraphSAGE", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t

    save_checkpoint(
        model, CHECKPOINT_PATTERN.format(seed=seed),
        extra={
            "seed": seed,
            "F1": metrics["F1"],
            "threshold": float(best_t),
            "hidden_channels": 64,
            "lr": 0.005,
            "in_channels": x.shape[1],
        },
    )

    return metrics


print(f"\nده seed جدید train می‌شن: {NEW_SEEDS} -- پنج seed موجود دست‌نخورده می‌مونن.")
print("روی این مقیاس ممکنه ۳۰-۶۰ دقیقه طول بکشه، صبور باش.")
df, summary = run_multi_seed(run_one_seed, seeds=NEW_SEEDS, name="SAML-D GraphSAGE، ده seed جدید")

print(f"\nده چک‌پوینت جدید ذخیره شدن: {[CHECKPOINT_PATTERN.format(seed=s) for s in NEW_SEEDS]}")
print("حالا هر ۱۵ چک‌پوینت (۵ قدیمی + ۱۰ جدید) رو داری.")
print("قدم بعدی: step25/step26/step28 نسخه‌ی ۱۵-seed رو اجرا کن.")
