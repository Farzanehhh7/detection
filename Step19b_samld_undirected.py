"""
فاز سه، قدم دوم، نسخه بدون وابستگی — نمونه‌برداری همسایگی با PyTorch خالص
=====================================================================
چون pyg-lib و torch-sparse روی این محیط، پایتون ۳.۱۴ و torch ۲.۱۲.۱،
درست کار نکردن، این نسخه کاملاً بدون این دو پکیج نوشته شده. منطقش
همون K=2 و S1=25 و S2=10 قبلیه، فقط پیاده‌سازیش دستیه: یک‌بار فهرست
همسایگی ورودی هر گره ساخته می‌شه، و برای هر batch یک زیرگراف محلی دو
مرحله‌ای ساخته و مدل روی همون زیرگراف کوچیک آموزش می‌بینه.

فقط یک seed اجرا می‌شه، هدف اعتبارسنجی pipeline است نه گزارش نهایی.
"""

import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from metrics_utils import evaluate_binary, find_best_threshold, set_seed

# DATA_PATH = "samld_processed.pt"
DATA_PATH = "samld_processed_v3.pt"
S1, S2 = 25, 10
BATCH_SIZE = 256  # کوچیک‌تر از نسخه NeighborLoader چون ساخت هر batch اینجا با حلقه پایتونه

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

print(f"تعداد گره: {x.shape[0]}   تعداد یال: {edge_index.shape[1]}   تعداد فیچر: {x.shape[1]}")
print(f"Train: {train_mask.sum().item()}   Val: {val_mask.sum().item()}   Test: {test_mask.sum().item()}")

n_pos = (y[train_mask] == 1).sum().item()
n_neg = (y[train_mask] == 0).sum().item()
class_weight = torch.tensor([1.0, n_neg / max(n_pos, 1)])
print(f"Class weight (illicit): {n_neg / max(n_pos, 1):.2f}")


# ============================================================
# ۱. فهرست همسایگی ورودی، یک‌بار ساخته می‌شه، نه هر batch
# ============================================================
print("در حال ساخت فهرست همسایگی...")
adjacency = defaultdict(list)
src_list = edge_index[0].tolist()
dst_list = edge_index[1].tolist()
for s, d in zip(src_list, dst_list):
    adjacency[d].append(s)
    adjacency[s].append(d)  # نسخه undirected: هر تراکنش دو طرفه برای نمونه‌برداری همسایگی حساب می‌شه
print(f"فهرست همسایگی برای {len(adjacency)} گره با حداقل یک همسایه ورودی ساخته شد.")


def sample_k(neighbors, k):
    if len(neighbors) <= k:
        return neighbors
    return random.sample(neighbors, k)


def build_batch_subgraph(seed_nodes):
    """
    برای فهرستی از گره‌های seed، یک زیرگراف محلی دو مرحله‌ای می‌سازه:
    اول S1 همسایه مستقیم هر seed، بعد S2 همسایه هرکدوم از اون‌ها.
    خروجی: فیچر و edge_index محلی، به‌علاوه اندیس محلی خود seed ها.
    """
    hop1_of = {}
    all_hop1 = set()
    for s in seed_nodes:
        neighs = sample_k(adjacency.get(s, []), S1)
        hop1_of[s] = neighs
        all_hop1.update(neighs)

    hop2_of = {}
    all_hop2 = set()
    for n in all_hop1:
        neighs = sample_k(adjacency.get(n, []), S2)
        hop2_of[n] = neighs
        all_hop2.update(neighs)

    all_nodes = list(set(seed_nodes) | all_hop1 | all_hop2)
    local_id = {n: i for i, n in enumerate(all_nodes)}

    edges_src, edges_dst = [], []
    for s in seed_nodes:
        for n in hop1_of[s]:
            edges_src.append(local_id[n])
            edges_dst.append(local_id[s])
    for n in all_hop1:
        for n2 in hop2_of[n]:
            edges_src.append(local_id[n2])
            edges_dst.append(local_id[n])

    node_idx_tensor = torch.tensor(all_nodes, dtype=torch.long)
    sub_x = x[node_idx_tensor]
    if len(edges_src) == 0:
        sub_edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        sub_edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

train_node_ids = train_mask.nonzero(as_tuple=True)[0].tolist()
EPOCHS = 20

print("\nStarting training with pure-PyTorch neighbor sampling...")
for epoch in range(1, EPOCHS + 1):
    random.shuffle(train_node_ids)
    model.train()
    total_loss, n_batches = 0.0, 0

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

        total_loss += loss.item()
        n_batches += 1

    if epoch % 2 == 0:
        print(f"Epoch {epoch:03d} | میانگین Loss: {total_loss / n_batches:.4f} | تعداد batch: {n_batches}")

print("\nآموزش تمام شد. ارزیابی نهایی با inference کامل روی کل گراف...")

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

evaluate_binary("SAML-D GraphSAGE، undirected، تک seed", y_test, preds_test, probs_test)
print(f"\nthreshold انتخابی روی validation: {best_t:.2f}")