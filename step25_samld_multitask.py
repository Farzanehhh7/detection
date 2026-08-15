"""
فاز سه — multi-task واقعی روی SAML-D: سر دودویی به‌علاوه سر نوع ادغام‌شده
=====================================================================
یک ستون فقرات GraphSAGE مشترک، دو سر جدا: سر دودویی همیشگی illicit
در برابر licit، و یک سر جدید شانزده‌کلاسه برای نوع پول‌شویی، که در
اون نوع‌های ۱۱ و ۱۲، طبق نتیجه step23 روی v3، در یک کلاس «سایر»
ادغام شدن چون هرکدوم زیر پنج نمونه در test داشتن.

سر نوع فقط روی اکانت‌های illicit معنا داره؛ به‌جای فیلتر دستی، از
ignore_index=-1 در CrossEntropyLoss استفاده شده تا اکانت‌های licit،
که برچسب نوعشون -1 است، خودکار از این بخش loss کنار گذاشته بشن.

فقط یک seed، چون هدف این اجرا اعتبارسنجی خط لوله joint training است،
نه هنوز گزارش نهایی.
"""

import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import accuracy_score, classification_report
from metrics_utils import (
    evaluate_binary, find_best_threshold, report_at_percentile_thresholds, set_seed,
)

DATA_PATH = "samld_processed_v3.pt"
S1, S2 = 25, 10
BATCH_SIZE = 256
WEIGHT_CAP = 30.0
EPOCHS = 20
SEED = 42
TYPE_LOSS_WEIGHT = 0.5  # چون سر نوع کمکیه؛ نیاز به تنظیم دقیق‌تر بعداً

set_seed(SEED)
random.seed(SEED)

print("در حال بارگذاری داده پردازش‌شده SAML-D...")
data_dict = torch.load(DATA_PATH, weights_only=False)
x = data_dict["x"]
edge_index = data_dict["edge_index"]
y_binary = data_dict["y_binary"]
y_type_raw = data_dict["y_type"]
train_mask = data_dict["train_mask"]
val_mask = data_dict["val_mask"]
test_mask = data_dict["test_mask"]


# ============================================================
# ادغام نوع ۱۱ و ۱۲ به «سایر»، طبق تصمیم بعد از step23 روی v3
# ============================================================
def build_type_remap(num_old_types, merge_ids=(11, 12)):
    remap = {}
    for old in range(num_old_types):
        if old in merge_ids:
            remap[old] = min(merge_ids)  # هر دو به همون شماره کوچیک‌تر، ۱۱، می‌رن
        elif old < min(merge_ids):
            remap[old] = old
        else:
            remap[old] = old - (len(merge_ids) - 1)
    return remap


NUM_OLD_TYPES = data_dict["num_types"]
type_remap = build_type_remap(NUM_OLD_TYPES)
NUM_TYPE_CLASSES = len(set(type_remap.values()))
print(f"تعداد کلاس نوع بعد از ادغام: {NUM_TYPE_CLASSES}   (قبلاً {NUM_OLD_TYPES})")

y_type = y_type_raw.clone()
for old_id, new_id in type_remap.items():
    y_type[y_type_raw == old_id] = new_id
# -1 دست‌نخورده می‌مونه چون توی merge_ids یا range نیست

n_pos = (y_binary[train_mask] == 1).sum().item()
n_neg = (y_binary[train_mask] == 0).sum().item()
class_weight_binary = torch.tensor([1.0, min(n_neg / max(n_pos, 1), WEIGHT_CAP)])
print(f"Train: {train_mask.sum().item()}   Val: {val_mask.sum().item()}   Test: {test_mask.sum().item()}")
print(f"illicit در train: {n_pos}   وزن دودویی اعمال‌شده: {class_weight_binary[1].item():.2f}")

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


class MultiTaskGraphSAGE(nn.Module):
    """ستون فقرات مشترک، دو سر جدا: دودویی و نوع."""

    def __init__(self, in_channels, hidden_channels, num_type_classes):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.binary_head = nn.Linear(hidden_channels, 2)
        self.type_head = nn.Linear(hidden_channels, num_type_classes)

    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.3, training=self.training)
        h = F.dropout(F.relu(self.conv2(h, edge_index)), p=0.3, training=self.training)
        return self.binary_head(h), self.type_head(h)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = MultiTaskGraphSAGE(in_channels=x.shape[1], hidden_channels=64, num_type_classes=NUM_TYPE_CLASSES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
binary_criterion = nn.CrossEntropyLoss(weight=class_weight_binary.to(device))
type_criterion = nn.CrossEntropyLoss(ignore_index=-1)  # اکانت‌های licit خودکار نادیده گرفته می‌شن

train_node_ids = train_mask.nonzero(as_tuple=True)[0].tolist()

print("\nStarting joint training...")
for epoch in range(1, EPOCHS + 1):
    random.shuffle(train_node_ids)
    model.train()
    total_bin_loss, total_type_loss, n_batches = 0.0, 0.0, 0

    for i in range(0, len(train_node_ids), BATCH_SIZE):
        seed_nodes = train_node_ids[i:i + BATCH_SIZE]
        sub_x, sub_edge_index, seed_local_idx = build_batch_subgraph(seed_nodes)
        sub_x, sub_edge_index = sub_x.to(device), sub_edge_index.to(device)
        seed_local_idx = seed_local_idx.to(device)
        batch_y_binary = y_binary[seed_nodes].to(device)
        batch_y_type = y_type[seed_nodes].to(device)

        optimizer.zero_grad()
        out_binary, out_type = model(sub_x, sub_edge_index)
        loss_binary = binary_criterion(out_binary[seed_local_idx], batch_y_binary)
        loss_type = type_criterion(out_type[seed_local_idx], batch_y_type)
        loss = loss_binary + TYPE_LOSS_WEIGHT * loss_type
        loss.backward()
        optimizer.step()

        total_bin_loss += loss_binary.item()
        total_type_loss += loss_type.item()
        n_batches += 1

    if epoch % 2 == 0:
        print(f"Epoch {epoch:03d} | Binary Loss: {total_bin_loss / n_batches:.4f} | "
              f"Type Loss: {total_type_loss / n_batches:.4f}")

print("\nآموزش تمام شد. ارزیابی نهایی...")

model.eval()
with torch.no_grad():
    out_binary, out_type = model(x.to(device), edge_index.to(device))
    probs_binary = F.softmax(out_binary, dim=1)[:, 1].cpu().numpy()
    preds_type = out_type.argmax(dim=1).cpu().numpy()

# --- ارزیابی سر دودویی، با گزارش صدک‌محور چون F1 خام گمراه‌کننده بود ---
y_test_binary = y_binary[test_mask].numpy()
probs_test_binary = probs_binary[test_mask.numpy()]
print("\n=== سر دودویی، گزارش صدک‌محور روی test ===")
report_at_percentile_thresholds(y_test_binary, probs_test_binary, percentiles=(90, 95, 99))

# --- ارزیابی سر نوع، فقط روی اکانت‌های illicit واقعی در test ---
y_test_type = y_type[test_mask].numpy()
preds_test_type = preds_type[test_mask.numpy()]
has_type = y_test_type != -1

print(f"\n=== سر نوع، فقط روی {has_type.sum()} اکانت illicit با نوع مشخص در test ===")
acc = accuracy_score(y_test_type[has_type], preds_test_type[has_type])
print(f"دقت کلی: {acc:.4f}")
print(classification_report(y_test_type[has_type], preds_test_type[has_type], zero_division=0))