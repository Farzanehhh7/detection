"""
فاز سه — threshold صدک‌محور روی SAML-D، برای دور زدن رانش نرخ پایه val/test
=====================================================================
چون نسبت illicit در val=0.21% و در test=0.59% تقریباً سه برابر فرق
داره، threshold بهینه‌شده روی F1 در val ممکنه برای test کالیبره نباشه.
این اسکریپت به‌جای تکیه به همون threshold، گزارش صدک‌محور می‌ده تا
ببینیم توانایی رتبه‌بندی واقعی مدل، مستقل از این رانش، چقدره.
"""

import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from metrics_utils import report_at_percentile_thresholds, bootstrap_test_ci, set_seed

DATA_PATH = "samld_processed_v3.pt"
S1, S2 = 25, 10
BATCH_SIZE = 256
WEIGHT_CAP = 30.0
EPOCHS = 20
SEED = 42

set_seed(SEED)
random.seed(SEED)

print("در حال بارگذاری داده پردازش‌شده SAML-D...")
data_dict = torch.load(DATA_PATH, weights_only=False)
x = data_dict["x"]
edge_index = data_dict["edge_index"]
y = data_dict["y_binary"]
train_mask = data_dict["train_mask"]
test_mask = data_dict["test_mask"]

n_pos = (y[train_mask] == 1).sum().item()
n_neg = (y[train_mask] == 0).sum().item()
class_weight = torch.tensor([1.0, min(n_neg / max(n_pos, 1), WEIGHT_CAP)])

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StructuralOnlyGraphSAGE(in_channels=x.shape[1], hidden_channels=64, out_channels=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))

train_node_ids = train_mask.nonzero(as_tuple=True)[0].tolist()

print("در حال آموزش...")
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

print("آموزش تمام شد.\n")

model.eval()
with torch.no_grad():
    out = model(x.to(device), edge_index.to(device))
    probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

y_test = y[test_mask].numpy()
probs_test = probs_all[test_mask.numpy()]

print("=== گزارش threshold صدک‌محور روی test ===")
report_at_percentile_thresholds(y_test, probs_test, percentiles=(90, 95, 99))

print("\n=== فاصله اطمینان bootstrap برای PR-AUC-محور، مستقل از threshold ===")
from sklearn.metrics import average_precision_score
print(f"PR-AUC واقعی این seed: {average_precision_score(y_test, probs_test):.4f}")