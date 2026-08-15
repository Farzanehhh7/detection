"""
بستن ردیف نه از جدول محدودیت‌ها — نشت احتمالی یال train به test
=====================================================================
اول جهت زمانی یال‌ها تشخیص داده می‌شود: آیا src_ts <= dst_ts همیشه
برقرار است؟ اگر برقرار باشد، هیچ گره train نمی‌تواند از طریق
پیام‌رسانی از یک گره با timestep بالاتر تاثیر بگیرد، چون aggregation
در PyG از edge_index[0] به edge_index[1] جریان دارد، و نگرانی نشت
به‌طور کامل رفع می‌شود.

اگر یال رو به عقب وجود داشته باشد، یک نسخه masked از edge_index
ساخته می‌شود که فقط یال‌های رو به جلو در زمان را نگه می‌دارد، و
دقیقاً همان معماری فقط-ساختاری فاز یک، با همان پنج seed، هم روی
گراف کامل و هم روی گراف masked آموزش داده می‌شود تا اثر واقعی این
نشت احتمالی روی F1 سنجیده شود.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from metrics_utils import (
    evaluate_binary, find_best_threshold, get_temporal_split_masks,
    run_multi_seed, build_edge_index,
)

FEATURES_PATH = "datasets/elliptic_txs_features.csv"
EDGES_PATH = "datasets/elliptic_txs_edgelist.csv"
CLASSES_PATH = "datasets/elliptic_txs_classes.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading Elliptic Dataset...")
df_feat = pd.read_csv(FEATURES_PATH, header=None)
df_edge = pd.read_csv(EDGES_PATH)
df_class = pd.read_csv(CLASSES_PATH)

df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
df_class.columns = ["txId", "class"]
df_class["label"] = df_class["class"].map({"1": 1, "2": 0, "unknown": -1})

map_id, edge_index_cpu = build_edge_index(df_feat["txId"].values, df_edge["txId1"], df_edge["txId2"])
edge_index = edge_index_cpu.to(device)

x_raw = df_feat.drop(columns=["txId", "time_step"]).values
scaler = StandardScaler()
x = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float).to(device)
y = torch.tensor(df_class["label"].values, dtype=torch.long).to(device)

time_steps_raw = torch.tensor(df_feat["time_step"].values, dtype=torch.long)

train_mask, val_mask, test_mask = get_temporal_split_masks(
    time_steps_raw, y, train_end=27, val_end=34, device=device
)

n_pos = (y[train_mask] == 1).sum().item()
n_neg = (y[train_mask] == 0).sum().item()
class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)


# ============================================================
# ۱. تشخیص جهت زمانی یال‌ها
# ============================================================
src_ts = time_steps_raw[edge_index_cpu[0]]
dst_ts = time_steps_raw[edge_index_cpu[1]]

forward = (src_ts <= dst_ts).sum().item()
backward = (src_ts > dst_ts).sum().item()
total = len(src_ts)

print(f"\nتعداد کل یال: {total}")
print(f"یال رو به جلو در زمان، src_ts <= dst_ts: {forward}   ({forward/total*100:.2f}%)")
print(f"یال رو به عقب در زمان، src_ts > dst_ts:  {backward}   ({backward/total*100:.2f}%)")

TRAIN_END = 27
dst_in_train = dst_ts <= TRAIN_END
src_ahead_of_dst = src_ts > dst_ts
leaking_edges = (dst_in_train & src_ahead_of_dst).sum().item()
print(f"\nیال‌هایی که می‌توانند یک گره train را از گره‌ای با timestep بالاتر متاثر کنند: {leaking_edges}")

if leaking_edges == 0:
    print("نتیجه اولیه: هیچ یال نشت‌کننده‌ای پیدا نشد؛ گراف ذاتاً رو به جلو در زمان است.")
else:
    print("نتیجه اولیه: یال نشت‌کننده وجود دارد؛ آموزش روی نسخه masked هم انجام می‌شود.")

forward_edge_mask = (src_ts <= dst_ts).to(device)
edge_index_masked = edge_index[:, forward_edge_mask]
print(f"\nتعداد یال بعد از masking: {edge_index_masked.shape[1]}   "
      f"حذف‌شده: {edge_index.shape[1] - edge_index_masked.shape[1]}")


# ============================================================
# ۲. همان معماری فقط-ساختاری فاز یک، بدون تغییر
# ============================================================
class TripleAttentionLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.struct_conv = SAGEConv(in_channels, hidden_channels)
        self.attn_gate = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        h = F.dropout(self.struct_conv(x, edge_index), p=0.2, training=self.training)
        weight = F.softmax(self.attn_gate(h), dim=1)  # همیشه دقیقاً ۱ با یک جریان
        return weight * h


class StructuralOnlyGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.layer1 = TripleAttentionLayer(in_channels, hidden_channels)
        self.layer2 = TripleAttentionLayer(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.dropout(F.relu(self.layer1(x, edge_index)))
        h = self.dropout(F.relu(self.layer2(h, edge_index)))
        return self.classifier(h)


EPOCHS = 200


def run_one_seed(seed, ei):
    model = StructuralOnlyGraphSAGE(in_channels=165, hidden_channels=64, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, ei)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x, ei)
        probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    y_val = y[val_mask].cpu().numpy()
    probs_val = probs_all[val_mask.cpu().numpy()]
    best_t, _ = find_best_threshold(y_val, probs_val)

    y_test = y[test_mask].cpu().numpy()
    probs_test = probs_all[test_mask.cpu().numpy()]
    preds_test = (probs_test >= best_t).astype(int)

    metrics = evaluate_binary("StructuralOnly", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


print("\n\n=== گراف کامل، باید تقریباً با مرجع قبلی F1=0.4427 یکی باشد ===")
df_full, summary_full = run_multi_seed(
    lambda seed: run_one_seed(seed, edge_index),
    seeds=(42, 1, 7, 123, 2024), name="گراف کامل"
)

print("\n\n=== گراف masked، فقط یال‌های رو به جلو در زمان ===")
df_masked, summary_masked = run_multi_seed(
    lambda seed: run_one_seed(seed, edge_index_masked),
    seeds=(42, 1, 7, 123, 2024), name="گراف masked"
)

print("\n\n=== مقایسه نهایی، ردیف نه ===")
print(f"{'گراف کامل':20s} F1 = {summary_full.loc['mean', 'F1']:.4f} ± {summary_full.loc['std', 'F1']:.4f}")
print(f"{'گراف masked':20s} F1 = {summary_masked.loc['mean', 'F1']:.4f} ± {summary_masked.loc['std', 'F1']:.4f}")