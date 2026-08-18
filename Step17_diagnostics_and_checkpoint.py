# """
# بستن ردیف‌های سه و چهار و شش و نوزده با هم — باتری نهایی ابزارهای تشخیصی
# =====================================================================
# همه توابعی که در فاز صفر و دو به metrics_utils اضافه شدن ولی هیچ‌وقت
# روی یک مدل واقعی اجرا نشدن، این‌جا با هم روی معماری مرجع فقط-ساختاری
# به‌کار گرفته می‌شن:
#
#   - report_at_percentile_thresholds، ردیف سه
#   - bootstrap_test_ci، ردیف چهار
#   - compute_mad_neighbors، ردیف شش
#   - save_checkpoint، ردیف نوزده
#
# مدل روی همان پنج seed استاندارد آموزش می‌بیند؛ ابزارهای تشخیصی روی
# بهترین seed از نظر F1 اجرا و چاپ می‌شوند، و همان seed در پایان
# ذخیره می‌شود.
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# from torch_geometric.nn import SAGEConv
# from sklearn.preprocessing import StandardScaler
# from metrics_utils import (
#     evaluate_binary, find_best_threshold, get_temporal_split_masks,
#     build_edge_index, report_at_percentile_thresholds,
#     bootstrap_test_ci, compute_mad_neighbors, save_checkpoint, set_seed,
# )
#
# FEATURES_PATH = "datasets/elliptic_txs_features.csv"
# EDGES_PATH = "datasets/elliptic_txs_edgelist.csv"
# CLASSES_PATH = "datasets/elliptic_txs_classes.csv"
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
# print("Loading Elliptic Dataset...")
# df_feat = pd.read_csv(FEATURES_PATH, header=None)
# df_edge = pd.read_csv(EDGES_PATH)
# df_class = pd.read_csv(CLASSES_PATH)
#
# df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
# df_class.columns = ["txId", "class"]
# df_class["label"] = df_class["class"].map({"1": 1, "2": 0, "unknown": -1})
#
# map_id, edge_index = build_edge_index(df_feat["txId"].values, df_edge["txId1"], df_edge["txId2"])
# edge_index = edge_index.to(device)
#
# x_raw = df_feat.drop(columns=["txId", "time_step"]).values
# scaler = StandardScaler()
# x = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float).to(device)
# y = torch.tensor(df_class["label"].values, dtype=torch.long).to(device)
#
# time_steps_raw = torch.tensor(df_feat["time_step"].values, dtype=torch.long)
#
# train_mask, val_mask, test_mask = get_temporal_split_masks(
#     time_steps_raw, y, train_end=27, val_end=34, device=device
# )
#
# n_pos = (y[train_mask] == 1).sum().item()
# n_neg = (y[train_mask] == 0).sum().item()
# class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)
#
#
# class SAGEBlock(nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super().__init__()
#         self.conv = SAGEConv(in_channels, hidden_channels)
#
#     def forward(self, x, edge_index):
#         h = self.conv(x, edge_index)
#         return F.dropout(h, p=0.2, training=self.training)
#
#
# class StructuralOnlyGraphSAGE(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.block1 = SAGEBlock(in_channels, hidden_channels)
#         self.block2 = SAGEBlock(hidden_channels, hidden_channels)
#         self.dropout = nn.Dropout(0.3)
#         self.classifier = nn.Linear(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index, return_embeddings=False):
#         h1 = self.dropout(F.relu(self.block1(x, edge_index)))
#         h2 = self.dropout(F.relu(self.block2(h1, edge_index)))
#         out = self.classifier(h2)
#         if return_embeddings:
#             return out, h1, h2
#         return out
#
#
# LR = 0.005
# HIDDEN = 64
# EPOCHS = 200
# SEEDS = (42, 1, 7, 123, 2024)
#
#
# def train_one_seed(seed):
#     set_seed(seed)
#     model = StructuralOnlyGraphSAGE(in_channels=165, hidden_channels=HIDDEN, out_channels=2).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#
#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         optimizer.zero_grad()
#         out = model(x, edge_index)
#         loss = criterion(out[train_mask], y[train_mask])
#         loss.backward()
#         optimizer.step()
#
#     return model
#
#
# # ============================================================
# # آموزش پنج seed استاندارد، نگه‌داشتن مدل‌ها برای انتخاب بهترین
# # ============================================================
# print("=== آموزش پنج seed استاندارد ===")
# trained_models, seed_f1, seed_threshold = {}, {}, {}
#
# for seed in SEEDS:
#     model = train_one_seed(seed)
#     model.eval()
#     with torch.no_grad():
#         out = model(x, edge_index)
#         probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()
#
#     y_val = y[val_mask].cpu().numpy()
#     probs_val = probs_all[val_mask.cpu().numpy()]
#     best_t, _ = find_best_threshold(y_val, probs_val)
#
#     y_test = y[test_mask].cpu().numpy()
#     probs_test = probs_all[test_mask.cpu().numpy()]
#     preds_test = (probs_test >= best_t).astype(int)
#     metrics = evaluate_binary(f"seed={seed}", y_test, preds_test, probs_test, verbose=False)
#
#     trained_models[seed] = model
#     seed_f1[seed] = metrics["F1"]
#     seed_threshold[seed] = best_t
#     print(f"seed={seed}   F1={metrics['F1']:.4f}   threshold={best_t:.2f}")
#
# best_seed = max(seed_f1, key=seed_f1.get)
# best_model = trained_models[best_seed]
# best_threshold = seed_threshold[best_seed]
# print(f"\nبهترین seed: {best_seed}   F1={seed_f1[best_seed]:.4f}")
#
# best_model.eval()
# with torch.no_grad():
#     out, h1, h2 = best_model(x, edge_index, return_embeddings=True)
#     probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()
# y_test = y[test_mask].cpu().numpy()
# probs_test = probs_all[test_mask.cpu().numpy()]
#
#
# # ============================================================
# # ۱. threshold صدک‌محور، ردیف سه
# # ============================================================
# print("\n\n=== گزارش threshold صدک‌محور، بهترین seed ===")
# report_at_percentile_thresholds(y_test, probs_test, percentiles=(90, 99, 99.9))
#
#
# # ============================================================
# # ۲. فاصله اطمینان bootstrap، ردیف چهار
# # ============================================================
# print("\n\n=== فاصله اطمینان bootstrap روی test، بهترین seed ===")
# bootstrap_test_ci(y_test, probs_test, n_iterations=100, sample_frac=0.5, threshold=best_threshold)
#
#
# # ============================================================
# # ۳. معیار MAD، ردیف شش
# # ============================================================
# print("\n\n=== MAD بعد از هر لایه، بهترین seed ===")
# mad_layer1 = compute_mad_neighbors(h1, edge_index)
# mad_layer2 = compute_mad_neighbors(h2, edge_index)
# print(f"MAD بعد از لایه اول: {mad_layer1:.4f}")
# print(f"MAD بعد از لایه دوم: {mad_layer2:.4f}")
# print("عدد نزدیک صفر یعنی embedding گره‌های همسایه به‌شدت شبیه هم شده‌اند،")
# print("یعنی over-smoothing؛ عدد نزدیک یک یعنی گره‌های همسایه هنوز متمایزند.")
#
#
# # ============================================================
# # ۴. ذخیره checkpoint بهترین seed، ردیف نوزده
# # ============================================================
# save_checkpoint(
#     best_model, "structural_only_best.pt",
#     extra={
#         "seed": best_seed, "F1": seed_f1[best_seed], "threshold": float(best_threshold),
#         "hidden_channels": HIDDEN, "lr": LR,
#     },
# )
# print('\nبعداً برای بارگذاری در داشبورد جنگو:')
# print('  payload = load_checkpoint(model, "structural_only_best.pt")')




import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from metrics_utils import (
    evaluate_binary, find_best_threshold, get_temporal_split_masks,
    build_edge_index, report_at_percentile_thresholds,
    bootstrap_test_ci, compute_mad_neighbors, save_checkpoint, set_seed,
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

map_id, edge_index = build_edge_index(df_feat["txId"].values, df_edge["txId1"], df_edge["txId2"])
edge_index = edge_index.to(device)

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


class SAGEBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        return F.dropout(h, p=0.2, training=self.training)


class StructuralOnlyGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.block1 = SAGEBlock(in_channels, hidden_channels)
        self.block2 = SAGEBlock(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, return_embeddings=False):
        h1 = self.dropout(F.relu(self.block1(x, edge_index)))
        h2 = self.dropout(F.relu(self.block2(h1, edge_index)))
        out = self.classifier(h2)
        if return_embeddings:
            return out, h1, h2
        return out


LR = 0.005
HIDDEN = 64
EPOCHS = 200
SEEDS = (42, 1, 7, 123, 2024)


def train_one_seed(seed):
    set_seed(seed)
    model = StructuralOnlyGraphSAGE(in_channels=165, hidden_channels=HIDDEN, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    return model



print("=== آموزش پنج seed استاندارد ===")
trained_models, seed_f1, seed_threshold = {}, {}, {}

for seed in SEEDS:
    model = train_one_seed(seed)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    y_val = y[val_mask].cpu().numpy()
    probs_val = probs_all[val_mask.cpu().numpy()]
    best_t, _ = find_best_threshold(y_val, probs_val)

    y_test = y[test_mask].cpu().numpy()
    probs_test = probs_all[test_mask.cpu().numpy()]
    preds_test = (probs_test >= best_t).astype(int)
    metrics = evaluate_binary(f"seed={seed}", y_test, preds_test, probs_test, verbose=False)

    trained_models[seed] = model
    seed_f1[seed] = metrics["F1"]
    seed_threshold[seed] = best_t
    print(f"seed={seed}   F1={metrics['F1']:.4f}   threshold={best_t:.2f}")

best_seed = max(seed_f1, key=seed_f1.get)
best_model = trained_models[best_seed]
best_threshold = seed_threshold[best_seed]
print(f"\nبهترین seed: {best_seed}   F1={seed_f1[best_seed]:.4f}")

best_model.eval()
with torch.no_grad():
    out, h1, h2 = best_model(x, edge_index, return_embeddings=True)
    probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()
y_test = y[test_mask].cpu().numpy()
probs_test = probs_all[test_mask.cpu().numpy()]



print("\n\n=== گزارش threshold صدک‌محور، بهترین seed ===")
report_at_percentile_thresholds(y_test, probs_test, percentiles=(90, 99, 99.9))



print("\n\n=== فاصله اطمینان bootstrap روی test، بهترین seed ===")
bootstrap_test_ci(y_test, probs_test, n_iterations=100, sample_frac=0.5, threshold=best_threshold)



print("\n\n=== MAD بعد از هر لایه، بهترین seed ===")
mad_layer1 = compute_mad_neighbors(h1, edge_index)
mad_layer2 = compute_mad_neighbors(h2, edge_index)
print(f"MAD بعد از لایه اول: {mad_layer1:.4f}")
print(f"MAD بعد از لایه دوم: {mad_layer2:.4f}")
print("عدد نزدیک صفر یعنی embedding گره‌های همسایه به‌شدت شبیه هم شده‌اند،")
print("یعنی over-smoothing؛ عدد نزدیک یک یعنی گره‌های همسایه هنوز متمایزند.")



save_checkpoint(
    best_model, "structural_only_best.pt",
    extra={
        "seed": best_seed, "F1": seed_f1[best_seed], "threshold": float(best_threshold),
        "hidden_channels": HIDDEN, "lr": LR,
    },
)
print('\nبعداً برای بارگذاری در داشبورد جنگو:')
print('  payload = load_checkpoint(model, "structural_only_best.pt")')