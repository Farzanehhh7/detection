# """
# فاز صفر و یک با هم — زیرساخت چند-seed به علاوه ablation سه‌گانه واقعی
# =====================================================================
# این اسکریپت دو کار همزمان انجام می‌ده:
#
#   ۱. هر پیکربندی رو روی ۵ seed اجرا می‌کنه و میانگین ± انحراف معیار
#      گزارش می‌ده، نه یک عدد تنها از یک اجرا.
#
#   ۲. چهار پیکربندی از Triple Attention رو با هم مقایسه می‌کنه: فقط
#      ساختاری، ساختاری+زمانی، ساختاری+کلی، و هر سه با هم — تا مشخص بشه
#      هر جریان چقدر واقعاً ارزش اضافه می‌کنه، نه فقط چقدر gate weight
#      می‌گیره.
#
# معماری پایه همون SAGEConv + جریمه آنتروپی نسخه دومی است که خودت نوشتی؛
# فقط جریان‌های زمانی و کلی این بار قابل خاموش‌کردن شدن، و threshold
# دیگه ثابت روی ۰.۵ نیست — روی validation انتخاب می‌شه.
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
#     run_multi_seed,
# )
#
# # ============================================================
# # ۱. مسیر فایل‌ها و بارگذاری داده — دست‌نخورده نسبت به نسخه قبلی
# # ============================================================
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
# map_id = {j: i for i, j in enumerate(df_feat["txId"].values)}
# edge_index = torch.tensor([
#     [map_id[src] for src in df_edge["txId1"]],
#     [map_id[dst] for dst in df_edge["txId2"]],
# ], dtype=torch.long).to(device)
#
# x_raw = df_feat.drop(columns=["txId", "time_step"]).values
# scaler = StandardScaler()
# x = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float).to(device)
# y = torch.tensor(df_class["label"].values, dtype=torch.long).to(device)
#
# time_steps_raw = torch.tensor(df_feat["time_step"].values, dtype=torch.long)
# time_steps = (time_steps_raw - 1).to(device)
# NUM_TIMESTEPS = int(time_steps.max().item()) + 1
#
# # ============================================================
# # ۲. تقسیم سه‌گانه — train_end=27 یعنی تکه‌ی قدیمی train به دو نیم
# #    می‌شه، test همون بالای ۳۴ قبلی و دست‌نخورده می‌مونه
# # ============================================================
# train_mask, val_mask, test_mask = get_temporal_split_masks(
#     time_steps_raw, y, train_end=27, val_end=34, device=device
# )
# print(f"Train: {train_mask.sum().item()}   Val: {val_mask.sum().item()}   Test: {test_mask.sum().item()}")
#
# n_pos = (y[train_mask] == 1).sum().item()
# n_neg = (y[train_mask] == 0).sum().item()
# class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)
# print(f"Class Weight (illicit): {n_neg / n_pos:.2f}")
#
#
# # ============================================================
# # ۳. اجزای معماری — تنها تغییر: use_temporal و use_global قابل خاموش‌کردن
# # ============================================================
# def compute_global_context(x, time_steps, num_timesteps):
#     feat_dim = x.size(1)
#     ctx_sum = torch.zeros(num_timesteps, feat_dim, device=x.device)
#     counts = torch.zeros(num_timesteps, device=x.device)
#     ctx_sum.index_add_(0, time_steps, x)
#     counts.index_add_(0, time_steps, torch.ones_like(time_steps, dtype=torch.float))
#     counts = counts.clamp(min=1).unsqueeze(1)
#     return ctx_sum / counts
#
#
# class TemporalEmbedding(nn.Module):
#     def __init__(self, num_timesteps, embed_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(num_timesteps, embed_dim)
#
#     def forward(self, time_steps):
#         return self.embedding(time_steps)
#
#
# class TripleAttentionLayer(nn.Module):
#     def __init__(self, in_channels, hidden_channels, temporal_dim, global_in_channels,
#                  use_temporal=True, use_global=True):
#         super().__init__()
#         self.use_temporal = use_temporal
#         self.use_global = use_global
#         self.struct_conv = SAGEConv(in_channels, hidden_channels)
#         n_streams = 1 + int(use_temporal) + int(use_global)
#         if use_temporal:
#             self.temporal_proj = nn.Sequential(nn.Linear(temporal_dim, hidden_channels), nn.ReLU())
#         if use_global:
#             self.global_proj = nn.Sequential(nn.Linear(global_in_channels, hidden_channels), nn.ReLU())
#         self.attn_gate = nn.Linear(hidden_channels * n_streams, n_streams)
#
#     def forward(self, x, edge_index, temporal_emb, global_ctx_per_node):
#         streams = [F.dropout(self.struct_conv(x, edge_index), p=0.2, training=self.training)]
#         if self.use_temporal:
#             streams.append(self.temporal_proj(temporal_emb))
#         if self.use_global:
#             streams.append(self.global_proj(global_ctx_per_node))
#
#         combined = torch.cat(streams, dim=1)
#         weights = F.softmax(self.attn_gate(combined), dim=1)
#         out = sum(weights[:, i:i + 1] * streams[i] for i in range(len(streams)))
#         return out, weights
#
#
# class ATGATGraphSAGE(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_timesteps,
#                  temporal_dim=32, use_temporal=True, use_global=True):
#         super().__init__()
#         self.num_timesteps = num_timesteps
#         self.use_temporal = use_temporal
#         self.use_global = use_global
#         self.temporal_embed = TemporalEmbedding(num_timesteps, temporal_dim)
#
#         self.layer1 = TripleAttentionLayer(in_channels, hidden_channels, temporal_dim,
#                                             in_channels, use_temporal, use_global)
#         self.layer2 = TripleAttentionLayer(hidden_channels, hidden_channels, temporal_dim,
#                                             hidden_channels, use_temporal, use_global)
#
#         self.dropout = nn.Dropout(0.3)
#         self.classifier = nn.Linear(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index, time_steps):
#         t_emb = self.temporal_embed(time_steps) if self.use_temporal else None
#
#         gc1 = compute_global_context(x, time_steps, self.num_timesteps)[time_steps] if self.use_global else None
#         h, w1 = self.layer1(x, edge_index, t_emb, gc1)
#         h = self.dropout(F.relu(h))
#
#         gc2 = compute_global_context(h, time_steps, self.num_timesteps)[time_steps] if self.use_global else None
#         h, w2 = self.layer2(h, edge_index, t_emb, gc2)
#         h = self.dropout(F.relu(h))
#
#         out = self.classifier(h)
#         return out, (w1, w2)
#
#
# def gate_entropy(weights):
#     return -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
#
#
# ENTROPY_LAMBDA = 0.05
# EPOCHS = 200
#
#
# # ============================================================
# # ۴. یک اجرای کامل برای یک seed و یک پیکربندی
# # ============================================================
# def run_one_seed(seed, use_temporal, use_global):
#     model = ATGATGraphSAGE(
#         in_channels=165, hidden_channels=64, out_channels=2,
#         num_timesteps=NUM_TIMESTEPS, temporal_dim=32,
#         use_temporal=use_temporal, use_global=use_global,
#     ).to(device)
#
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#
#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         optimizer.zero_grad()
#         out, (w1, w2) = model(x, edge_index, time_steps)
#         ce_loss = criterion(out[train_mask], y[train_mask])
#         if use_temporal or use_global:
#             entropy_bonus = gate_entropy(w1) + gate_entropy(w2)
#             loss = ce_loss - ENTROPY_LAMBDA * entropy_bonus
#         else:
#             loss = ce_loss
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     with torch.no_grad():
#         out, _ = model(x, edge_index, time_steps)
#         probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()
#
#     # threshold فقط روی validation، هیچ‌وقت روی test
#     y_val = y[val_mask].cpu().numpy()
#     probs_val = probs_all[val_mask.cpu().numpy()]
#     best_t, _ = find_best_threshold(y_val, probs_val)
#
#     y_test = y[test_mask].cpu().numpy()
#     probs_test = probs_all[test_mask.cpu().numpy()]
#     preds_test = (probs_test >= best_t).astype(int)
#
#     metrics = evaluate_binary("Triple Attention", y_test, preds_test, probs_test, verbose=False)
#     metrics["threshold"] = best_t
#     return metrics
#
#
# # ============================================================
# # ۵. اجرای چهار پیکربندی، هرکدام روی ۵ seed
# # ============================================================
# configs = {
#     "فقط ساختاری": dict(use_temporal=False, use_global=False),
#     "ساختاری + زمانی": dict(use_temporal=True, use_global=False),
#     "ساختاری + کلی": dict(use_temporal=False, use_global=True),
#     "هر سه جریان": dict(use_temporal=True, use_global=True),
# }
#
# all_summaries = {}
# for config_name, kwargs in configs.items():
#     df, summary = run_multi_seed(
#         lambda seed, kwargs=kwargs: run_one_seed(seed, **kwargs),
#         seeds=(42, 1, 7, 123, 2024),
#         name=config_name,
#     )
#     all_summaries[config_name] = summary
#
# print("\n\n=== جدول نهایی ablation — میانگین ± انحراف معیار F1 روی ۵ seed ===")
# for config_name, summary in all_summaries.items():
#     print(f"{config_name:20s} F1 = {summary.loc['mean', 'F1']:.4f} ± {summary.loc['std', 'F1']:.4f}")




"""
فاز صفر و یک با هم — زیرساخت چند-seed به علاوه ablation سه‌گانه واقعی
=====================================================================
این اسکریپت دو کار همزمان انجام می‌ده:

  ۱. هر پیکربندی رو روی ۵ seed اجرا می‌کنه و میانگین ± انحراف معیار
     گزارش می‌ده، نه یک عدد تنها از یک اجرا.

  ۲. چهار پیکربندی از Triple Attention رو با هم مقایسه می‌کنه: فقط
     ساختاری، ساختاری+زمانی، ساختاری+کلی، و هر سه با هم — تا مشخص بشه
     هر جریان چقدر واقعاً ارزش اضافه می‌کنه، نه فقط چقدر gate weight
     می‌گیره.

معماری پایه همون SAGEConv + جریمه آنتروپی نسخه دومی است که خودت نوشتی؛
فقط جریان‌های زمانی و کلی این بار قابل خاموش‌کردن شدن، و threshold
دیگه ثابت روی ۰.۵ نیست — روی validation انتخاب می‌شه.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from metrics_utils import (
    evaluate_binary, find_best_threshold, get_temporal_split_masks,
    run_multi_seed,
)

# ============================================================
# ۱. مسیر فایل‌ها و بارگذاری داده — دست‌نخورده نسبت به نسخه قبلی
# ============================================================
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

map_id = {j: i for i, j in enumerate(df_feat["txId"].values)}
edge_index = torch.tensor([
    [map_id[src] for src in df_edge["txId1"]],
    [map_id[dst] for dst in df_edge["txId2"]],
], dtype=torch.long).to(device)

x_raw = df_feat.drop(columns=["txId", "time_step"]).values
scaler = StandardScaler()
x = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float).to(device)
y = torch.tensor(df_class["label"].values, dtype=torch.long).to(device)

time_steps_raw = torch.tensor(df_feat["time_step"].values, dtype=torch.long)
time_steps = (time_steps_raw - 1).to(device)
NUM_TIMESTEPS = int(time_steps.max().item()) + 1

# ============================================================
# ۲. تقسیم سه‌گانه — train_end=27 یعنی تکه‌ی قدیمی train به دو نیم
#    می‌شه، test همون بالای ۳۴ قبلی و دست‌نخورده می‌مونه
# ============================================================
train_mask, val_mask, test_mask = get_temporal_split_masks(
    time_steps_raw, y, train_end=27, val_end=34, device=device
)
print(f"Train: {train_mask.sum().item()}   Val: {val_mask.sum().item()}   Test: {test_mask.sum().item()}")

n_pos = (y[train_mask] == 1).sum().item()
n_neg = (y[train_mask] == 0).sum().item()
class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)
print(f"Class Weight (illicit): {n_neg / n_pos:.2f}")


# ============================================================
# ۳. اجزای معماری — تنها تغییر: use_temporal و use_global قابل خاموش‌کردن
# ============================================================
def compute_global_context(x, time_steps, num_timesteps):
    """
    نسخه قبلی این تابع میانگین بازه زمانی رو خام برمی‌گردوند، یعنی
    همه گره‌های یک بازه دقیقاً یک بردار یکسان می‌گرفتن و هیچ اطلاعات
    تمایزدهنده‌ای بین گره‌ها نداشت. ablation ثابت کرد این طراحی
    عملکرد رو بدتر می‌کنه، نه بهتر.

    این‌جا به‌جای میانگین خام، انحراف هر گره از میانگین بازه زمانی‌اش
    رو برمی‌گردونیم. این بردار دیگه برای هر گره فرق می‌کنه و دقیقاً
    همون سیگنالی‌ست که معماری از اول قرار بود بگیره: چقدر این تراکنش
    از وضعیت معمول بازار در همون لحظه فاصله داره.
    """
    feat_dim = x.size(1)
    ctx_sum = torch.zeros(num_timesteps, feat_dim, device=x.device)
    counts = torch.zeros(num_timesteps, device=x.device)
    ctx_sum.index_add_(0, time_steps, x)
    counts.index_add_(0, time_steps, torch.ones_like(time_steps, dtype=torch.float))
    counts = counts.clamp(min=1).unsqueeze(1)
    timestep_mean = ctx_sum / counts
    deviation = x - timestep_mean[time_steps]
    return deviation


class TemporalEmbedding(nn.Module):
    def __init__(self, num_timesteps, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_timesteps, embed_dim)

    def forward(self, time_steps):
        return self.embedding(time_steps)


class TripleAttentionLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_dim, global_in_channels,
                 use_temporal=True, use_global=True):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_global = use_global
        self.struct_conv = SAGEConv(in_channels, hidden_channels)
        n_streams = 1 + int(use_temporal) + int(use_global)
        if use_temporal:
            self.temporal_proj = nn.Sequential(nn.Linear(temporal_dim, hidden_channels), nn.ReLU())
        if use_global:
            self.global_proj = nn.Sequential(nn.Linear(global_in_channels, hidden_channels), nn.ReLU())
        self.attn_gate = nn.Linear(hidden_channels * n_streams, n_streams)

    def forward(self, x, edge_index, temporal_emb, global_ctx_per_node):
        streams = [F.dropout(self.struct_conv(x, edge_index), p=0.2, training=self.training)]
        if self.use_temporal:
            streams.append(self.temporal_proj(temporal_emb))
        if self.use_global:
            streams.append(self.global_proj(global_ctx_per_node))

        combined = torch.cat(streams, dim=1)
        weights = F.softmax(self.attn_gate(combined), dim=1)
        out = sum(weights[:, i:i + 1] * streams[i] for i in range(len(streams)))
        return out, weights


class ATGATGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_timesteps,
                 temporal_dim=32, use_temporal=True, use_global=True):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.use_temporal = use_temporal
        self.use_global = use_global
        self.temporal_embed = TemporalEmbedding(num_timesteps, temporal_dim)

        self.layer1 = TripleAttentionLayer(in_channels, hidden_channels, temporal_dim,
                                            in_channels, use_temporal, use_global)
        self.layer2 = TripleAttentionLayer(hidden_channels, hidden_channels, temporal_dim,
                                            hidden_channels, use_temporal, use_global)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, time_steps):
        t_emb = self.temporal_embed(time_steps) if self.use_temporal else None

        gc1 = compute_global_context(x, time_steps, self.num_timesteps) if self.use_global else None
        h, w1 = self.layer1(x, edge_index, t_emb, gc1)
        h = self.dropout(F.relu(h))

        gc2 = compute_global_context(h, time_steps, self.num_timesteps) if self.use_global else None
        h, w2 = self.layer2(h, edge_index, t_emb, gc2)
        h = self.dropout(F.relu(h))

        out = self.classifier(h)
        return out, (w1, w2)


def gate_entropy(weights):
    return -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()


ENTROPY_LAMBDA = 0.05
EPOCHS = 200


# ============================================================
# ۴. یک اجرای کامل برای یک seed و یک پیکربندی
# ============================================================
def run_one_seed(seed, use_temporal, use_global):
    model = ATGATGraphSAGE(
        in_channels=165, hidden_channels=64, out_channels=2,
        num_timesteps=NUM_TIMESTEPS, temporal_dim=32,
        use_temporal=use_temporal, use_global=use_global,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out, (w1, w2) = model(x, edge_index, time_steps)
        ce_loss = criterion(out[train_mask], y[train_mask])
        if use_temporal or use_global:
            entropy_bonus = gate_entropy(w1) + gate_entropy(w2)
            loss = ce_loss - ENTROPY_LAMBDA * entropy_bonus
        else:
            loss = ce_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out, _ = model(x, edge_index, time_steps)
        probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    # threshold فقط روی validation، هیچ‌وقت روی test
    y_val = y[val_mask].cpu().numpy()
    probs_val = probs_all[val_mask.cpu().numpy()]
    best_t, _ = find_best_threshold(y_val, probs_val)

    y_test = y[test_mask].cpu().numpy()
    probs_test = probs_all[test_mask.cpu().numpy()]
    preds_test = (probs_test >= best_t).astype(int)

    metrics = evaluate_binary("Triple Attention", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


# ============================================================
# ۵. اجرای چهار پیکربندی، هرکدام روی ۵ seed
#
# نسخه دوم این اسکریپت — تنها تفاوت با اجرای قبلی، تعریف جدید جریان
# کلی است. "ساختاری + کلی" و "هر سه جریان" این بار انحراف هر گره از
# میانگین بازه زمانی‌اش رو می‌بینن، نه یک بردار یکسان برای همه.
# اگه این نسخه هنوز از "فقط ساختاری" جلو نزنه، یعنی مشکل عمیق‌تر از
# طراحی فیچره و باید سراغ جریان زمانی هم بریم.
# ============================================================
configs = {
    "فقط ساختاری": dict(use_temporal=False, use_global=False),
    "ساختاری + زمانی": dict(use_temporal=True, use_global=False),
    "ساختاری + کلی": dict(use_temporal=False, use_global=True),
    "هر سه جریان": dict(use_temporal=True, use_global=True),
}

all_summaries = {}
for config_name, kwargs in configs.items():
    df, summary = run_multi_seed(
        lambda seed, kwargs=kwargs: run_one_seed(seed, **kwargs),
        seeds=(42, 1, 7, 123, 2024),
        name=config_name,
    )
    all_summaries[config_name] = summary

print("\n\n=== جدول نهایی ablation — میانگین ± انحراف معیار F1 روی ۵ seed ===")
for config_name, summary in all_summaries.items():
    print(f"{config_name:20s} F1 = {summary.loc['mean', 'F1']:.4f} ± {summary.loc['std', 'F1']:.4f}")