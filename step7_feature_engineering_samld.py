# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import pandas as pd
# # from torch_geometric.nn import GATConv
# # from sklearn.preprocessing import StandardScaler
# # from metrics_utils import evaluate_binary, set_seed
# #
# # set_seed(42)
# #
# # FEATURES_PATH = "datasets/elliptic_txs_features.csv"
# # EDGES_PATH = "datasets/elliptic_txs_edgelist.csv"
# # CLASSES_PATH = "datasets/elliptic_txs_classes.csv"
# #
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Using device: {device}")
# #
# # print("Loading Elliptic Dataset...")
# # df_feat = pd.read_csv(FEATURES_PATH, header=None)
# # df_edge = pd.read_csv(EDGES_PATH)
# # df_class = pd.read_csv(CLASSES_PATH)
# #
# # df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
# # df_class.columns = ["txId", "class"]
# # df_class["label"] = df_class["class"].map({"1": 1, "2": 0, "unknown": -1})
# #
# # map_id = {j: i for i, j in enumerate(df_feat["txId"].values)}
# # edge_index = torch.tensor([
# #     [map_id[src] for src in df_edge["txId1"]],
# #     [map_id[dst] for dst in df_edge["txId2"]],
# # ], dtype=torch.long).to(device)
# #
# # x_raw = df_feat.drop(columns=["txId", "time_step"]).values
# # scaler = StandardScaler()
# # x = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float).to(device)
# # y = torch.tensor(df_class["label"].values, dtype=torch.long).to(device)
# #
# # time_steps = torch.tensor(df_feat["time_step"].values - 1, dtype=torch.long).to(device)
# # NUM_TIMESTEPS = int(time_steps.max().item()) + 1
# # print(f"Number of timesteps: {NUM_TIMESTEPS}")
# #
# # train_mask = (y != -1) & (torch.tensor(df_feat["time_step"].values).to(device) <= 34)
# # test_mask = (y != -1) & (torch.tensor(df_feat["time_step"].values).to(device) > 34)
# #
# # n_pos = (y[train_mask] == 1).sum().item()
# # n_neg = (y[train_mask] == 0).sum().item()
# # class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)
# # print(f"Class Weight (illicit): {n_neg / n_pos:.2f}")
# #
# #
# # def compute_global_context(x, time_steps, num_timesteps):
# #     feat_dim = x.size(1)
# #     ctx_sum = torch.zeros(num_timesteps, feat_dim, device=x.device)
# #     counts = torch.zeros(num_timesteps, device=x.device)
# #     ctx_sum.index_add_(0, time_steps, x)
# #     counts.index_add_(0, time_steps, torch.ones_like(time_steps, dtype=torch.float))
# #     counts = counts.clamp(min=1).unsqueeze(1)
# #     return ctx_sum / counts
# #
# #
# # class TemporalEmbedding(nn.Module):
# #     def __init__(self, num_timesteps, embed_dim):
# #         super().__init__()
# #         self.embedding = nn.Embedding(num_timesteps, embed_dim)
# #
# #     def forward(self, time_steps):
# #         return self.embedding(time_steps)
# #
# #
# # class TripleAttentionLayer(nn.Module):
# #     def __init__(self, in_channels, hidden_channels, temporal_dim, global_in_channels, heads=4):
# #         super().__init__()
# #         self.struct_conv = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.2)
# #         self.temporal_proj = nn.Sequential(nn.Linear(temporal_dim, hidden_channels), nn.ReLU())
# #         self.global_proj = nn.Sequential(nn.Linear(global_in_channels, hidden_channels), nn.ReLU())
# #         self.attn_gate = nn.Linear(hidden_channels * 3, 3)
# #
# #     def forward(self, x, edge_index, temporal_emb, global_ctx_per_node):
# #         h_struct = self.struct_conv(x, edge_index)
# #         h_temporal = self.temporal_proj(temporal_emb)
# #         h_global = self.global_proj(global_ctx_per_node)
# #
# #         combined = torch.cat([h_struct, h_temporal, h_global], dim=1)
# #         weights = F.softmax(self.attn_gate(combined), dim=1)
# #         ws, wt, wg = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
# #
# #         out = ws * h_struct + wt * h_temporal + wg * h_global
# #         return out, weights
# #
# #
# # class ATGATGraphSAGE(nn.Module):
# #     def __init__(self, in_channels, hidden_channels, out_channels,
# #                  num_timesteps, temporal_dim=32, heads=4):
# #         super().__init__()
# #         self.num_timesteps = num_timesteps
# #         self.temporal_embed = TemporalEmbedding(num_timesteps, temporal_dim)
# #
# #         self.layer1 = TripleAttentionLayer(in_channels, hidden_channels, temporal_dim, in_channels, heads)
# #         self.layer2 = TripleAttentionLayer(hidden_channels, hidden_channels, temporal_dim, hidden_channels, heads)
# #
# #         self.dropout = nn.Dropout(0.3)
# #         self.classifier = nn.Linear(hidden_channels, out_channels)
# #
# #     def forward(self, x, edge_index, time_steps):
# #         t_emb = self.temporal_embed(time_steps)
# #
# #         global_ctx1 = compute_global_context(x, time_steps, self.num_timesteps)[time_steps]
# #         h, w1 = self.layer1(x, edge_index, t_emb, global_ctx1)
# #         h = self.dropout(F.relu(h))
# #
# #         global_ctx2 = compute_global_context(h, time_steps, self.num_timesteps)[time_steps]
# #         h, w2 = self.layer2(h, edge_index, t_emb, global_ctx2)
# #         h = self.dropout(F.relu(h))
# #
# #         out = self.classifier(h)
# #         return out, (w1, w2)
# #
# #
# # model = ATGATGraphSAGE(
# #     in_channels=165, hidden_channels=64, out_channels=2,
# #     num_timesteps=NUM_TIMESTEPS, temporal_dim=32, heads=4,
# # ).to(device)
# #
# # optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
# # criterion = nn.CrossEntropyLoss(weight=class_weights)
# #
# # print("\nStarting Triple Attention Training...")
# # for epoch in range(1, 201):
# #     model.train()
# #     optimizer.zero_grad()
# #     out, _ = model(x, edge_index, time_steps)
# #     loss = criterion(out[train_mask], y[train_mask])
# #     loss.backward()
# #     optimizer.step()
# #
# #     if epoch % 20 == 0:
# #         print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
# #
# #
# # model.eval()
# # with torch.no_grad():
# #     out, (w1, w2) = model(x, edge_index, time_steps)
# #
# # print("\nMean weights (ws=structural, wt=temporal, wg=global) - Layer 1:",
# #       w1.mean(dim=0).cpu().numpy().round(3))
# # print("Mean weights (ws=structural, wt=temporal, wg=global) - Layer 2:",
# #       w2.mean(dim=0).cpu().numpy().round(3))
# #
# # probs = F.softmax(out, dim=1)[test_mask, 1].cpu().numpy()
# # preds = out[test_mask].argmax(dim=1).cpu().numpy()
# # y_true = y[test_mask].cpu().numpy()
# #
# # evaluate_binary("ATGAT-GraphSAGE Triple Attention (Final)", y_true, preds, probs)
# # print("\nFor comparison:")
# # print("  Random Forest (Without Graph)   F1 = 0.806")
# # print("  Hybrid SAGE+XGBoost (step5)     F1 = 0.794")
# # print("  Simple GraphSAGE (step3)        F1 = 0.649")
#
#
#
#
# """
# فاز ۲ - قدم ۹ (نسخه دوم): معماری ATGAT-GraphSAGE با Triple Attention
# =================================================================
# دو اصلاح نسبت به نسخه اول، بر اساس چیزی که از وزن‌های ws/wt/wg
# نسخه قبلی یاد گرفتیم (لایه دوم تقریبا فقط رو ws=0.98 قفل شده بود):
#
#   ۱. جریان ساختاری از GATConv به SAGEConv تغییر کرد - چون هم اسم
#      رسمی معماری پروژه‌ت GraphSAGE‌ست نه GAT، هم خودمون تو step3
#      در برابر step4 با چشم دیدیم GraphSAGE (F1=0.649) بهتر از GAT
#      (F1=0.465) عمل می‌کنه. وقتی جریان ساختاری تقریبا کل مدل رو
#      قبضه کرد، داشت ضعف GAT رو با خودش می‌کشید.
#
#   ۲. یه جریمه آنتروپی به loss اضافه شد - این مانع می‌شه گیت (ws,
#      wt, wg) خیلی زود روی یه جریان قفل بشه و بقیه رو نادیده بگیره،
#      و به جریان‌های زمانی و کلی فرصت واقعی برای اثبات خودشون می‌ده.
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# from torch_geometric.nn import SAGEConv
# from sklearn.preprocessing import StandardScaler
# from metrics_utils import evaluate_binary, set_seed
#
# set_seed(42)
#
# # ============================================================
# # ۱. مسیر فایل‌ها و بارگذاری داده
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
# time_steps = torch.tensor(df_feat["time_step"].values - 1, dtype=torch.long).to(device)
# NUM_TIMESTEPS = int(time_steps.max().item()) + 1
# print(f"تعداد بازه‌های زمانی: {NUM_TIMESTEPS}")
#
# train_mask = (y != -1) & (torch.tensor(df_feat["time_step"].values).to(device) <= 34)
# test_mask = (y != -1) & (torch.tensor(df_feat["time_step"].values).to(device) > 34)
#
# n_pos = (y[train_mask] == 1).sum().item()
# n_neg = (y[train_mask] == 0).sum().item()
# class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)
# print(f"Class Weight (illicit): {n_neg / n_pos:.2f}")
#
#
# # ============================================================
# # ۲. اجزای معماری
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
#     def __init__(self, in_channels, hidden_channels, temporal_dim, global_in_channels):
#         super().__init__()
#         self.struct_conv = SAGEConv(in_channels, hidden_channels)
#         self.temporal_proj = nn.Sequential(nn.Linear(temporal_dim, hidden_channels), nn.ReLU())
#         self.global_proj = nn.Sequential(nn.Linear(global_in_channels, hidden_channels), nn.ReLU())
#         self.attn_gate = nn.Linear(hidden_channels * 3, 3)
#
#     def forward(self, x, edge_index, temporal_emb, global_ctx_per_node):
#         h_struct = self.struct_conv(x, edge_index)
#         h_struct = F.dropout(h_struct, p=0.2, training=self.training)
#         h_temporal = self.temporal_proj(temporal_emb)
#         h_global = self.global_proj(global_ctx_per_node)
#
#         combined = torch.cat([h_struct, h_temporal, h_global], dim=1)
#         weights = F.softmax(self.attn_gate(combined), dim=1)  # [N, 3] -> ws, wt, wg
#         ws, wt, wg = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
#
#         out = ws * h_struct + wt * h_temporal + wg * h_global
#         return out, weights
#
#
# class ATGATGraphSAGE(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels,
#                  num_timesteps, temporal_dim=32):
#         super().__init__()
#         self.num_timesteps = num_timesteps
#         self.temporal_embed = TemporalEmbedding(num_timesteps, temporal_dim)
#
#         self.layer1 = TripleAttentionLayer(in_channels, hidden_channels, temporal_dim, in_channels)
#         self.layer2 = TripleAttentionLayer(hidden_channels, hidden_channels, temporal_dim, hidden_channels)
#
#         self.dropout = nn.Dropout(0.3)
#         self.classifier = nn.Linear(hidden_channels, out_channels)
#
#     def forward(self, x, edge_index, time_steps):
#         t_emb = self.temporal_embed(time_steps)
#
#         global_ctx1 = compute_global_context(x, time_steps, self.num_timesteps)[time_steps]
#         h, w1 = self.layer1(x, edge_index, t_emb, global_ctx1)
#         h = self.dropout(F.relu(h))
#
#         global_ctx2 = compute_global_context(h, time_steps, self.num_timesteps)[time_steps]
#         h, w2 = self.layer2(h, edge_index, t_emb, global_ctx2)
#         h = self.dropout(F.relu(h))
#
#         out = self.classifier(h)
#         return out, (w1, w2)
#
#
# def gate_entropy(weights):
#     """آنتروپی توزیع وزن‌ها - هرچه بیشتر، توزیع متعادل‌تره (کمتر قفل‌شده رو یه جریان)"""
#     return -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
#
#
# # ============================================================
# # ۳. ساخت مدل و اموزش
# # ============================================================
# model = ATGATGraphSAGE(
#     in_channels=165, hidden_channels=64, out_channels=2,
#     num_timesteps=NUM_TIMESTEPS, temporal_dim=32,
# ).to(device)
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
# criterion = nn.CrossEntropyLoss(weight=class_weights)
#
# # هرچقدر این عدد بزرگ‌تر باشه، فشار بیشتری برای متعادل نگه‌داشتن
# # سه جریانه. اگه بعد از اجرا دیدی هنوز خیلی قفل می‌کنه، این رو
# # مثلا به 0.1 افزایش بده؛ اگه خیلی مانع یادگیری شد، به 0.02 کم کن.
# ENTROPY_LAMBDA = 0.05
#
# print("\nStarting Triple Attention Training (v2 - SAGEConv + Entropy Regularization)...")
# for epoch in range(1, 201):
#     model.train()
#     optimizer.zero_grad()
#     out, (w1, w2) = model(x, edge_index, time_steps)
#
#     ce_loss = criterion(out[train_mask], y[train_mask])
#     entropy_bonus = gate_entropy(w1) + gate_entropy(w2)
#     loss = ce_loss - ENTROPY_LAMBDA * entropy_bonus
#
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 20 == 0:
#         print(f"Epoch {epoch:03d} | CE Loss: {ce_loss.item():.4f} | Gate Entropy: {entropy_bonus.item():.4f}")
#
# # ============================================================
# # ۴. ارزیابی نهایی
# # ============================================================
# model.eval()
# with torch.no_grad():
#     out, (w1, w2) = model(x, edge_index, time_steps)
#
# print("\nمیانگین وزن‌ها (ws=ساختاری, wt=زمانی, wg=کلی) - لایه اول :",
#       w1.mean(dim=0).cpu().numpy().round(3))
# print("میانگین وزن‌ها (ws=ساختاری, wt=زمانی, wg=کلی) - لایه دوم  :",
#       w2.mean(dim=0).cpu().numpy().round(3))
#
# probs = F.softmax(out, dim=1)[test_mask, 1].cpu().numpy()
# preds = out[test_mask].argmax(dim=1).cpu().numpy()
# y_true = y[test_mask].cpu().numpy()
#
# evaluate_binary("ATGAT-GraphSAGE Triple Attention v2 (Final)", y_true, preds, probs)
# print("\nبرای مقایسه:")
# print("  Random Forest (بدون گراف)     F1 = 0.806")
# print("  هیبرید SAGE+XGBoost (step5)   F1 = 0.794")
# print("  GraphSAGE ساده (step3)        F1 = 0.649")
# print("  Triple Attention v1 (GAT)     F1 = 0.562")


"""
فاز ۲ - قدم ۹ (نسخه دوم): معماری ATGAT-GraphSAGE با Triple Attention
=================================================================
دو اصلاح نسبت به نسخه اول، بر اساس چیزی که از وزن‌های ws/wt/wg
نسخه قبلی یاد گرفتیم (لایه دوم تقریبا فقط رو ws=0.98 قفل شده بود):

  ۱. جریان ساختاری از GATConv به SAGEConv تغییر کرد - چون هم اسم
     رسمی معماری پروژه‌ت GraphSAGE‌ست نه GAT، هم خودمون تو step3
     در برابر step4 با چشم دیدیم GraphSAGE (F1=0.649) بهتر از GAT
     (F1=0.465) عمل می‌کنه. وقتی جریان ساختاری تقریبا کل مدل رو
     قبضه کرد، داشت ضعف GAT رو با خودش می‌کشید.

  ۲. یه جریمه آنتروپی به loss اضافه شد - این مانع می‌شه گیت (ws,
     wt, wg) خیلی زود روی یه جریان قفل بشه و بقیه رو نادیده بگیره،
     و به جریان‌های زمانی و کلی فرصت واقعی برای اثبات خودشون می‌ده.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from metrics_utils import evaluate_binary, set_seed

set_seed(42)

# ============================================================
# ۱. مسیر فایل‌ها و بارگذاری داده
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

time_steps = torch.tensor(df_feat["time_step"].values - 1, dtype=torch.long).to(device)
NUM_TIMESTEPS = int(time_steps.max().item()) + 1
print(f"تعداد بازه‌های زمانی: {NUM_TIMESTEPS}")

train_mask = (y != -1) & (torch.tensor(df_feat["time_step"].values).to(device) <= 34)
test_mask = (y != -1) & (torch.tensor(df_feat["time_step"].values).to(device) > 34)

n_pos = (y[train_mask] == 1).sum().item()
n_neg = (y[train_mask] == 0).sum().item()
class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)
print(f"Class Weight (illicit): {n_neg / n_pos:.2f}")


# ============================================================
# ۲. اجزای معماری
# ============================================================
def compute_global_context(x, time_steps, num_timesteps):
    feat_dim = x.size(1)
    ctx_sum = torch.zeros(num_timesteps, feat_dim, device=x.device)
    counts = torch.zeros(num_timesteps, device=x.device)
    ctx_sum.index_add_(0, time_steps, x)
    counts.index_add_(0, time_steps, torch.ones_like(time_steps, dtype=torch.float))
    counts = counts.clamp(min=1).unsqueeze(1)
    return ctx_sum / counts


class TemporalEmbedding(nn.Module):
    def __init__(self, num_timesteps, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_timesteps, embed_dim)

    def forward(self, time_steps):
        return self.embedding(time_steps)


class TripleAttentionLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, temporal_dim, global_in_channels):
        super().__init__()
        self.struct_conv = SAGEConv(in_channels, hidden_channels)
        self.temporal_proj = nn.Sequential(nn.Linear(temporal_dim, hidden_channels), nn.ReLU())
        self.global_proj = nn.Sequential(nn.Linear(global_in_channels, hidden_channels), nn.ReLU())
        self.attn_gate = nn.Linear(hidden_channels * 3, 3)

    def forward(self, x, edge_index, temporal_emb, global_ctx_per_node):
        h_struct = self.struct_conv(x, edge_index)
        h_struct = F.dropout(h_struct, p=0.2, training=self.training)
        h_temporal = self.temporal_proj(temporal_emb)
        h_global = self.global_proj(global_ctx_per_node)

        combined = torch.cat([h_struct, h_temporal, h_global], dim=1)
        weights = F.softmax(self.attn_gate(combined), dim=1)  # [N, 3] -> ws, wt, wg
        ws, wt, wg = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        out = ws * h_struct + wt * h_temporal + wg * h_global
        return out, weights


class ATGATGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_timesteps, temporal_dim=32):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.temporal_embed = TemporalEmbedding(num_timesteps, temporal_dim)

        self.layer1 = TripleAttentionLayer(in_channels, hidden_channels, temporal_dim, in_channels)
        self.layer2 = TripleAttentionLayer(hidden_channels, hidden_channels, temporal_dim, hidden_channels)

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, time_steps):
        t_emb = self.temporal_embed(time_steps)

        global_ctx1 = compute_global_context(x, time_steps, self.num_timesteps)[time_steps]
        h, w1 = self.layer1(x, edge_index, t_emb, global_ctx1)
        h = self.dropout(F.relu(h))

        global_ctx2 = compute_global_context(h, time_steps, self.num_timesteps)[time_steps]
        h, w2 = self.layer2(h, edge_index, t_emb, global_ctx2)
        h = self.dropout(F.relu(h))

        out = self.classifier(h)
        return out, (w1, w2)


def gate_entropy(weights):
    """آنتروپی توزیع وزن‌ها - هرچه بیشتر، توزیع متعادل‌تره (کمتر قفل‌شده رو یه جریان)"""
    return -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()


# ============================================================
# ۳. ساخت مدل و اموزش
# ============================================================
model = ATGATGraphSAGE(
    in_channels=165, hidden_channels=64, out_channels=2,
    num_timesteps=NUM_TIMESTEPS, temporal_dim=32,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# هرچقدر این عدد بزرگ‌تر باشه، فشار بیشتری برای متعادل نگه‌داشتن
# سه جریانه. اگه بعد از اجرا دیدی هنوز خیلی قفل می‌کنه، این رو
# مثلا به 0.1 افزایش بده؛ اگه خیلی مانع یادگیری شد، به 0.02 کم کن.
ENTROPY_LAMBDA = 0.05

print("\nStarting Triple Attention Training (v2 - SAGEConv + Entropy Regularization)...")
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out, (w1, w2) = model(x, edge_index, time_steps)

    ce_loss = criterion(out[train_mask], y[train_mask])
    entropy_bonus = gate_entropy(w1) + gate_entropy(w2)
    loss = ce_loss - ENTROPY_LAMBDA * entropy_bonus

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | CE Loss: {ce_loss.item():.4f} | Gate Entropy: {entropy_bonus.item():.4f}")

# ============================================================
# ۴. ارزیابی نهایی
# ============================================================
model.eval()
with torch.no_grad():
    out, (w1, w2) = model(x, edge_index, time_steps)

print("\nمیانگین وزن‌ها (ws=ساختاری, wt=زمانی, wg=کلی) - لایه اول :",
      w1.mean(dim=0).cpu().numpy().round(3))
print("میانگین وزن‌ها (ws=ساختاری, wt=زمانی, wg=کلی) - لایه دوم  :",
      w2.mean(dim=0).cpu().numpy().round(3))

probs = F.softmax(out, dim=1)[test_mask, 1].cpu().numpy()
preds = out[test_mask].argmax(dim=1).cpu().numpy()
y_true = y[test_mask].cpu().numpy()

evaluate_binary("ATGAT-GraphSAGE Triple Attention v2 (Final)", y_true, preds, probs)
print("\nبرای مقایسه:")
print("  Random Forest (بدون گراف)     F1 = 0.806")
print("  هیبرید SAGE+XGBoost (step5)   F1 = 0.794")
print("  GraphSAGE ساده (step3)        F1 = 0.649")
print("  Triple Attention v1 (GAT)     F1 = 0.562")