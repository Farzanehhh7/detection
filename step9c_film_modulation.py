"""
دور سوم و احتمالاً آخر — تعدیل به‌جای رقابت
=====================================================================
جریان کلی همون نسخه انحرافی قبلیه که جواب داد. جریان زمانی این بار
دیگه یک جریان مستقل رقیب در softmax gate نیست؛ به‌شکل FiLM جریان
ساختاری رو تعدیل می‌کنه: gamma و beta از embedding بازه زمانی
ساخته می‌شن و می‌گن "ساختار رو تو این بازه چطور بخون"، نه اینکه
خودشون مستقیم ادعای کلاس‌بندی داشته باشن.

معیار موفقیت ساده است: اگه F1 این نسخه از 0.4427 یعنی فقط-ساختاری
بهتر نشد، جمع‌بندی این‌که سه جریان به شکل فعلی ارزش اضافه نمی‌کنن
قطعی می‌شه و می‌ریم فاز دو.
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

train_mask, val_mask, test_mask = get_temporal_split_masks(
    time_steps_raw, y, train_end=27, val_end=34, device=device
)
print(f"Train: {train_mask.sum().item()}   Val: {val_mask.sum().item()}   Test: {test_mask.sum().item()}")

n_pos = (y[train_mask] == 1).sum().item()
n_neg = (y[train_mask] == 0).sum().item()
class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)


def compute_global_deviation(x, time_steps, num_timesteps):
    feat_dim = x.size(1)
    ctx_sum = torch.zeros(num_timesteps, feat_dim, device=x.device)
    counts = torch.zeros(num_timesteps, device=x.device)
    ctx_sum.index_add_(0, time_steps, x)
    counts.index_add_(0, time_steps, torch.ones_like(time_steps, dtype=torch.float))
    counts = counts.clamp(min=1).unsqueeze(1)
    timestep_mean = ctx_sum / counts
    return x - timestep_mean[time_steps]


class TemporalEmbedding(nn.Module):
    def __init__(self, num_timesteps, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_timesteps, embed_dim)

    def forward(self, time_steps):
        return self.embedding(time_steps)


class ModulatedLayer(nn.Module):
    """
    ساختاری از GraphSAGE می‌آد، بعد با FiLM از embedding زمانی
    تعدیل می‌شه، بعد انحراف کلی مستقیم بهش اضافه می‌شه. دیگه هیچ
    softmax gate ای بین جریان‌ها رقابت نمی‌ندازه.
    """

    def __init__(self, in_channels, hidden_channels, temporal_dim, global_in_channels):
        super().__init__()
        self.struct_conv = SAGEConv(in_channels, hidden_channels)
        self.temporal_gamma = nn.Linear(temporal_dim, hidden_channels)
        self.temporal_beta = nn.Linear(temporal_dim, hidden_channels)
        self.global_proj = nn.Sequential(nn.Linear(global_in_channels, hidden_channels), nn.ReLU())

    def forward(self, x, edge_index, temporal_emb, global_dev):
        h_struct = F.dropout(self.struct_conv(x, edge_index), p=0.2, training=self.training)
        gamma = torch.tanh(self.temporal_gamma(temporal_emb))
        beta = self.temporal_beta(temporal_emb)
        h_modulated = h_struct * (1.0 + gamma) + beta
        h_global = self.global_proj(global_dev)
        return h_modulated + h_global


class ATGATModulated(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_timesteps, temporal_dim=32):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.temporal_embed = TemporalEmbedding(num_timesteps, temporal_dim)
        self.layer1 = ModulatedLayer(in_channels, hidden_channels, temporal_dim, in_channels)
        self.layer2 = ModulatedLayer(hidden_channels, hidden_channels, temporal_dim, hidden_channels)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, time_steps):
        t_emb = self.temporal_embed(time_steps)

        gd1 = compute_global_deviation(x, time_steps, self.num_timesteps)
        h = self.layer1(x, edge_index, t_emb, gd1)
        h = self.dropout(F.relu(h))

        gd2 = compute_global_deviation(h, time_steps, self.num_timesteps)
        h = self.layer2(h, edge_index, t_emb, gd2)
        h = self.dropout(F.relu(h))

        return self.classifier(h)


EPOCHS = 200


def run_one_seed(seed):
    model = ATGATModulated(
        in_channels=165, hidden_channels=64, out_channels=2,
        num_timesteps=NUM_TIMESTEPS, temporal_dim=32,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index, time_steps)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, time_steps)
        probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    y_val = y[val_mask].cpu().numpy()
    probs_val = probs_all[val_mask.cpu().numpy()]
    best_t, _ = find_best_threshold(y_val, probs_val)

    y_test = y[test_mask].cpu().numpy()
    probs_test = probs_all[test_mask.cpu().numpy()]
    preds_test = (probs_test >= best_t).astype(int)

    metrics = evaluate_binary("FiLM Modulated", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


df, summary = run_multi_seed(run_one_seed, seeds=(42, 1, 7, 123, 2024), name="ساختاری + FiLM زمانی + انحراف کلی")

print("\n\n=== مقایسه نهایی ===")
print(f"{'فقط ساختاری، از دور قبل':32s} F1 = 0.4427 ± 0.0323")
print(f"{'FiLM زمانی + انحراف کلی':32s} F1 = {summary.loc['mean', 'F1']:.4f} ± {summary.loc['std', 'F1']:.4f}")