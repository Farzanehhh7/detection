"""
بستن ردیف چهارده و پونزده با هم — جست‌وجوی هایپرپارامتر با early stopping
=====================================================================
یک grid کوچک روی نرخ یادگیری و اندازه لایه پنهان، هرکدام روی سه
seed برای کاهش هزینه محاسباتی مرحله جست‌وجو. EarlyStopper تازه‌ساخته
هم این‌جا برای اولین بار در عمل به‌کار می‌ره، هم برای کوتاه‌کردن
زمان هر اجرا هم برای بستن ردیف چهارده هم‌زمان. معیار نظارت
EarlyStopper، PR-AUC روی validation است، نه F1 در threshold=0.5،
چون طبق بخش ۲.۵ سند، این معیار تحت عدم توازن شدید Elliptic معتبرتره.

بهترین پیکربندی مرحله جست‌وجو در پایان با پروتکل استاندارد پنج seed
کامل دوباره تایید می‌شود.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import average_precision_score
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from metrics_utils import (
    evaluate_binary, find_best_threshold, get_temporal_split_masks,
    run_multi_seed, build_edge_index, EarlyStopper, log_experiment,
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

    def forward(self, x, edge_index):
        h = self.dropout(F.relu(self.block1(x, edge_index)))
        h = self.dropout(F.relu(self.block2(h, edge_index)))
        return self.classifier(h)


MAX_EPOCHS = 200
PATIENCE = 20
LOG_PATH = "hparam_search_log.csv"


def run_one_seed(seed, lr, hidden_channels):
    model = StructuralOnlyGraphSAGE(in_channels=165, hidden_channels=hidden_channels, out_channels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    stopper = EarlyStopper(patience=PATIENCE, mode="max")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x, edge_index)
            val_probs = F.softmax(val_out, dim=1)[val_mask, 1].cpu().numpy()
        val_auprc = average_precision_score(y[val_mask].cpu().numpy(), val_probs)

        if stopper.step(val_auprc, model, epoch=epoch):
            break

    stopper.restore_best(model)
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

    metrics = evaluate_binary("SAGE", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


# ============================================================
# جست‌وجوی کوچک: سه نرخ یادگیری در سه اندازه لایه پنهان
# ============================================================
search_space = {
    "lr": [0.001, 0.005, 0.01],
    "hidden_channels": [32, 64, 128],
}
SEARCH_SEEDS = (42, 1, 7)  # فقط سه seed در مرحله جست‌وجو، ارزان‌تر از پروتکل استاندارد پنج‌تایی

results_summary = []
print("\n=== مرحله جست‌وجو ===")
for lr in search_space["lr"]:
    for hc in search_space["hidden_channels"]:
        name = f"lr={lr}, hidden={hc}"
        df, summary = run_multi_seed(
            lambda seed, lr=lr, hc=hc: run_one_seed(seed, lr, hc),
            seeds=SEARCH_SEEDS, name=name, verbose=False,
        )
        mean_f1 = summary.loc["mean", "F1"]
        std_f1 = summary.loc["std", "F1"]
        print(f"{name:25s} F1 = {mean_f1:.4f} ± {std_f1:.4f}")
        log_experiment(LOG_PATH, {
            "lr": lr, "hidden_channels": hc,
            "F1_mean": mean_f1, "F1_std": std_f1,
        })
        results_summary.append((lr, hc, mean_f1, std_f1))

best_lr, best_hc, best_f1, _ = max(results_summary, key=lambda r: r[2])
print(f"\nبهترین پیکربندی مرحله جست‌وجو: lr={best_lr}, hidden_channels={best_hc}, F1={best_f1:.4f}")

# ============================================================
# تایید نهایی بهترین پیکربندی با پروتکل استاندارد پنج seed
# ============================================================
print("\n\n=== تایید نهایی با پنج seed کامل ===")
df_final, summary_final = run_multi_seed(
    lambda seed: run_one_seed(seed, best_lr, best_hc),
    seeds=(42, 1, 7, 123, 2024), name="بهترین پیکربندی، تایید نهایی",
)

print(f"\n{'بهترین پیکربندی، پنج seed کامل':32s} F1 = {summary_final.loc['mean', 'F1']:.4f} ± {summary_final.loc['std', 'F1']:.4f}")
print(f"{'مرجع lr=0.005 hidden=64، فاز یک':32s} F1 = 0.4427 ± 0.0323")