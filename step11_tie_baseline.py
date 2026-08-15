"""
فاز دو، قدم دوم — TIE baseline، بدون هیچ گراف
=====================================================================
همون embedding یادگرفتنی هر بازه زمانی که در Triple Attention به‌عنوان
جریان زمانی استفاده شد، این‌بار به‌تنهایی و بدون SAGEConv یا GCNConv.
مدل فقط یک MLP روی فیچرهای خام به‌علاوه این embedding است، هیچ
edge_index ای اصلاً وارد فوروارد نمی‌شه.

هدف: بفهمیم چقدر از عملکرد مدل‌های گرافی واقعاً از ساختار گراف میاد
و چقدرش صرفاً از دونستن این‌که تراکنش در کدام بازه زمانی رخ داده.
اگه این baseline خیلی نزدیک GraphSAGE عمل کنه، یعنی ارزش افزوده
واقعی گراف کمتر از چیزیه که فکر می‌کردیم.

همون تقسیم فاز صفر و یک: train تا timestep بیست‌وهفت، validation
بیست‌وهشت تا سی‌وچهار، test بالای سی‌وچهار.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from metrics_utils import (
    evaluate_binary, find_best_threshold, get_temporal_split_masks,
    run_multi_seed,
)

FEATURES_PATH = "datasets/elliptic_txs_features.csv"
CLASSES_PATH = "datasets/elliptic_txs_classes.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading Elliptic Dataset...")
df_feat = pd.read_csv(FEATURES_PATH, header=None)
df_class = pd.read_csv(CLASSES_PATH)

df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
df_class.columns = ["txId", "class"]
df_class["label"] = df_class["class"].map({"1": 1, "2": 0, "unknown": -1})

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


class TIEBaseline(nn.Module):
    def __init__(self, in_channels, num_timesteps, temporal_dim, hidden_channels, out_channels):
        super().__init__()
        self.temporal_embed = nn.Embedding(num_timesteps, temporal_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + temporal_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, time_steps):
        t_emb = self.temporal_embed(time_steps)
        combined = torch.cat([x, t_emb], dim=1)
        return self.mlp(combined)


EPOCHS = 200


def run_one_seed(seed):
    model = TIEBaseline(
        in_channels=165, num_timesteps=NUM_TIMESTEPS, temporal_dim=32,
        hidden_channels=64, out_channels=2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, time_steps)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(x, time_steps)
        probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    y_val = y[val_mask].cpu().numpy()
    probs_val = probs_all[val_mask.cpu().numpy()]
    best_t, _ = find_best_threshold(y_val, probs_val)

    y_test = y[test_mask].cpu().numpy()
    probs_test = probs_all[test_mask.cpu().numpy()]
    preds_test = (probs_test >= best_t).astype(int)

    metrics = evaluate_binary("TIE Baseline", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


df, summary = run_multi_seed(run_one_seed, seeds=(42, 1, 7, 123, 2024), name="TIE Baseline")

print("\n\n=== مقایسه نهایی فاز دو، قدم دوم ===")
print(f"{'TIE Baseline، بدون گراف':32s} F1 = {summary.loc['mean', 'F1']:.4f} ± {summary.loc['std', 'F1']:.4f}")
print(f"{'GraphSAGE فقط ساختاری، از فاز یک':32s} F1 = 0.4427 ± 0.0323")