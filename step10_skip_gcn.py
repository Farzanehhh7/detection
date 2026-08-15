"""
فاز دو، قدم اول — Skip-GCN، baseline مرجع مقاله Weber
=====================================================================
این معماری یک یال skip مستقیم از فیچر ورودی به خروجی داره. اگه
وزن‌های لایه‌های GCN صفر بشن، مدل دقیقاً معادل یک Logistic Regression
روی فیچرهای خام می‌شه. یعنی Skip-GCN تئوریاً تضمین می‌کنه از یک
رگرسیون خطی ساده بدتر نشه.

برای سازگاری با فاز صفر و یک، همون تقسیم train تا timestep بیست‌وهفت،
validation بیست‌وهشت تا سی‌وچهار، و test بالای سی‌وچهار استفاده شده.
Logistic Regression هم دوباره روی دقیقاً همین تقسیم فیت شده، نه با
عدد قدیمی جدول baseline_results.csv که روی تقسیم متفاوتی حساب شده
بود، تا مقایسه واقعاً apples-to-apples باشه.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
x_scaled = scaler.fit_transform(x_raw)
x = torch.tensor(x_scaled, dtype=torch.float).to(device)
y = torch.tensor(df_class["label"].values, dtype=torch.long).to(device)

time_steps_raw = torch.tensor(df_feat["time_step"].values, dtype=torch.long)

train_mask, val_mask, test_mask = get_temporal_split_masks(
    time_steps_raw, y, train_end=27, val_end=34, device=device
)
print(f"Train: {train_mask.sum().item()}   Val: {val_mask.sum().item()}   Test: {test_mask.sum().item()}")

n_pos = (y[train_mask] == 1).sum().item()
n_neg = (y[train_mask] == 0).sum().item()
class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)


# ============================================================
# صفر، Logistic Regression روی همین دقیقاً تقسیم
# ============================================================
train_idx = train_mask.cpu().numpy()
val_idx = val_mask.cpu().numpy()
test_idx = test_mask.cpu().numpy()
y_np = y.cpu().numpy()

lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(x_scaled[train_idx], y_np[train_idx])

lr_probs_val = lr.predict_proba(x_scaled[val_idx])[:, 1]
lr_best_t, _ = find_best_threshold(y_np[val_idx], lr_probs_val)

lr_probs_test = lr.predict_proba(x_scaled[test_idx])[:, 1]
lr_preds_test = (lr_probs_test >= lr_best_t).astype(int)
lr_metrics = evaluate_binary(
    "Logistic Regression", y_np[test_idx], lr_preds_test, lr_probs_test, verbose=False
)
print(f"\nLogistic Regression روی همین تقسیم:  F1 = {lr_metrics['F1']:.4f}   threshold = {lr_best_t:.2f}")


# ============================================================
# یک، معماری Skip-GCN
# ============================================================
class SkipGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.skip = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        return h + self.skip(x)


EPOCHS = 200


def run_one_seed(seed):
    model = SkipGCN(in_channels=165, hidden_channels=128, out_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

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

    metrics = evaluate_binary("Skip-GCN", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


df, summary = run_multi_seed(run_one_seed, seeds=(42, 1, 7, 123, 2024), name="Skip-GCN")

print("\n\n=== مقایسه نهایی فاز دو، قدم اول ===")
print(f"{'Logistic Regression':32s} F1 = {lr_metrics['F1']:.4f}")
print(f"{'Skip-GCN':32s} F1 = {summary.loc['mean', 'F1']:.4f} ± {summary.loc['std', 'F1']:.4f}")
print(f"{'GraphSAGE فقط ساختاری، از فاز یک':32s} F1 = 0.4427 ± 0.0323")