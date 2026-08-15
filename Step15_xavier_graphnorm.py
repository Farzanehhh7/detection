"""
بستن ردیف یک — تست Xavier initialization و GraphNorm
=====================================================================
طبق توصیه دقیق Dang و همکاران ۲۰۲۶: بهترین پیکربندی GraphSAGE در
مقاله‌شون Xavier initialization بدون GraphNorm بوده، و GraphNorm رو
مخصوص معماری‌های مبتنی بر GAT پیشنهاد دادن. این‌جا هر دو گزینه را
جداگانه و با هم، روی همان معماری فقط-ساختاری مرجع فاز یک، تست
می‌کنیم تا ببینیم آیا نتیجه منفی فاز یک با initialization بهتر عوض
می‌شود یا نه.

پیکربندی پیش‌فرض هم دوباره، داخل همین اسکریپت و همین کلاس، اجرا
می‌شود، نه فقط رجوع به عدد قدیمی؛ درسی که از تفاوت جزئی F1 در آزمایش
نشت یال گرفتیم این بود که حتی تغییرات کوچک در تعریف کلاس مدل، ترتیب
مصرف اعداد تصادفی مقداردهی اولیه را جابه‌جا می‌کند، پس مقایسه منصفانه
فقط داخل یک اسکریپت واحد معتبر است.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SAGEConv, GraphNorm
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


# ============================================================
# معماری: دقیقاً همان فقط-ساختاری قبلی، این‌بار با گزینه GraphNorm
# ============================================================
class SAGEBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, use_graphnorm=False):
        super().__init__()
        self.conv = SAGEConv(in_channels, hidden_channels)
        self.use_graphnorm = use_graphnorm
        if use_graphnorm:
            self.norm = GraphNorm(hidden_channels)

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        if self.use_graphnorm:
            h = self.norm(h)
        return F.dropout(h, p=0.2, training=self.training)


class StructuralOnlyGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, use_graphnorm=False):
        super().__init__()
        self.block1 = SAGEBlock(in_channels, hidden_channels, use_graphnorm)
        self.block2 = SAGEBlock(hidden_channels, hidden_channels, use_graphnorm)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        h = self.dropout(F.relu(self.block1(x, edge_index)))
        h = self.dropout(F.relu(self.block2(h, edge_index)))
        return self.classifier(h)


def apply_xavier(model):
    """SAGEConv در torch_geometric از دو لایه خطی داخلی، lin_l و lin_r، ساخته شده؛ هر دو باید صریح مقداردهی بشن."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, SAGEConv):
            if hasattr(module, "lin_l") and module.lin_l is not None:
                nn.init.xavier_uniform_(module.lin_l.weight)
                if module.lin_l.bias is not None:
                    nn.init.zeros_(module.lin_l.bias)
            if hasattr(module, "lin_r") and module.lin_r is not None:
                nn.init.xavier_uniform_(module.lin_r.weight)
    return model


EPOCHS = 200


def run_one_seed(seed, use_xavier, use_graphnorm):
    model = StructuralOnlyGraphSAGE(
        in_channels=165, hidden_channels=64, out_channels=2, use_graphnorm=use_graphnorm
    ).to(device)
    if use_xavier:
        model = apply_xavier(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
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

    metrics = evaluate_binary("SAGE", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


configs = {
    "پیش‌فرض، مرجع تازه": dict(use_xavier=False, use_graphnorm=False),
    "فقط Xavier": dict(use_xavier=True, use_graphnorm=False),
    "فقط GraphNorm": dict(use_xavier=False, use_graphnorm=True),
    "Xavier + GraphNorm": dict(use_xavier=True, use_graphnorm=True),
}

all_summaries = {}
for name, kwargs in configs.items():
    df, summary = run_multi_seed(
        lambda seed, kwargs=kwargs: run_one_seed(seed, **kwargs),
        seeds=(42, 1, 7, 123, 2024), name=name,
    )
    all_summaries[name] = summary

print("\n\n=== جدول نهایی، تست Xavier و GraphNorm ===")
for name, summary in all_summaries.items():
    print(f"{name:20s} F1 = {summary.loc['mean', 'F1']:.4f} ± {summary.loc['std', 'F1']:.4f}")
print(f"{'مرجع اصلی فاز یک':20s} F1 = 0.4427 ± 0.0323   (برای مقایسه با پیکربندی پیش‌فرض تازه بالا)")