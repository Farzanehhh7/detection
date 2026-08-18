# """
# Step32_elliptic_graph_vs_nograph_more_seeds.py
#
# Resolves the open statistical-power question from Phase 2: is GraphSAGE
# really indistinguishable from TIE (no graph at all) on Elliptic, or did
# 5 seeds just not have enough power to tell? (Step12 found p=0.5576 with
# 5 seeds.) This re-runs the exact same two architectures -- unchanged --
# with 15 seeds instead of 5: the original 5 (42, 1, 7, 123, 2024) plus 10
# new ones, so the original numbers remain a direct subset for comparison.
#
# Both model classes are copied EXACTLY from your own reviewed scripts,
# not reimplemented from scratch:
#   - StructuralOnlyGraphSAGE: identical to Step17's version (SAGEBlock,
#     no gate, hidden=64, dropout=0.2/0.3)
#   - TIEBaseline: identical to step11's version (temporal embedding + MLP,
#     no edge_index anywhere in its forward pass)
#
# Same rigor split (train<=27, val 28-34, test>34), same class weighting,
# same val-only threshold tuning, same paired_significance_test from
# metrics_utils.
#
# WHAT WAS TESTED HERE, STATED HONESTLY: this script's LOGIC was smoke-
# tested against synthetic Elliptic-shaped dummy data (165 features, ~1000
# nodes, timesteps 1-49) to confirm it runs without errors end to end --
# NOT against your real elliptic_txs_*.csv files, which weren't available
# in this session. Please run it for real before trusting any numbers.
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
#     run_multi_seed, build_edge_index, paired_significance_test,
# )
#
# FEATURES_PATH = "datasets/elliptic_txs_features.csv"
# EDGES_PATH = "datasets/elliptic_txs_edgelist.csv"
# CLASSES_PATH = "datasets/elliptic_txs_classes.csv"
#
# # original 5 seeds first, then 10 new ones -- the original 5 stay a direct
# # subset so you can sanity-check this run against your existing Step12 numbers
# SEEDS = (42, 1, 7, 123, 2024, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
# EPOCHS = 200
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
#
#
# # ============================================================
# # 1. StructuralOnlyGraphSAGE -- identical to Step17
# # ============================================================
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
#     def forward(self, x, edge_index):
#         h = self.dropout(F.relu(self.block1(x, edge_index)))
#         h = self.dropout(F.relu(self.block2(h, edge_index)))
#         return self.classifier(h)
#
#
# # ============================================================
# # 2. TIEBaseline -- identical to step11, genuinely no edge_index anywhere
# # ============================================================
# class TIEBaseline(nn.Module):
#     def __init__(self, in_channels, num_timesteps, temporal_dim, hidden_channels, out_channels):
#         super().__init__()
#         self.temporal_embed = nn.Embedding(num_timesteps, temporal_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(in_channels + temporal_dim, hidden_channels),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_channels, out_channels),
#         )
#
#     def forward(self, x, time_steps):
#         t_emb = self.temporal_embed(time_steps)
#         combined = torch.cat([x, t_emb], dim=1)
#         return self.mlp(combined)
#
#
# def load_data():
#     print("Loading Elliptic Dataset...")
#     df_feat = pd.read_csv(FEATURES_PATH, header=None)
#     df_edge = pd.read_csv(EDGES_PATH)
#     df_class = pd.read_csv(CLASSES_PATH)
#
#     df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
#     df_class.columns = ["txId", "class"]
#     df_class["label"] = df_class["class"].map({"1": 1, "2": 0, "unknown": -1})
#
#     map_id, edge_index = build_edge_index(df_feat["txId"].values, df_edge["txId1"], df_edge["txId2"])
#     edge_index = edge_index.to(device)
#
#     x_raw = df_feat.drop(columns=["txId", "time_step"]).values
#     scaler = StandardScaler()
#     x = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float).to(device)
#     y = torch.tensor(df_class["label"].values, dtype=torch.long).to(device)
#
#     time_steps_raw = torch.tensor(df_feat["time_step"].values, dtype=torch.long)
#     time_steps = (time_steps_raw - 1).to(device)
#     num_timesteps = int(time_steps.max().item()) + 1
#
#     train_mask, val_mask, test_mask = get_temporal_split_masks(
#         time_steps_raw, y, train_end=27, val_end=34, device=device
#     )
#     print(f"Train: {train_mask.sum().item()}   Val: {val_mask.sum().item()}   Test: {test_mask.sum().item()}")
#
#     n_pos = (y[train_mask] == 1).sum().item()
#     n_neg = (y[train_mask] == 0).sum().item()
#     class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)
#
#     return x, edge_index, y, time_steps, num_timesteps, train_mask, val_mask, test_mask, class_weights
#
#
# def train_eval_graphsage(seed, x, edge_index, y, train_mask, val_mask, test_mask, class_weights):
#     model = StructuralOnlyGraphSAGE(in_channels=165, hidden_channels=64, out_channels=2).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
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
#
#     metrics = evaluate_binary("GraphSAGE", y_test, preds_test, probs_test, verbose=False)
#     metrics["threshold"] = best_t
#     return metrics
#
#
# def train_eval_tie(seed, x, y, time_steps, num_timesteps, train_mask, val_mask, test_mask, class_weights):
#     model = TIEBaseline(
#         in_channels=165, num_timesteps=num_timesteps, temporal_dim=32,
#         hidden_channels=64, out_channels=2,
#     ).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
#
#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         optimizer.zero_grad()
#         out = model(x, time_steps)
#         loss = criterion(out[train_mask], y[train_mask])
#         loss.backward()
#         optimizer.step()
#
#     model.eval()
#     with torch.no_grad():
#         out = model(x, time_steps)
#         probs_all = F.softmax(out, dim=1)[:, 1].cpu().numpy()
#
#     y_val = y[val_mask].cpu().numpy()
#     probs_val = probs_all[val_mask.cpu().numpy()]
#     best_t, _ = find_best_threshold(y_val, probs_val)
#
#     y_test = y[test_mask].cpu().numpy()
#     probs_test = probs_all[test_mask.cpu().numpy()]
#     preds_test = (probs_test >= best_t).astype(int)
#
#     metrics = evaluate_binary("TIE Baseline", y_test, preds_test, probs_test, verbose=False)
#     metrics["threshold"] = best_t
#     return metrics
#
#
# if __name__ == "__main__":
#     x, edge_index, y, time_steps, num_timesteps, train_mask, val_mask, test_mask, class_weights = load_data()
#
#     print(f"\n=== GraphSAGE, {len(SEEDS)} seeds ===")
#     df_sage, summary_sage = run_multi_seed(
#         lambda seed: train_eval_graphsage(seed, x, edge_index, y, train_mask, val_mask, test_mask, class_weights),
#         seeds=SEEDS, name="GraphSAGE ساختاری",
#     )
#
#     print(f"\n=== TIE Baseline, {len(SEEDS)} seeds ===")
#     df_tie, summary_tie = run_multi_seed(
#         lambda seed: train_eval_tie(seed, x, y, time_steps, num_timesteps, train_mask, val_mask, test_mask, class_weights),
#         seeds=SEEDS, name="TIE Baseline",
#     )
#
#     print(f"\n\n=== نتیجه نهایی، {len(SEEDS)} seed ===")
#     print(f"{'GraphSAGE':20s} F1 = {summary_sage.loc['mean', 'F1']:.4f} ± {summary_sage.loc['std', 'F1']:.4f}")
#     print(f"{'TIE (بدون گراف)':20s} F1 = {summary_tie.loc['mean', 'F1']:.4f} ± {summary_tie.loc['std', 'F1']:.4f}")
#     print(f"\nمقایسه با نتیجه پنج-seed قبلی (Step12): GraphSAGE F1≈0.4427, TIE F1≈0.4229, p=0.5576")
#
#     print(f"\n=== آزمون معناداری، {len(SEEDS)} seed ===")
#     paired_significance_test(
#         df_sage["F1"].tolist(), df_tie["F1"].tolist(),
#         "GraphSAGE ساختاری", "TIE بدون گراف",
#     )



import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from metrics_utils import (
    evaluate_binary, find_best_threshold, get_temporal_split_masks,
    run_multi_seed, build_edge_index, paired_significance_test,
)

FEATURES_PATH = "datasets/elliptic_txs_features.csv"
EDGES_PATH = "datasets/elliptic_txs_edgelist.csv"
CLASSES_PATH = "datasets/elliptic_txs_classes.csv"


SEEDS = (42, 1, 7, 123, 2024, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12)
EPOCHS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


def load_data():
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
    time_steps = (time_steps_raw - 1).to(device)
    num_timesteps = int(time_steps.max().item()) + 1

    train_mask, val_mask, test_mask = get_temporal_split_masks(
        time_steps_raw, y, train_end=27, val_end=34, device=device
    )
    print(f"Train: {train_mask.sum().item()}   Val: {val_mask.sum().item()}   Test: {test_mask.sum().item()}")

    n_pos = (y[train_mask] == 1).sum().item()
    n_neg = (y[train_mask] == 0).sum().item()
    class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)

    return x, edge_index, y, time_steps, num_timesteps, train_mask, val_mask, test_mask, class_weights


def train_eval_graphsage(seed, x, edge_index, y, train_mask, val_mask, test_mask, class_weights):
    model = StructuralOnlyGraphSAGE(in_channels=165, hidden_channels=64, out_channels=2).to(device)
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

    metrics = evaluate_binary("GraphSAGE", y_test, preds_test, probs_test, verbose=False)
    metrics["threshold"] = best_t
    return metrics


def train_eval_tie(seed, x, y, time_steps, num_timesteps, train_mask, val_mask, test_mask, class_weights):
    model = TIEBaseline(
        in_channels=165, num_timesteps=num_timesteps, temporal_dim=32,
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


if __name__ == "__main__":
    x, edge_index, y, time_steps, num_timesteps, train_mask, val_mask, test_mask, class_weights = load_data()

    print(f"\n=== GraphSAGE, {len(SEEDS)} seeds ===")
    df_sage, summary_sage = run_multi_seed(
        lambda seed: train_eval_graphsage(seed, x, edge_index, y, train_mask, val_mask, test_mask, class_weights),
        seeds=SEEDS, name="GraphSAGE ساختاری",
    )

    print(f"\n=== TIE Baseline, {len(SEEDS)} seeds ===")
    df_tie, summary_tie = run_multi_seed(
        lambda seed: train_eval_tie(seed, x, y, time_steps, num_timesteps, train_mask, val_mask, test_mask, class_weights),
        seeds=SEEDS, name="TIE Baseline",
    )

    print(f"\n\n=== نتیجه نهایی، {len(SEEDS)} seed ===")
    for metric in ("F1", "PR-AUC", "AUC"):
        print(f"{'GraphSAGE':20s} {metric} = {summary_sage.loc['mean', metric]:.4f} ± {summary_sage.loc['std', metric]:.4f}")
        print(f"{'TIE (بدون گراف)':20s} {metric} = {summary_tie.loc['mean', metric]:.4f} ± {summary_tie.loc['std', metric]:.4f}")
    print(f"\nمقایسه با نتیجه پنج-seed قبلی (Step12): GraphSAGE F1≈0.4427, TIE F1≈0.4229, p=0.5576")


    for metric in ("F1", "PR-AUC", "AUC"):
        print(f"\n=== آزمون معناداری روی {metric}، {len(SEEDS)} seed ===")
        paired_significance_test(
            df_sage[metric].tolist(), df_tie[metric].tolist(),
            f"GraphSAGE ساختاری ({metric})", f"TIE بدون گراف ({metric})",
        )
        wins_sage = int((df_sage[metric].values > df_tie[metric].values).sum())
        print(f"نسبت برد به‌ازای {len(SEEDS)} seed: GraphSAGE {wins_sage}/{len(SEEDS)}   "
              f"TIE {len(SEEDS) - wins_sage}/{len(SEEDS)}")

    df_sage.to_csv("step32_graphsage_15seed_results.csv", index=False)
    df_tie.to_csv("step32_tie_15seed_results.csv", index=False)
    print("\nنتیجه‌ی تک‌تک seed ها ذخیره شد: step32_graphsage_15seed_results.csv و step32_tie_15seed_results.csv")