import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from metrics_utils import evaluate_gnn, set_seed
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("Loading Elliptic Dataset...")
FEATURES_PATH = "datasets/elliptic_txs_features.csv"
EDGES_PATH = "datasets/elliptic_txs_edgelist.csv"
CLASSES_PATH = "datasets/elliptic_txs_classes.csv"

df_feat = pd.read_csv(FEATURES_PATH, header=None)
df_edge = pd.read_csv(EDGES_PATH)
df_class = pd.read_csv(CLASSES_PATH)

df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
df_class.columns = ["txId", "class"]
df_class['label'] = df_class['class'].map({'1': 1, '2': 0, 'unknown': -1})

map_id = {j: i for i, j in enumerate(df_feat['txId'].values)}
edge_index = torch.tensor([
    [map_id[src] for src in df_edge['txId1']],
    [map_id[dst] for dst in df_edge['txId2']]
], dtype=torch.long).to(device)

x_raw = df_feat.drop(columns=['txId', 'time_step']).values
scaler = StandardScaler()
x = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float).to(device)
y = torch.tensor(df_class['label'].values, dtype=torch.long).to(device)

train_mask = (y != -1) & (torch.tensor(df_feat['time_step'].values).to(device) <= 34)
test_mask = (y != -1) & (torch.tensor(df_feat['time_step'].values).to(device) > 34)

n_pos = (y[train_mask] == 1).sum().item()
n_neg = (y[train_mask] == 0).sum().item()
class_weights = torch.tensor([1.0, n_neg / n_pos]).to(device)
print(f"Illicit Weight: {n_neg / n_pos:.2f}")


class GATFraudDetector(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x


model = GATFraudDetector(in_channels=165, hidden_channels=64, out_channels=2, heads=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

print("\nStarting GATv2 Training...")

for epoch in range(1, 401):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

evaluate_gnn("GATv2 (Final)", model, x, edge_index, y, test_mask, two_class_softmax=True)