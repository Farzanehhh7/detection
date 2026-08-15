import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from metrics_utils import evaluate_gnn, set_seed
set_seed(42)

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
], dtype=torch.long)

x_raw = df_feat.drop(columns=['txId', 'time_step']).values
scaler = StandardScaler()
x = torch.tensor(scaler.fit_transform(x_raw), dtype=torch.float)
y = torch.tensor(df_class['label'].values, dtype=torch.long)

train_mask = (y != -1) & (torch.tensor(df_feat['time_step'].values) <= 34)
test_mask = (y != -1) & (torch.tensor(df_feat['time_step'].values) > 34)

n_pos = (y[train_mask] == 1).sum().item()
n_neg = (y[train_mask] == 0).sum().item()
class_weights = torch.tensor([1.0, n_neg / n_pos])


class ProGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        self.lin_skip = torch.nn.Linear(in_channels, hidden_channels)

    def forward(self, x, edge_index):
        identity = self.lin_skip(x)
        x_out = self.conv1(x, edge_index)
        x = F.relu(x_out + identity)  # Skip Connection
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv3(x, edge_index)
        return x


model = ProGraphSAGE(in_channels=165, hidden_channels=256, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

print(f"Starting Professional Training (Class Weight: {n_neg / n_pos:.2f})...")

for epoch in range(1, 301):
    model.train()
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

evaluate_gnn("GraphSAGE Pro (Final)", model, x, edge_index, y, test_mask, two_class_softmax=True)