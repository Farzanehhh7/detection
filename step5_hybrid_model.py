import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from metrics_utils import evaluate_gnn, set_seed
set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df_feat = pd.read_csv("datasets/elliptic_txs_features.csv", header=None)
df_edge = pd.read_csv("datasets/elliptic_txs_edgelist.csv")
df_class = pd.read_csv("datasets/elliptic_txs_classes.csv")

df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
df_class.columns = ["txId", "class"]
df_class['label'] = df_class['class'].map({'1': 1, '2': 0, 'unknown': -1})

map_id = {j: i for i, j in enumerate(df_feat['txId'].values)}
edge_index = torch.tensor([[map_id[src] for src in df_edge['txId1']],
                           [map_id[dst] for dst in df_edge['txId2']]], dtype=torch.long).to(device)

x_raw = df_feat.drop(columns=['txId', 'time_step']).values
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_raw)
x = torch.tensor(x_scaled, dtype=torch.float).to(device)
y = torch.tensor(df_class['label'].values, dtype=torch.long).to(device)

train_mask = (y != -1) & (torch.tensor(df_feat['time_step'].values).to(device) <= 34)
test_mask = (y != -1) & (torch.tensor(df_feat['time_step'].values).to(device) > 34)

class EmbeddingExtractor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embedding_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return x

emb_model = EmbeddingExtractor(in_channels=165, hidden_channels=128, embedding_dim=64).to(device)
optimizer = torch.optim.Adam(emb_model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

emb_model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = emb_model(x, edge_index)
    loss = criterion(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()

emb_model.eval()
with torch.no_grad():
    graph_embeddings = emb_model(x, edge_index).cpu().numpy()

X_combined = np.hstack([x_raw, graph_embeddings])
X_train = X_combined[train_mask.cpu()]
y_train = y[train_mask].cpu().numpy()
X_test = X_combined[test_mask.cpu()]
y_test = y[test_mask].cpu().numpy()

n_neg_train = (y_train == 0).sum()
n_pos_train = (y_train == 1).sum()
scale_pos_weight = float(n_neg_train / n_pos_train)

xgb_hybrid = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_hybrid.fit(X_train, y_train)

y_pred = xgb_hybrid.predict(X_test)
y_prob = xgb_hybrid.predict_proba(X_test)[:, 1]

print(f"F1 illicit: {f1_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")