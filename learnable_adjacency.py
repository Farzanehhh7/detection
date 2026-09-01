"""
learnable_adjacency.py

Learnable Adjacency module for the Elliptic / SAML-D fraud-detection pipeline.

Design choice (per the Triple Attention lesson from Phase 1):
    Every time a stream was forced to COMPETE inside a softmax gate (v1 raw gate,
    v2 entropy-penalized gate), performance dropped. Every time a stream instead
    MODULATED or EXTENDED the existing structure without competing for weight,
    it helped. So this module does NOT add a second gated branch. Instead, it
    adds inferred (behaviorally-similar) edges directly into the same edge_index
    as the real transaction edges, tagged with an `is_inferred` edge feature.
    A single edge-aware GNN (GATv2Conv, which supports edge_dim) then does one
    unified message-passing pass over both real and inferred edges.

Pipeline:
    1. build_behavioral_index / top_k_inferred_edges
       -> FAISS approximate nearest-neighbor search over behavioral feature
          vectors, producing top-k inferred edges + a cosine-similarity score
          per edge. Uses FAISS (not brute-force) because Elliptic has ~200k
          nodes and O(N^2) pairwise similarity is not tractable.
    2. merge_real_and_inferred_edges
       -> concatenates real + inferred edges into a single edge_index, with a
          2-column edge_attr: [is_inferred, weight].
    3. EdgeAwareGraphSAGEPlus
       -> a plain two-layer GATv2Conv model that consumes edge_attr directly.
          No gate, no second stream -- one graph, one model.
    4. check_temporal_leakage
       -> sanity check that inferred edges aren't silently leaking
          train/test-period information (mirrors the edge-leakage check
          already done for real edges in Phase 1, Section 7.2).
    5. evaluate_by_degree_bucket
       -> the key comparison for this module's value: split evaluation into
          low-real-degree nodes vs the rest. The inferred edges should help
          most exactly where the real graph gives little signal (the A-E case).

Usage: import the pieces you need into your existing training script. This
file intentionally does not hardcode your training loop / 5-seed protocol --
plug EdgeAwareGraphSAGEPlus into the same seed loop and rigor split you
already use for the other models in Phase 1/2, so results stay comparable.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv
import faiss


# ---------------------------------------------------------------------------
# 1. Behavioral similarity search
# ---------------------------------------------------------------------------

def build_behavioral_index(features: np.ndarray, use_gpu: bool = False):
    """
    Build a FAISS index for fast approximate cosine-similarity search.

    Args:
        features: (N, D) float32 array of behavioral feature vectors.
                  IMPORTANT: only include features available at train/inference
                  time for every node (no label-derived or future-leaking
                  features) -- see check_temporal_leakage below.
        use_gpu:  set True if you have a CUDA FAISS build available.

    Returns:
        (index, normalized_features) -- normalized_features is what was
        actually indexed (L2-normalized, so inner product == cosine similarity).
    """
    features = features.astype(np.float32).copy()
    faiss.normalize_L2(features)  # in-place; inner product now equals cosine sim
    index = faiss.IndexFlatIP(features.shape[1])
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(features)
    return index, features


def top_k_inferred_edges(features: np.ndarray, k: int = 10, sim_threshold: float = 0.0,
                          exclude_self: bool = True, use_gpu: bool = False):
    """
    For every node, find its top-k most behaviorally-similar neighbors and
    return them as a directed edge list with a similarity score per edge.

    This is the "learnable adjacency" step: k and sim_threshold are the two
    hyperparameters to search over (see module docstring -- use the same
    winner's-curse-aware full-seed reverification protocol as Phase 1,
    Section 7.6 before trusting a chosen k).

    Returns:
        edge_index_inferred: LongTensor [2, E]
        edge_sim:             FloatTensor [E]  (cosine similarity, doubles as
                               an edge weight/feature)
    """
    index, norm_features = build_behavioral_index(features, use_gpu=use_gpu)
    search_k = k + 1 if exclude_self else k  # +1: a node's own nearest neighbor is itself
    sims, idxs = index.search(norm_features, search_k)

    n = features.shape[0]
    src_list, dst_list, sim_list = [], [], []
    for i in range(n):
        for rank in range(search_k):
            j = int(idxs[i, rank])
            s = float(sims[i, rank])
            if exclude_self and j == i:
                continue
            if s < sim_threshold:
                continue
            src_list.append(i)
            dst_list.append(j)
            sim_list.append(s)

    edge_index_inferred = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_sim = torch.tensor(sim_list, dtype=torch.float)
    return edge_index_inferred, edge_sim


# ---------------------------------------------------------------------------
# 2. Merge real + inferred edges into a single graph
# ---------------------------------------------------------------------------

def merge_real_and_inferred_edges(edge_index_real: torch.Tensor,
                                   edge_index_inferred: torch.Tensor,
                                   edge_sim_inferred: torch.Tensor,
                                   real_edge_weight: torch.Tensor = None):
    """
    Merge real transaction edges and inferred behavioral-similarity edges
    into one graph, with a 2-column edge feature: [is_inferred, weight].

    Real edges:     is_inferred = 0, weight = real_edge_weight (default 1.0)
    Inferred edges: is_inferred = 1, weight = cosine similarity
    """
    n_real = edge_index_real.size(1)
    n_inferred = edge_index_inferred.size(1)

    if real_edge_weight is None:
        real_weight = torch.ones(n_real, dtype=torch.float)
    else:
        real_weight = real_edge_weight.view(-1)

    real_flag = torch.zeros(n_real, dtype=torch.float)
    inferred_flag = torch.ones(n_inferred, dtype=torch.float)

    edge_index = torch.cat([edge_index_real, edge_index_inferred], dim=1)
    edge_attr = torch.cat([
        torch.stack([real_flag, real_weight], dim=1),
        torch.stack([inferred_flag, edge_sim_inferred], dim=1),
    ], dim=0)

    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# 3. Single, non-competing edge-aware model
# ---------------------------------------------------------------------------

class EdgeAwareGraphSAGEPlus(nn.Module):
    """
    One unified message-passing stream over real + inferred edges together.
    Deliberately NOT a two-branch gated architecture -- see module docstring
    for why that pattern was avoided.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 edge_dim: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads,
                                edge_dim=edge_dim, dropout=dropout, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1,
                                edge_dim=edge_dim, dropout=dropout, concat=False)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_attr)
        h = F.elu(h)
        return self.classifier(h)


# ---------------------------------------------------------------------------
# 4. Leakage sanity check for the inferred edges
# ---------------------------------------------------------------------------

def check_temporal_leakage(node_timestep: np.ndarray, edge_index_inferred: torch.Tensor,
                            feature_cutoff_timestep: int = None):
    """
    Mirrors the real-edge leakage check from Phase 1 Section 7.2, applied to
    the inferred edges. Reports how many inferred edges connect a
    train-period node to a test-period node, so you can decide whether that
    is acceptable for your training setup (e.g. transductive full-graph
    training) or needs to be filtered out.
    """
    ei = edge_index_inferred.numpy()
    src_t = node_timestep[ei[0]]
    dst_t = node_timestep[ei[1]]
    cross_split = None
    if feature_cutoff_timestep is not None:
        cross_split = int(((src_t <= feature_cutoff_timestep) !=
                            (dst_t <= feature_cutoff_timestep)).sum())
    return {
        "num_inferred_edges": int(ei.shape[1]),
        "num_cross_split_edges": cross_split if cross_split is not None else "not_checked",
        "timestep_range": (int(node_timestep.min()), int(node_timestep.max())),
    }


# ---------------------------------------------------------------------------
# 5. The key evaluation: low-degree subset vs the rest
# ---------------------------------------------------------------------------

def evaluate_by_degree_bucket(y_true, y_pred_proba, real_degree, low_degree_threshold: int = 1):
    """
    Splits evaluation into low-real-degree nodes (<= threshold) vs the rest,
    reporting AUC / PR-AUC / F1 for each subset. This is the comparison that
    actually demonstrates the module's value: the inferred edges should help
    most exactly where the real graph gives little or no signal.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    low_mask = real_degree <= low_degree_threshold
    results = {}
    for name, mask in [("low_degree", low_mask), ("rest", ~low_mask),
                        ("overall", np.ones_like(low_mask, dtype=bool))]:
        if mask.sum() == 0:
            results[name] = None
            continue
        yt, yp = y_true[mask], y_pred_proba[mask]
        if len(np.unique(yt)) < 2:
            results[name] = {"n": int(mask.sum()), "note": "single class in subset, metrics undefined"}
            continue
        results[name] = {
            "n": int(mask.sum()),
            "auc": roc_auc_score(yt, yp),
            "pr_auc": average_precision_score(yt, yp),
            "f1": f1_score(yt, (yp > 0.5).astype(int)),
        }
    return results


# ---------------------------------------------------------------------------
# Smoke test on synthetic data -- confirms the pipeline runs end-to-end
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    n_nodes, n_feat = 500, 16
    x = torch.randn(n_nodes, n_feat)
    behavioral_features = x.numpy()

    n_real_edges = 800
    src = np.random.randint(0, n_nodes, n_real_edges)
    dst = np.random.randint(0, n_nodes, n_real_edges)
    edge_index_real = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    edge_index_inferred, edge_sim = top_k_inferred_edges(behavioral_features, k=5, sim_threshold=0.0)
    edge_index, edge_attr = merge_real_and_inferred_edges(edge_index_real, edge_index_inferred, edge_sim)

    model = EdgeAwareGraphSAGEPlus(in_channels=n_feat, hidden_channels=32, out_channels=2)
    out = model(x, edge_index, edge_attr)
    print("Output shape:", tuple(out.shape))

    node_timestep = np.random.randint(1, 50, size=n_nodes)
    leakage_report = check_temporal_leakage(node_timestep, edge_index_inferred, feature_cutoff_timestep=34)
    print("Leakage report:", leakage_report)

    real_degree = np.bincount(edge_index_real.numpy().flatten(), minlength=n_nodes)
    y_true = np.random.randint(0, 2, n_nodes)
    y_pred_proba = torch.softmax(out, dim=1)[:, 1].detach().numpy()
    degree_report = evaluate_by_degree_bucket(y_true, y_pred_proba, real_degree)
    print("Degree-bucket report:")
    for k_, v_ in degree_report.items():
        print(f"  {k_}: {v_}")