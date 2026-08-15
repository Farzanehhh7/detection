"""
اندازه‌گیری heterophily روی Elliptic
=====================================================================
دو تعریف رایج heterophily محاسبه می‌شود:

  edge homophily — از بین یال‌هایی که هر دو سرشان برچسب‌دار هستند،
  چه نسبتی دو سر با برچسب یکسان دارند.

  node homophily — برای هر گره برچسب‌دار، از بین همسایه‌های
  برچسب‌دارش، چه نسبتی برچسب یکسان با خودش دارند؛ جداگانه برای
  گره‌های illicit و licit گزارش می‌شود چون این تفکیک همان چیزی‌ست
  که مستقیم به بحث فصل شش مربوط می‌شود.

عدد نزدیک به یک یعنی homophily بالا، یال‌ها بیشتر بین گره‌های
هم‌برچسب. عدد پایین، به‌خصوص برای گره‌های illicit، یعنی بیشتر
همسایه‌های یک گره مشکوک در واقع مجازند، که دقیقاً همان نگرانی‌ای‌ست
که Dang و همکاران و Xu و همکاران درباره Elliptic مطرح کرده‌اند و
توضیح می‌دهد چرا میانگین‌گیری روی همسایگی سیگنال illicit را رقیق
می‌کند.
"""

import numpy as np
import pandas as pd
from collections import defaultdict

FEATURES_PATH = "datasets/elliptic_txs_features.csv"
EDGES_PATH = "datasets/elliptic_txs_edgelist.csv"
CLASSES_PATH = "datasets/elliptic_txs_classes.csv"

df_feat = pd.read_csv(FEATURES_PATH, header=None)
df_edge = pd.read_csv(EDGES_PATH)
df_class = pd.read_csv(CLASSES_PATH)

df_feat.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
df_class.columns = ["txId", "class"]
df_class["label"] = df_class["class"].map({"1": 1, "2": 0, "unknown": -1})

label_map = dict(zip(df_class["txId"], df_class["label"]))

src_labels = df_edge["txId1"].map(label_map)
dst_labels = df_edge["txId2"].map(label_map)
both_labeled = (src_labels != -1) & (dst_labels != -1)

n_both_labeled = int(both_labeled.sum())
n_same = int(((src_labels == dst_labels) & both_labeled).sum())
edge_homophily = n_same / n_both_labeled

print(f"تعداد کل یال: {len(df_edge)}")
print(f"یال با هر دو سر برچسب‌دار: {n_both_labeled}")
print(f"\nEdge homophily کلی: {edge_homophily:.4f}")

illicit_pairs = int(((src_labels == 1) & (dst_labels == 1) & both_labeled).sum())
licit_pairs = int(((src_labels == 0) & (dst_labels == 0) & both_labeled).sum())
mixed_pairs = n_both_labeled - illicit_pairs - licit_pairs

print(f"\nیال illicit به illicit: {illicit_pairs}   نسبت: {illicit_pairs/n_both_labeled:.4f}")
print(f"یال licit به licit:     {licit_pairs}   نسبت: {licit_pairs/n_both_labeled:.4f}")
print(f"یال مختلط:              {mixed_pairs}   نسبت: {mixed_pairs/n_both_labeled:.4f}")

# ------------------------------------------------------------
# node homophily، به تفکیک illicit در برابر licit
# ------------------------------------------------------------
neighbors = defaultdict(set)
for src, dst, ls, ld in zip(df_edge["txId1"], df_edge["txId2"], src_labels, dst_labels):
    if ls != -1 and ld != -1:
        neighbors[src].add(dst)
        neighbors[dst].add(src)

node_scores_all, node_scores_illicit, node_scores_licit = [], [], []
for node, neighs in neighbors.items():
    own_label = label_map.get(node, -1)
    if own_label == -1:
        continue
    neigh_labels = [label_map.get(n, -1) for n in neighs]
    neigh_labels = [l for l in neigh_labels if l != -1]
    if not neigh_labels:
        continue
    score = np.mean([1 if l == own_label else 0 for l in neigh_labels])
    node_scores_all.append(score)
    (node_scores_illicit if own_label == 1 else node_scores_licit).append(score)

print(f"\nNode homophily، میانگین کل:              {np.mean(node_scores_all):.4f}")
print(f"Node homophily، فقط گره‌های illicit:       {np.mean(node_scores_illicit):.4f}   بر اساس {len(node_scores_illicit)} گره")
print(f"Node homophily، فقط گره‌های licit:         {np.mean(node_scores_licit):.4f}   بر اساس {len(node_scores_licit)} گره")

print("\nتفسیر: عدد node homophily گره‌های illicit اگر زیر 0.5 باشد، یعنی")
print("بیشتر همسایه‌های یک گره مشکوک در واقع مجازند، و میانگین‌گیری")
print("SAGEConv یا هر جریان مبتنی بر میانگین همسایگی، به‌طور نظام‌مند")
print("سیگنال illicit را در جهت اکثریت همسایگان مجاز رقیق می‌کند.")