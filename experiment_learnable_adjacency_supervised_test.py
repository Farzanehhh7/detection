"""
experiment_learnable_adjacency_supervised_test.py

نسخه‌ی دوم و منصفانه‌تر تست ایده‌ی Learnable Adjacency. تست اول
(experiment_learnable_adjacency_standalone_test.py) شباهت خام cosine
روی ۱۱ فیچر رو سنجید و سیگنال ضعیفی داد (۲٪ تا ۱۴٪ در برابر ۳۹.۴٪
گراف واقعی). ولی اون تست، تابع شباهت رو "یاد نگرفته بود" -- فقط فیچر
خام رو مقایسه کرد.

اینجا دقیقاً همون سوالی که فرزانه مطرح کرد رو می‌سنجیم: آیا یک تابع
شباهتِ یادگرفته‌شده (نه خام) می‌تونه رابطه‌های پنهانی که در فیچر خام
دیده نمی‌شن رو کشف کنه؟

روش: یک MLP کوچیک دو-لایه با supervision (illicit/licit) روی همین
۱۱ فیچر train می‌کنیم -- کار اصلیش طبقه‌بندیه، ولی لایه‌ی میانی
(hidden layer) به‌عنوان یک embedding یادگرفته‌شده استفاده می‌شه.
اگه MLP یاد گرفته باشه چه ترکیبی از فیچرها واقعاً illicit رو مشخص
می‌کنه، حساب‌های illicit باید در فضای این embedding به هم نزدیک‌تر
باشن -- حتی اگه در فضای فیچر خام نزدیک نبودن.

دقیقاً همون چک step33/تست اول رو روی این embedding جدید تکرار
می‌کنیم، تا سه عدد مستقیماً قابل‌مقایسه باشن:
  - گراف واقعی تراکنش (step33):             39.4%
  - شباهت خام فیچر (تست اول):               2.0%
  - شباهت یادگرفته‌شده (این‌جا):              ؟
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from metrics_utils import set_seed

DATA_PATH = "samld_processed_v3.pt"
BEHAVIOURAL_PATH = "samld_behavioural_features.pt"
K = 5
N_ILLICIT_TO_CHECK = 50
EPOCHS = 60
SEED = 42


class SmallMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x, return_embedding=False):
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        if return_embedding:
            return out, h
        return out


def main():
    set_seed(SEED)
    print("در حال بارگذاری فیچرهای واقعی و تاییدشده...")
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data["x"]
    y_binary = data["y_binary"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]

    behavioural = torch.load(BEHAVIOURAL_PATH, map_location="cpu", weights_only=False)
    behav_x = behavioural["features_scaled"]

    behavior_vec = torch.cat([x, behav_x], dim=1)
    print(f"بردار رفتاری هر حساب: {behavior_vec.shape[1]} بعد")

    n_pos = (y_binary[train_mask] == 1).sum().item()
    n_neg = (y_binary[train_mask] == 0).sum().item()
    class_weight = torch.tensor([1.0, min(n_neg / max(n_pos, 1), 30.0)])
    print(f"وزن کلاس illicit: {class_weight[1].item():.2f}")

    model = SmallMLP(in_dim=behavior_vec.shape[1], hidden_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    print(f"\nدر حال آموزش MLP کوچیک با supervision (illicit/licit)، {EPOCHS} epoch...")
    model.train()
    x_train = behavior_vec[train_mask]
    y_train = y_binary[train_mask]
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        out = model(x_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  epoch={epoch:03d}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        _, learned_embedding = model(behavior_vec, return_embedding=True)
    learned_embedding = learned_embedding.numpy()
    y_binary_np = y_binary.numpy()
    test_mask_np = test_mask.numpy()

    print(f"\nساخت ایندکس نزدیک‌ترین‌همسایه روی embedding یادگرفته‌شده ({learned_embedding.shape[1]} بعد)...")
    nn_index = NearestNeighbors(n_neighbors=K + 1, algorithm="auto", metric="euclidean")
    nn_index.fit(learned_embedding)

    illicit_test_idx = np.where((y_binary_np == 1) & test_mask_np)[0]
    sample_idx = illicit_test_idx[:N_ILLICIT_TO_CHECK]
    _, neighbor_idx = nn_index.kneighbors(learned_embedding[sample_idx])

    top1_illicit_count = 0
    any_of_top_k_illicit_count = 0
    for row_i, acc_idx in enumerate(sample_idx):
        neighbors = neighbor_idx[row_i][1:K + 1]
        neighbor_labels = y_binary_np[neighbors]
        if neighbor_labels[0] == 1:
            top1_illicit_count += 1
        if neighbor_labels.any():
            any_of_top_k_illicit_count += 1

    n = len(sample_idx)
    print(f"\n{'=' * 70}\n=== نتیجه‌ی embedding یادگرفته‌شده (با supervision) ===\n{'=' * 70}")
    print(f"نزدیک‌ترین همسایه، خودش illicit بود: {top1_illicit_count}/{n} ({100 * top1_illicit_count / n:.1f}%)")
    print(f"حداقل یکی از {K} همسایه‌ی برتر illicit بود: {any_of_top_k_illicit_count}/{n} "
          f"({100 * any_of_top_k_illicit_count / n:.1f}%)")

    print(f"\n{'=' * 70}\n=== مقایسه‌ی سه‌طرفه ===\n{'=' * 70}")
    print(f"گراف واقعی تراکنش (step33)          : 39.4%")
    print(f"شباهت خام فیچر (تست اول)             : 2.0%")
    print(f"شباهت یادگرفته‌شده با supervision (این‌جا): {100 * top1_illicit_count / n:.1f}%")
    print("\nاگه این عدد نزدیک تست اول (۲٪) موند، مشکل از فیچرهاست نه تابع شباهت.")
    print("اگه به‌طور محسوس بالاتر رفت (نزدیک‌تر به ۳۹.۴٪)، یعنی یادگیری تابع شباهت واقعاً کمک می‌کنه.")


if __name__ == "__main__":
    main()