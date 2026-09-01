

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

DATA_PATH = "samld_processed_v3.pt"
BEHAVIOURAL_PATH = "samld_behavioural_features.pt"
K = 5  # همون تعداد همسایه‌ی برتر که step33 هم استفاده کرد، برای مقایسه‌ی مستقیم
N_ILLICIT_TO_CHECK = 50  # همون تعداد نمونه‌ی step33، برای مقایسه‌ی دقیق


def main():
    print("در حال بارگذاری فیچرهای واقعی و تاییدشده...")
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data["x"].numpy()  # 8 فیچر اصلی، از قبل train-only scale شده
    y_binary = data["y_binary"].numpy()
    test_mask = data["test_mask"].numpy()

    behavioural = torch.load(BEHAVIOURAL_PATH, map_location="cpu", weights_only=False)
    behav_x = behavioural["features_scaled"].numpy()  # 3 فیچر رفتاری، از قبل train-only scale شده

    behavior_vec = np.concatenate([x, behav_x], axis=1)
    print(f"بردار رفتاری هر حساب: {behavior_vec.shape[1]} بعد (۸ اصلی + ۳ رفتاری)")

    # نرمال‌سازی به بردار واحد، تا فاصله‌ی اقلیدسی روی این بردارها دقیقاً
    # معادل cosine similarity باشه -- امکان استفاده از الگوریتم tree-based سریع رو می‌ده
    norms = np.linalg.norm(behavior_vec, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit_vec = behavior_vec / norms

    print(f"\nساخت ایندکس نزدیک‌ترین‌همسایه روی {unit_vec.shape[0]} حساب...")
    nn_index = NearestNeighbors(n_neighbors=K + 1, algorithm="auto", metric="euclidean")
    nn_index.fit(unit_vec)

    illicit_test_idx = np.where((y_binary == 1) & test_mask)[0]
    print(f"{len(illicit_test_idx)} حساب illicit در test موجوده؛ "
          f"{min(N_ILLICIT_TO_CHECK, len(illicit_test_idx))} تای اول رو چک می‌کنیم "
          f"(دقیقاً هم‌اندازه‌ی نمونه‌ی step33 برای مقایسه‌ی مستقیم).")

    sample_idx = illicit_test_idx[:N_ILLICIT_TO_CHECK]
    _, neighbor_idx = nn_index.kneighbors(unit_vec[sample_idx])

    top1_illicit_count = 0
    any_of_top_k_illicit_count = 0
    for row_i, acc_idx in enumerate(sample_idx):
        # اولین همسایه خودِ حسابه (فاصله صفر)، پس از دومی شروع می‌کنیم
        neighbors = neighbor_idx[row_i][1:K + 1]
        neighbor_labels = y_binary[neighbors]
        if neighbor_labels[0] == 1:
            top1_illicit_count += 1
        if neighbor_labels.any():
            any_of_top_k_illicit_count += 1

    n = len(sample_idx)
    base_rate = y_binary.mean()

    print(f"\n{'=' * 70}\n=== نتیجه‌ی گراف شباهتی (بدون هیچ یال تراکنش واقعی) ===\n{'=' * 70}")
    print(f"نزدیک‌ترین همسایه‌ی رفتاری، خودش illicit بود: {top1_illicit_count}/{n} "
          f"({100 * top1_illicit_count / n:.1f}%)")
    print(f"حداقل یکی از {K} همسایه‌ی برتر illicit بود: {any_of_top_k_illicit_count}/{n} "
          f"({100 * any_of_top_k_illicit_count / n:.1f}%)")
    print(f"\nنرخ پایه‌ی illicit در کل داده: {100 * base_rate:.2f}%")
    print(f"نرخ پایه‌ی illicit در کل داده (برای مقایسه با گراف واقعی step33): {100 * base_rate:.2f}%")
    print(f"\nبرای مقایسه‌ی مستقیم، نتیجه‌ی step33 روی گراف واقعی تراکنش بود:")
    print(f"  نزدیک‌ترین همسایه illicit: 39.4% (13/33)")
    print(f"  حداقل یکی از ۵ همسایه illicit: 63.6% (21/33)")


if __name__ == "__main__":
    main()