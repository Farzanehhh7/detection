"""
experiment_time_of_day_test.py

تست دوم نشون داد یادگیری تابع شباهت کمک می‌کنه (۲٪ -> ۶٪)، ولی هنوز
خیلی پایین‌تر از گراف واقعیه (۳۹.۴٪). فرضیه: ۱۱ فیچر فعلی هیچ‌کدوم
"الگوی ساعت روز" رو کد نمی‌کنن -- دقیقاً همون سیگنالی که مثال A-E در
توضیح Learnable Adjacency روش تکیه داشت (فعالیت هماهنگ ساعت ۳ بامداد).

این اسکریپت دو فیچر جدید می‌سازه (فقط از تراکنش‌های pre-cutoff، عین
step35، بدون نشت):
  - typical_hour_sin / typical_hour_cos: میانگین دایره‌ای ساعت فعالیت
    هر حساب (sin/cos برای جلوگیری از ناپیوستگی نیمه‌شب: ساعت ۲۳ و ۰
    باید نزدیک هم باشن، نه دور)
  - hour_concentration: چقدر فعالیت حساب متمرکز روی یک ساعت خاصه
    (نزدیک ۱ = همیشه یک ساعت ثابت فعاله؛ نزدیک ۰ = پخش در طول روز)

بعد دقیقاً همون تست embedding-یادگرفته‌شده رو، این‌بار با ۱۴ فیچر
(۱۱ قبلی + ۳ فیچر ساعت)، تکرار می‌کنیم -- تا مستقیم ببینیم آیا این
فیچر جدید فاصله تا گراف واقعی (۳۹.۴٪) رو کم می‌کنه یا نه.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from metrics_utils import set_seed

FILE_PATH = "datasets/SAML-D.csv"
N_ROWS = 1_000_000
DATA_PATH = "samld_processed_v3.pt"
BEHAVIOURAL_PATH = "samld_behavioural_features.pt"

SENDER_COL, RECEIVER_COL = "Sender_account", "Receiver_account"
K = 5
N_ILLICIT_TO_CHECK = 50
EPOCHS = 60
SEED = 42


def compute_hour_features():
    print(f"در حال بارگذاری {N_ROWS} ردیف اول از SAML-D برای استخراج فیچر ساعت...")
    df = pd.read_csv(FILE_PATH, nrows=N_ROWS)

    df["_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("_dt").reset_index(drop=True)
    n = len(df)
    cutoff70_dt = df.loc[int(n * 0.70), "_dt"]
    df_pre = df[df["_dt"] <= cutoff70_dt].copy()

    # پارس ستون Time به ساعت عددی (۰ تا ۲۳)، مستقل از فرمت دقیق (HH:MM:SS یا مشابه)
    time_parsed = pd.to_datetime(df_pre["Time"], errors="coerce", format="mixed")
    df_pre["_hour"] = time_parsed.dt.hour
    valid_hour = df_pre["_hour"].notna()
    print(f"{valid_hour.sum()} از {len(df_pre)} تراکنش pre-cutoff ساعت معتبر دارن "
          f"({100 * valid_hour.mean():.1f}%)")
    df_pre = df_pre[valid_hour]

    all_accounts = pd.unique(df[[SENDER_COL, RECEIVER_COL]].values.ravel())

    sent = df_pre[[SENDER_COL, "_hour"]].rename(columns={SENDER_COL: "account", "_hour": "hour"})
    recv = df_pre[[RECEIVER_COL, "_hour"]].rename(columns={RECEIVER_COL: "account", "_hour": "hour"})
    combined = pd.concat([sent, recv], ignore_index=True)

    # میانگین دایره‌ای: هر ساعت رو به رادیان تبدیل می‌کنیم، میانگین sin و cos رو می‌گیریم
    angles = combined["hour"] * (2 * np.pi / 24)
    combined["sin_h"] = np.sin(angles)
    combined["cos_h"] = np.cos(angles)

    grouped = combined.groupby("account")[["sin_h", "cos_h"]].mean()
    # طول بردار میانگین = معیار تمرکز؛ نزدیک ۱ یعنی همیشه یک ساعت ثابت، نزدیک ۰ یعنی پخش در طول روز
    concentration = np.sqrt(grouped["sin_h"] ** 2 + grouped["cos_h"] ** 2)

    feat_df = pd.DataFrame({
        "typical_hour_sin": grouped["sin_h"],
        "typical_hour_cos": grouped["cos_h"],
        "hour_concentration": concentration,
    }).reindex(all_accounts).fillna(0.0)

    return feat_df, all_accounts


def main():
    set_seed(SEED)
    hour_feat_df, all_accounts_from_csv = compute_hour_features()

    print("\nدر حال بارگذاری داده‌ی پردازش‌شده و فیچرهای رفتاری قبلی...")
    data = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    x = data["x"]
    y_binary = data["y_binary"]
    train_mask = data["train_mask"]
    test_mask = data["test_mask"]
    saved_account_ids = data["account_ids"]

    behavioural = torch.load(BEHAVIOURAL_PATH, map_location="cpu", weights_only=False)
    behav_x = behavioural["features_scaled"]

    # align دقیق به همون ترتیب ذخیره‌شده -- عین step35، نه فرض بر یکسان بودن ترتیب
    hour_feat_aligned = hour_feat_df.reindex(saved_account_ids).fillna(0.0)
    hour_raw = hour_feat_aligned.values

    # sin/cos از قبل بین -۱ و ۱ محدودن، نیازی به scale ندارن؛ فقط concentration رو train-only scale می‌کنیم
    concentration_scaler = StandardScaler().fit(hour_raw[train_mask.numpy(), 2:3])
    hour_scaled = hour_raw.copy()
    hour_scaled[:, 2:3] = concentration_scaler.transform(hour_raw[:, 2:3])
    hour_tensor = torch.tensor(hour_scaled, dtype=torch.float32)

    behavior_vec = torch.cat([x, behav_x, hour_tensor], dim=1)
    print(f"بردار رفتاری هر حساب حالا: {behavior_vec.shape[1]} بعد "
          f"(۸ اصلی + ۳ رفتاری + ۳ ساعت)")

    n_pos = (y_binary[train_mask] == 1).sum().item()
    n_neg = (y_binary[train_mask] == 0).sum().item()
    class_weight = torch.tensor([1.0, min(n_neg / max(n_pos, 1), 30.0)])

    model = nn.Sequential(nn.Linear(behavior_vec.shape[1], 32), nn.ReLU())
    classifier_head = nn.Linear(32, 2)
    params = list(model.parameters()) + list(classifier_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.01, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    print(f"\nدر حال آموزش MLP کوچیک با supervision، {EPOCHS} epoch...")
    x_train = behavior_vec[train_mask]
    y_train = y_binary[train_mask]
    model.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        h = model(x_train)
        out = classifier_head(h)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"  epoch={epoch:03d}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        learned_embedding = model(behavior_vec).numpy()
    y_binary_np = y_binary.numpy()
    test_mask_np = test_mask.numpy()

    nn_index = NearestNeighbors(n_neighbors=K + 1, algorithm="auto", metric="euclidean")
    nn_index.fit(learned_embedding)

    illicit_test_idx = np.where((y_binary_np == 1) & test_mask_np)[0]
    sample_idx = illicit_test_idx[:N_ILLICIT_TO_CHECK]
    _, neighbor_idx = nn_index.kneighbors(learned_embedding[sample_idx])

    top1_illicit_count = 0
    any_of_top_k_illicit_count = 0
    for row_i in range(len(sample_idx)):
        neighbors = neighbor_idx[row_i][1:K + 1]
        neighbor_labels = y_binary_np[neighbors]
        if neighbor_labels[0] == 1:
            top1_illicit_count += 1
        if neighbor_labels.any():
            any_of_top_k_illicit_count += 1

    n = len(sample_idx)
    print(f"\n{'=' * 70}\n=== نتیجه با فیچر ساعت اضافه‌شده ===\n{'=' * 70}")
    print(f"نزدیک‌ترین همسایه illicit: {top1_illicit_count}/{n} ({100 * top1_illicit_count / n:.1f}%)")
    print(f"حداقل یکی از {K} همسایه illicit: {any_of_top_k_illicit_count}/{n} "
          f"({100 * any_of_top_k_illicit_count / n:.1f}%)")

    print(f"\n{'=' * 70}\n=== مقایسه‌ی چهارطرفه ===\n{'=' * 70}")
    print(f"گراف واقعی تراکنش (step33)                    : 39.4%")
    print(f"شباهت خام، بدون فیچر ساعت (تست اول)             : 2.0%")
    print(f"شباهت یادگرفته‌شده، بدون فیچر ساعت (تست دوم)     : 6.0%")
    print(f"شباهت یادگرفته‌شده، با فیچر ساعت (این‌جا)         : {100 * top1_illicit_count / n:.1f}%")


if __name__ == "__main__":
    main()