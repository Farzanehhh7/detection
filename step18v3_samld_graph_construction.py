# # """
# # فاز سه، قدم اول، نسخه سه — تقسیم اکانت بر اساس آخرین فعالیت، نه اولین ظهور و نه stratified random
# # =====================================================================
# # نسخه یک: تقسیم بر اساس اولین ظهور -> صددرصد اکانت‌های val/test فیچر صفر می‌گرفتن.
# # نسخه دو: فیچر از کل تاریخچه + تقسیم stratified random -> فیچر درست شد، ولی
# #   تقسیم دیگه زمانی نبود، دقیقاً همون ناسازگاری‌ای که در بخش ۶.۳ سند مرجع
# #   با مثال ChronoWave-GNN و در نقد مقاله Raja و همکاران رد کرده بودیم.
# #
# # این‌جا: برش زمانی فیچر همون مرز هفتاد درصدی قبلیه، ثابت برای همه اکانت‌ها.
# # تقسیم اکانت اما بر اساس زمان آخرین تراکنش هر اکانت است، نه اولین:
# #   - آخرین فعالیت پیش از مرز هفتاد درصد -> train
# #   - آخرین فعالیت بین مرز هفتاد و هشتادوپنج درصد -> val
# #   - آخرین فعالیت بعد از مرز هشتادوپنج درصد -> test
# #
# # چون اکثر اکانت‌ها بیش از یک تراکنش دارن، اکانت val/test معمولاً هم
# # فعالیت پیش از cutoff فیچر داره، هم فعالیت بعدش؛ یعنی فیچر معنادار
# # می‌گیره بدون این‌که تقسیم غیرزمانی بشه. اکانت‌هایی که واقعاً فقط بعد
# # از cutoff فیچر ظاهر می‌شن، به‌درستی cold-start می‌مونن و صریح گزارش
# # می‌شن.
# #
# # edge_index هم‌چنان از همه تراکنش‌ها ساخته می‌شه، همون تقریب آگاهانه
# # نسخه قبلی که توپولوژی رو دانش ثابت در نظر می‌گیره.
# # """
# #
# # import numpy as np
# # import pandas as pd
# # import torch
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# #
# # from metrics_utils import build_edge_index
# #
# # FILE_PATH = "datasets/SAML-D.csv"
# # N_ROWS = 1_000_000
# # OUTPUT_PREFIX = "samld_processed_v3"
# # import joblib
# # joblib.dump( "samld_scaler_v3.pkl")
# # print("scaler هم ذخیره شد در samld_scaler_v3.pkl")
# #
# #
# # print(f"در حال بارگذاری {N_ROWS} ردیف اول از SAML-D...")
# # df = pd.read_csv(FILE_PATH, nrows=N_ROWS)
# # print("ستون‌های موجود:", list(df.columns))
# #
# # SENDER_COL, RECEIVER_COL = "Sender_account", "Receiver_account"
# # LABEL_COL, TYPE_COL = "Is_laundering", "Laundering_type"
# # AMOUNT_COL, PTYPE_COL = "Amount", "Payment_type"
# #
# #
# # # ============================================================
# # # ۱. دو مرز زمانی سراسری، هفتاد و هشتادوپنج درصد، دقیقاً مثل نسخه‌های قبلی
# # # ============================================================
# # date_col = None
# # for candidate in ["Date", "Time", "Date_time", "Timestamp"]:
# #     if candidate in df.columns:
# #         date_col = candidate
# #         break
# #
# # HAS_TEMPORAL_CUTOFF = False
# #
# # if date_col is not None:
# #     df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
# #     if df["_dt"].isna().mean() < 0.05:
# #         df = df.sort_values("_dt").reset_index(drop=True)
# #         n = len(df)
# #         cutoff70_dt = df.loc[int(n * 0.70), "_dt"]
# #         cutoff85_dt = df.loc[int(n * 0.85), "_dt"]
# #         df["pre_cutoff"] = df["_dt"] <= cutoff70_dt
# #         HAS_TEMPORAL_CUTOFF = True
# #         print(f"cutoff فیچر، هفتاد درصد، روی {date_col}: {cutoff70_dt}")
# #         print(f"cutoff دوم، هشتادوپنج درصد، برای مرز val/test: {cutoff85_dt}")
# #     else:
# #         date_col = None
# #
# # if not HAS_TEMPORAL_CUTOFF:
# #     raise RuntimeError(
# #         "بدون ستون تاریخ معتبر، تقسیم بر اساس آخرین فعالیت اصلاً معنا ندارد؛ "
# #         "این نسخه عمداً متوقف می‌شود تا به‌جای fallback نامعتبر، صادقانه خطا بدهد."
# #     )
# #
# #
# # # ============================================================
# # # ۲. edge_index از همه تراکنش‌ها
# # # ============================================================
# # all_accounts = pd.unique(df[[SENDER_COL, RECEIVER_COL]].values.ravel())
# # print(f"\nتعداد کل اکانت‌های یکتا: {len(all_accounts)}")
# # print(f"تعداد کل تراکنش‌ها یعنی یال‌ها: {len(df)}")
# #
# # acc_map, edge_index = build_edge_index(all_accounts, df[SENDER_COL], df[RECEIVER_COL])
# #
# #
# # # ============================================================
# # # ۳. فیچر هر اکانت، فقط از تراکنش‌های پیش از cutoff هفتاد درصد
# # # ============================================================
# # df_hist = df[df["pre_cutoff"]].copy()
# #
# # sent_stats = df_hist.groupby(SENDER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
# # sent_stats.columns = ["sent_amount_sum", "sent_amount_mean", "sent_amount_count"]
# # sent_ptype = df_hist.groupby(SENDER_COL)[PTYPE_COL].nunique().rename("sent_payment_type_nunique")
# #
# # recv_stats = df_hist.groupby(RECEIVER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
# # recv_stats.columns = ["recv_amount_sum", "recv_amount_mean", "recv_amount_count"]
# # recv_ptype = df_hist.groupby(RECEIVER_COL)[PTYPE_COL].nunique().rename("recv_payment_type_nunique")
# #
# # feat_df = pd.concat([sent_stats, sent_ptype, recv_stats, recv_ptype], axis=1)
# # feat_df = feat_df.reindex(all_accounts).fillna(0.0)
# #
# # feature_cols = list(feat_df.columns)
# # print(f"\nستون‌های فیچر اکانت: {feature_cols}")
# #
# # cold_start = (feat_df[["sent_amount_count", "recv_amount_count"]].sum(axis=1) == 0)
# # print(f"تعداد اکانت‌های cold-start کلی: {cold_start.sum()}   از {len(all_accounts)}")
# #
# # scaler = StandardScaler()
# # x = scaler.fit_transform(feat_df.values)
# # x = torch.tensor(x, dtype=torch.float)
# #
# #
# # # ============================================================
# # # ۴. برچسب اکانت، از کل داده، دقیقاً مثل نسخه‌های قبلی
# # # ============================================================
# # illicit_tx = df[df[LABEL_COL] == 1]
# # illicit_accounts = set(illicit_tx[SENDER_COL]).union(set(illicit_tx[RECEIVER_COL]))
# #
# # binary_label = pd.Series(0, index=all_accounts)
# # binary_label.loc[list(illicit_accounts & set(all_accounts))] = 1
# #
# # type_by_account = {}
# # for acc in illicit_accounts:
# #     mask = (illicit_tx[SENDER_COL] == acc) | (illicit_tx[RECEIVER_COL] == acc)
# #     types = illicit_tx.loc[mask, TYPE_COL]
# #     if len(types) > 0:
# #         type_by_account[acc] = types.mode().iloc[0]
# #
# # le_type = LabelEncoder()
# # all_types = sorted(set(type_by_account.values()))
# # le_type.fit(all_types)
# # print(f"\nتعداد انواع پول‌شویی در سطح اکانت: {len(all_types)}")
# #
# # type_label = pd.Series(-1, index=all_accounts)
# # for acc, t in type_by_account.items():
# #     type_label.loc[acc] = le_type.transform([t])[0]
# #
# # y_binary = torch.tensor(binary_label.loc[all_accounts].values, dtype=torch.long)
# # y_type = torch.tensor(type_label.loc[all_accounts].values, dtype=torch.long)
# #
# # print(f"\nتعداد اکانت illicit: {(y_binary == 1).sum().item()}   "
# #       f"از {len(all_accounts)}   نسبت: {(y_binary == 1).float().mean().item():.4f}")
# #
# #
# # # ============================================================
# # # ۵. تقسیم اکانت بر اساس زمان آخرین تراکنش، نه اولین و نه تصادفی
# # # ============================================================
# # last_seen = pd.concat([
# #     df[[SENDER_COL, "_dt"]].rename(columns={SENDER_COL: "account"}),
# #     df[[RECEIVER_COL, "_dt"]].rename(columns={RECEIVER_COL: "account"}),
# # ]).groupby("account")["_dt"].max()
# # last_seen = last_seen.reindex(all_accounts)
# #
# # split_series = pd.Series("test", index=all_accounts)
# # split_series[last_seen <= cutoff70_dt] = "train"
# # split_series[(last_seen > cutoff70_dt) & (last_seen <= cutoff85_dt)] = "val"
# #
# # train_mask = torch.tensor((split_series == "train").values)
# # val_mask = torch.tensor((split_series == "val").values)
# # test_mask = torch.tensor((split_series == "test").values)
# #
# # print(f"\nتقسیم اکانت بر اساس آخرین فعالیت:  train={train_mask.sum().item()}   "
# #       f"val={val_mask.sum().item()}   test={test_mask.sum().item()}")
# #
# # labels_np = y_binary.numpy()
# # print(f"نسبت illicit در هر تقسیم:  "
# #       f"train={labels_np[train_mask.numpy()].mean():.4f}   "
# #       f"val={labels_np[val_mask.numpy()].mean():.4f}   "
# #       f"test={labels_np[test_mask.numpy()].mean():.4f}")
# #
# # cold_start_np = cold_start.reindex(all_accounts).values
# # print(f"\nنرخ cold-start در هر تقسیم:  "
# #       f"train={cold_start_np[train_mask.numpy()].mean():.4f}   "
# #       f"val={cold_start_np[val_mask.numpy()].mean():.4f}   "
# #       f"test={cold_start_np[test_mask.numpy()].mean():.4f}")
# # print("(این‌بار انتظار می‌ره نرخ val/test خیلی پایین‌تر از صددرصد نسخه اول باشه،")
# # print(" ولی صفر هم نباشه، چون اکانت‌های واقعاً جدید هنوز باید cold-start بمونن.)")
# #
# #
# # # ============================================================
# # # ۶. ذخیره
# # # ============================================================
# # torch.save({
# #     "x": x,
# #     "edge_index": edge_index,
# #     "y_binary": y_binary,
# #     "y_type": y_type,
# #     "train_mask": train_mask,
# #     "val_mask": val_mask,
# #     "test_mask": test_mask,
# #     "feature_cols": feature_cols,
# #     "num_types": len(all_types),
# #     "account_ids": list(all_accounts),
# #     "cold_start_mask": torch.tensor(cold_start_np),
# #     "has_temporal_cutoff": HAS_TEMPORAL_CUTOFF,
# # }, f"{OUTPUT_PREFIX}.pt")
# #
# # print(f"\nهمه چیز ذخیره شد در {OUTPUT_PREFIX}.pt")
# # print("اسم فایل عمداً v3 است. step19 تا step23 باید DATA_PATH را به این فایل")
# # print("تغییر بدهند و از نو اجرا بشوند.")
# """
# فاز سه، قدم اول، نسخه سه — تقسیم اکانت بر اساس آخرین فعالیت، نه اولین ظهور و نه stratified random
# =====================================================================
# نسخه یک: تقسیم بر اساس اولین ظهور -> صددرصد اکانت‌های val/test فیچر صفر می‌گرفتن.
# نسخه دو: فیچر از کل تاریخچه + تقسیم stratified random -> فیچر درست شد، ولی
#   تقسیم دیگه زمانی نبود، دقیقاً همون ناسازگاری‌ای که در بخش ۶.۳ سند مرجع
#   با مثال ChronoWave-GNN و در نقد مقاله Raja و همکاران رد کرده بودیم.
#
# این‌جا: برش زمانی فیچر همون مرز هفتاد درصدی قبلیه، ثابت برای همه اکانت‌ها.
# تقسیم اکانت اما بر اساس زمان آخرین تراکنش هر اکانت است، نه اولین:
#   - آخرین فعالیت پیش از مرز هفتاد درصد -> train
#   - آخرین فعالیت بین مرز هفتاد و هشتادوپنج درصد -> val
#   - آخرین فعالیت بعد از مرز هشتادوپنج درصد -> test
#
# چون اکثر اکانت‌ها بیش از یک تراکنش دارن، اکانت val/test معمولاً هم
# فعالیت پیش از cutoff فیچر داره، هم فعالیت بعدش؛ یعنی فیچر معنادار
# می‌گیره بدون این‌که تقسیم غیرزمانی بشه. اکانت‌هایی که واقعاً فقط بعد
# از cutoff فیچر ظاهر می‌شن، به‌درستی cold-start می‌مونن و صریح گزارش
# می‌شن.
#
# edge_index هم‌چنان از همه تراکنش‌ها ساخته می‌شه، همون تقریب آگاهانه
# نسخه قبلی که توپولوژی رو دانش ثابت در نظر می‌گیره.
# """
#
# import numpy as np
# import pandas as pd
# import torch
# import joblib
# from sklearn.preprocessing import LabelEncoder, StandardScaler
#
# from metrics_utils import build_edge_index
#
# FILE_PATH = "datasets/SAML-D.csv"
# N_ROWS = 1_000_000
# OUTPUT_PREFIX = "samld_processed_v3"
#
#
# print(f"در حال بارگذاری {N_ROWS} ردیف اول از SAML-D...")
# df = pd.read_csv(FILE_PATH, nrows=N_ROWS)
# print("ستون‌های موجود:", list(df.columns))
#
# SENDER_COL, RECEIVER_COL = "Sender_account", "Receiver_account"
# LABEL_COL, TYPE_COL = "Is_laundering", "Laundering_type"
# AMOUNT_COL, PTYPE_COL = "Amount", "Payment_type"
#
#
# # ============================================================
# # ۱. دو مرز زمانی سراسری، هفتاد و هشتادوپنج درصد، دقیقاً مثل نسخه‌های قبلی
# # ============================================================
# date_col = None
# for candidate in ["Date", "Time", "Date_time", "Timestamp"]:
#     if candidate in df.columns:
#         date_col = candidate
#         break
#
# HAS_TEMPORAL_CUTOFF = False
#
# if date_col is not None:
#     df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
#     if df["_dt"].isna().mean() < 0.05:
#         df = df.sort_values("_dt").reset_index(drop=True)
#         n = len(df)
#         cutoff70_dt = df.loc[int(n * 0.70), "_dt"]
#         cutoff85_dt = df.loc[int(n * 0.85), "_dt"]
#         df["pre_cutoff"] = df["_dt"] <= cutoff70_dt
#         HAS_TEMPORAL_CUTOFF = True
#         print(f"cutoff فیچر، هفتاد درصد، روی {date_col}: {cutoff70_dt}")
#         print(f"cutoff دوم، هشتادوپنج درصد، برای مرز val/test: {cutoff85_dt}")
#     else:
#         date_col = None
#
# if not HAS_TEMPORAL_CUTOFF:
#     raise RuntimeError(
#         "بدون ستون تاریخ معتبر، تقسیم بر اساس آخرین فعالیت اصلاً معنا ندارد؛ "
#         "این نسخه عمداً متوقف می‌شود تا به‌جای fallback نامعتبر، صادقانه خطا بدهد."
#     )
#
#
# # ============================================================
# # ۲. edge_index از همه تراکنش‌ها
# # ============================================================
# all_accounts = pd.unique(df[[SENDER_COL, RECEIVER_COL]].values.ravel())
# print(f"\nتعداد کل اکانت‌های یکتا: {len(all_accounts)}")
# print(f"تعداد کل تراکنش‌ها یعنی یال‌ها: {len(df)}")
#
# acc_map, edge_index = build_edge_index(all_accounts, df[SENDER_COL], df[RECEIVER_COL])
#
#
# # ============================================================
# # ۳. فیچر هر اکانت، فقط از تراکنش‌های پیش از cutoff هفتاد درصد
# # ============================================================
# df_hist = df[df["pre_cutoff"]].copy()
#
# sent_stats = df_hist.groupby(SENDER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
# sent_stats.columns = ["sent_amount_sum", "sent_amount_mean", "sent_amount_count"]
# sent_ptype = df_hist.groupby(SENDER_COL)[PTYPE_COL].nunique().rename("sent_payment_type_nunique")
#
# recv_stats = df_hist.groupby(RECEIVER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
# recv_stats.columns = ["recv_amount_sum", "recv_amount_mean", "recv_amount_count"]
# recv_ptype = df_hist.groupby(RECEIVER_COL)[PTYPE_COL].nunique().rename("recv_payment_type_nunique")
#
# feat_df = pd.concat([sent_stats, sent_ptype, recv_stats, recv_ptype], axis=1)
# feat_df = feat_df.reindex(all_accounts).fillna(0.0)
#
# feature_cols = list(feat_df.columns)
# print(f"\nستون‌های فیچر اکانت: {feature_cols}")
#
# cold_start = (feat_df[["sent_amount_count", "recv_amount_count"]].sum(axis=1) == 0)
# print(f"تعداد اکانت‌های cold-start کلی: {cold_start.sum()}   از {len(all_accounts)}")
#
# scaler = StandardScaler()
# x = scaler.fit_transform(feat_df.values)
# x = torch.tensor(x, dtype=torch.float)
#
#
# # ============================================================
# # ۴. برچسب اکانت، از کل داده، دقیقاً مثل نسخه‌های قبلی
# # ============================================================
# illicit_tx = df[df[LABEL_COL] == 1]
# illicit_accounts = set(illicit_tx[SENDER_COL]).union(set(illicit_tx[RECEIVER_COL]))
#
# binary_label = pd.Series(0, index=all_accounts)
# binary_label.loc[list(illicit_accounts & set(all_accounts))] = 1
#
# type_by_account = {}
# for acc in illicit_accounts:
#     mask = (illicit_tx[SENDER_COL] == acc) | (illicit_tx[RECEIVER_COL] == acc)
#     types = illicit_tx.loc[mask, TYPE_COL]
#     if len(types) > 0:
#         type_by_account[acc] = types.mode().iloc[0]
#
# le_type = LabelEncoder()
# all_types = sorted(set(type_by_account.values()))
# le_type.fit(all_types)
# print(f"\nتعداد انواع پول‌شویی در سطح اکانت: {len(all_types)}")
#
# type_label = pd.Series(-1, index=all_accounts)
# for acc, t in type_by_account.items():
#     type_label.loc[acc] = le_type.transform([t])[0]
#
# y_binary = torch.tensor(binary_label.loc[all_accounts].values, dtype=torch.long)
# y_type = torch.tensor(type_label.loc[all_accounts].values, dtype=torch.long)
#
# print(f"\nتعداد اکانت illicit: {(y_binary == 1).sum().item()}   "
#       f"از {len(all_accounts)}   نسبت: {(y_binary == 1).float().mean().item():.4f}")
#
#
# # ============================================================
# # ۵. تقسیم اکانت بر اساس زمان آخرین تراکنش، نه اولین و نه تصادفی
# # ============================================================
# last_seen = pd.concat([
#     df[[SENDER_COL, "_dt"]].rename(columns={SENDER_COL: "account"}),
#     df[[RECEIVER_COL, "_dt"]].rename(columns={RECEIVER_COL: "account"}),
# ]).groupby("account")["_dt"].max()
# last_seen = last_seen.reindex(all_accounts)
#
# split_series = pd.Series("test", index=all_accounts)
# split_series[last_seen <= cutoff70_dt] = "train"
# split_series[(last_seen > cutoff70_dt) & (last_seen <= cutoff85_dt)] = "val"
#
# train_mask = torch.tensor((split_series == "train").values)
# val_mask = torch.tensor((split_series == "val").values)
# test_mask = torch.tensor((split_series == "test").values)
#
# print(f"\nتقسیم اکانت بر اساس آخرین فعالیت:  train={train_mask.sum().item()}   "
#       f"val={val_mask.sum().item()}   test={test_mask.sum().item()}")
#
# labels_np = y_binary.numpy()
# print(f"نسبت illicit در هر تقسیم:  "
#       f"train={labels_np[train_mask.numpy()].mean():.4f}   "
#       f"val={labels_np[val_mask.numpy()].mean():.4f}   "
#       f"test={labels_np[test_mask.numpy()].mean():.4f}")
#
# cold_start_np = cold_start.reindex(all_accounts).values
# print(f"\nنرخ cold-start در هر تقسیم:  "
#       f"train={cold_start_np[train_mask.numpy()].mean():.4f}   "
#       f"val={cold_start_np[val_mask.numpy()].mean():.4f}   "
#       f"test={cold_start_np[test_mask.numpy()].mean():.4f}")
# print("(این‌بار انتظار می‌ره نرخ val/test خیلی پایین‌تر از صددرصد نسخه اول باشه،")
# print(" ولی صفر هم نباشه، چون اکانت‌های واقعاً جدید هنوز باید cold-start بمونن.)")
#
#
# # ============================================================
# # ۶. ذخیره
# # ============================================================
# torch.save({
#     "x": x,
#     "edge_index": edge_index,
#     "y_binary": y_binary,
#     "y_type": y_type,
#     "train_mask": train_mask,
#     "val_mask": val_mask,
#     "test_mask": test_mask,
#     "feature_cols": feature_cols,
#     "num_types": len(all_types),
#     "account_ids": list(all_accounts),
#     "cold_start_mask": torch.tensor(cold_start_np),
#     "has_temporal_cutoff": HAS_TEMPORAL_CUTOFF,
# }, f"{OUTPUT_PREFIX}.pt")
#
# joblib.dump(scaler, "samld_scaler_v3.pkl")
# print("scaler هم ذخیره شد در samld_scaler_v3.pkl")
#
# print(f"\nهمه چیز ذخیره شد در {OUTPUT_PREFIX}.pt")
# print("اسم فایل عمداً v3 است. step19 تا step23 باید DATA_PATH را به این فایل")
# print("تغییر بدهند و از نو اجرا بشوند.")


import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

from metrics_utils import build_edge_index

FILE_PATH = "datasets/SAML-D.csv"
N_ROWS = 1_000_000
OUTPUT_PREFIX = "samld_processed_v3"


print(f"در حال بارگذاری {N_ROWS} ردیف اول از SAML-D...")
df = pd.read_csv(FILE_PATH, nrows=N_ROWS)
print("ستون‌های موجود:", list(df.columns))

SENDER_COL, RECEIVER_COL = "Sender_account", "Receiver_account"
LABEL_COL, TYPE_COL = "Is_laundering", "Laundering_type"
AMOUNT_COL, PTYPE_COL = "Amount", "Payment_type"


date_col = None
for candidate in ["Date", "Time", "Date_time", "Timestamp"]:
    if candidate in df.columns:
        date_col = candidate
        break

HAS_TEMPORAL_CUTOFF = False

if date_col is not None:
    df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    if df["_dt"].isna().mean() < 0.05:
        df = df.sort_values("_dt").reset_index(drop=True)
        n = len(df)
        cutoff70_dt = df.loc[int(n * 0.70), "_dt"]
        cutoff85_dt = df.loc[int(n * 0.85), "_dt"]
        df["pre_cutoff"] = df["_dt"] <= cutoff70_dt
        HAS_TEMPORAL_CUTOFF = True
        print(f"cutoff فیچر، هفتاد درصد، روی {date_col}: {cutoff70_dt}")
        print(f"cutoff دوم، هشتادوپنج درصد، برای مرز val/test: {cutoff85_dt}")
    else:
        date_col = None

if not HAS_TEMPORAL_CUTOFF:
    raise RuntimeError(
        "بدون ستون تاریخ معتبر، تقسیم بر اساس آخرین فعالیت اصلاً معنا ندارد؛ "
        "این نسخه عمداً متوقف می‌شود تا به‌جای fallback نامعتبر، صادقانه خطا بدهد."
    )


all_accounts = pd.unique(df[[SENDER_COL, RECEIVER_COL]].values.ravel())
print(f"\nتعداد کل اکانت‌های یکتا: {len(all_accounts)}")
print(f"تعداد کل تراکنش‌ها یعنی یال‌ها: {len(df)}")

acc_map, edge_index = build_edge_index(all_accounts, df[SENDER_COL], df[RECEIVER_COL])


last_seen = pd.concat([
    df[[SENDER_COL, "_dt"]].rename(columns={SENDER_COL: "account"}),
    df[[RECEIVER_COL, "_dt"]].rename(columns={RECEIVER_COL: "account"}),
]).groupby("account")["_dt"].max()
last_seen = last_seen.reindex(all_accounts)

split_series = pd.Series("test", index=all_accounts)
split_series[last_seen <= cutoff70_dt] = "train"
split_series[(last_seen > cutoff70_dt) & (last_seen <= cutoff85_dt)] = "val"

train_mask = torch.tensor((split_series == "train").values)
val_mask = torch.tensor((split_series == "val").values)
test_mask = torch.tensor((split_series == "test").values)



df_hist = df[df["pre_cutoff"]].copy()

sent_stats = df_hist.groupby(SENDER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
sent_stats.columns = ["sent_amount_sum", "sent_amount_mean", "sent_amount_count"]
sent_ptype = df_hist.groupby(SENDER_COL)[PTYPE_COL].nunique().rename("sent_payment_type_nunique")

recv_stats = df_hist.groupby(RECEIVER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
recv_stats.columns = ["recv_amount_sum", "recv_amount_mean", "recv_amount_count"]
recv_ptype = df_hist.groupby(RECEIVER_COL)[PTYPE_COL].nunique().rename("recv_payment_type_nunique")

feat_df = pd.concat([sent_stats, sent_ptype, recv_stats, recv_ptype], axis=1)
feat_df = feat_df.reindex(all_accounts).fillna(0.0)

feature_cols = list(feat_df.columns)
print(f"\nستون‌های فیچر اکانت: {feature_cols}")

cold_start = (feat_df[["sent_amount_count", "recv_amount_count"]].sum(axis=1) == 0)
print(f"تعداد اکانت‌های cold-start کلی: {cold_start.sum()}   از {len(all_accounts)}")

# رفع نشت: قبلا fit_transform روی کل feat_df (train+val+test با هم) بود
#الان fit فقط روی حساب‌های train، exactly مثل step26 و step35
scaler = StandardScaler()
scaler.fit(feat_df.values[train_mask.numpy()])
x = scaler.transform(feat_df.values)
x = torch.tensor(x, dtype=torch.float)


illicit_tx = df[df[LABEL_COL] == 1]
illicit_accounts = set(illicit_tx[SENDER_COL]).union(set(illicit_tx[RECEIVER_COL]))

binary_label = pd.Series(0, index=all_accounts)
binary_label.loc[list(illicit_accounts & set(all_accounts))] = 1

type_by_account = {}
for acc in illicit_accounts:
    mask = (illicit_tx[SENDER_COL] == acc) | (illicit_tx[RECEIVER_COL] == acc)
    types = illicit_tx.loc[mask, TYPE_COL]
    if len(types) > 0:
        type_by_account[acc] = types.mode().iloc[0]

le_type = LabelEncoder()
all_types = sorted(set(type_by_account.values()))
le_type.fit(all_types)
print(f"\nتعداد انواع پول‌شویی در سطح اکانت: {len(all_types)}")

type_label = pd.Series(-1, index=all_accounts)
for acc, t in type_by_account.items():
    type_label.loc[acc] = le_type.transform([t])[0]

y_binary = torch.tensor(binary_label.loc[all_accounts].values, dtype=torch.long)
y_type = torch.tensor(type_label.loc[all_accounts].values, dtype=torch.long)

print(f"\nتعداد اکانت illicit: {(y_binary == 1).sum().item()}   "
      f"از {len(all_accounts)}   نسبت: {(y_binary == 1).float().mean().item():.4f}")


print(f"\nتقسیم اکانت بر اساس آخرین فعالیت:  train={train_mask.sum().item()}   "
      f"val={val_mask.sum().item()}   test={test_mask.sum().item()}")

labels_np = y_binary.numpy()
print(f"نسبت illicit در هر تقسیم:  "
      f"train={labels_np[train_mask.numpy()].mean():.4f}   "
      f"val={labels_np[val_mask.numpy()].mean():.4f}   "
      f"test={labels_np[test_mask.numpy()].mean():.4f}")

cold_start_np = cold_start.reindex(all_accounts).values
print(f"\nنرخ cold-start در هر تقسیم:  "
      f"train={cold_start_np[train_mask.numpy()].mean():.4f}   "
      f"val={cold_start_np[val_mask.numpy()].mean():.4f}   "
      f"test={cold_start_np[test_mask.numpy()].mean():.4f}")
print("(این‌بار انتظار می‌ره نرخ val/test خیلی پایین‌تر از صددرصد نسخه اول باشه،")
print(" ولی صفر هم نباشه، چون اکانت‌های واقعاً جدید هنوز باید cold-start بمونن.)")


# ============================================================
# ۷. ذخیره
# ============================================================
torch.save({
    "x": x,
    "edge_index": edge_index,
    "y_binary": y_binary,
    "y_type": y_type,
    "train_mask": train_mask,
    "val_mask": val_mask,
    "test_mask": test_mask,
    "feature_cols": feature_cols,
    "num_types": len(all_types),
    "account_ids": list(all_accounts),
    "cold_start_mask": torch.tensor(cold_start_np),
    "has_temporal_cutoff": HAS_TEMPORAL_CUTOFF,
}, f"{OUTPUT_PREFIX}.pt")

joblib.dump(scaler, "samld_scaler_v3.pkl")
print("scaler هم ذخیره شد در samld_scaler_v3.pkl")

print(f"\nهمه چیز ذخیره شد در {OUTPUT_PREFIX}.pt")
print("اسم فایل عمداً v3 است. step19 تا step23 باید DATA_PATH را به این فایل")
print("تغییر بدهند و از نو اجرا بشوند.")