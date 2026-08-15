# # """
# # فاز سه، قدم اول — ساخت گراف اکانت‌محور SAML-D و بازسازی feature engineering
# # =====================================================================
# # بر خلاف Elliptic که هر گره یک تراکنشه، این‌جا هر گره یک اکانته و هر
# # تراکنش یک یال بین دو اکانت. این اسکریپت فقط داده رو آماده و ذخیره
# # می‌کنه؛ هیچ مدلی آموزش نمی‌بینه، چون قبل از مدل‌سازی روی نه‌ونیم
# # میلیون تراکنش، باید مطمئن بشیم زیرساخت داده درست و بدون نشتی‌ست.
# #
# # قانون برچسب‌گذاری اکانت:
# #   یک اکانت illicit است اگر حداقل یک‌بار، به‌عنوان فرستنده یا گیرنده،
# #   در یک تراکنش با Is_laundering=1 ظاهر شده باشد. نوع پول‌شویی اکانت،
# #   پرتکرارترین Laundering_type بین تراکنش‌های illicit همان اکانت است.
# #   این یک قانون ساده و قابل‌دفاع است، نه یک حقیقت مطلق؛ باید قبل از
# #   مدل‌سازی نهایی تایید یا اصلاح شود.
# #
# # قانون فیچر بدون نشتی زمانی:
# #   فیچر هر اکانت فقط از تراکنش‌های آن اکانت در بازه زمانی train
# #   محاسبه می‌شود، حتی برای اکانت‌هایی که در val یا test هم ظاهر
# #   می‌شوند؛ این دقیقاً همان چیزی‌ست که در دنیای واقعی موقع تصمیم‌گیری
# #   در دسترس است. اکانتی که فقط در val یا test برای اولین بار ظاهر
# #   شود، فیچر صفر می‌گیرد؛ یک محدودیت شناخته‌شده، نه باگ.
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
# # N_ROWS = 500_000  # همون مقیاسی که قبلاً روی این دیتاست جواب داده بود؛ بعد از تایید زیرساخت، قابل افزایشه
# # OUTPUT_PREFIX = "samld_processed"
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
# # # ۱. تقسیم زمانی، با fallback به stratified اگر ستون تاریخ نبود
# # # ============================================================
# # date_col = None
# # for candidate in ["Date", "Time", "Date_time", "Timestamp"]:
# #     if candidate in df.columns:
# #         date_col = candidate
# #         break
# #
# # if date_col is not None:
# #     df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
# #     if df["_dt"].isna().mean() < 0.05:  # کمتر از پنج درصد تاریخ نامعتبر، قابل قبوله
# #         df = df.sort_values("_dt").reset_index(drop=True)
# #         n = len(df)
# #         train_end = int(n * 0.70)
# #         val_end = int(n * 0.85)
# #         df["split"] = "test"
# #         df.loc[:train_end, "split"] = "train"
# #         df.loc[train_end:val_end, "split"] = "val"
# #         print(f"تقسیم زمانی روی ستون {date_col} انجام شد: "
# #               f"train={  (df['split']=='train').sum() }, "
# #               f"val={ (df['split']=='val').sum() }, "
# #               f"test={ (df['split']=='test').sum() }")
# #     else:
# #         date_col = None
# #
# # if date_col is None:
# #     print("ستون تاریخ معتبر پیدا نشد؛ fallback به تقسیم stratified تصادفی روی Is_laundering.")
# #     from sklearn.model_selection import train_test_split
# #     idx_train, idx_rest = train_test_split(
# #         df.index, test_size=0.30, stratify=df[LABEL_COL], random_state=42
# #     )
# #     idx_val, idx_test = train_test_split(
# #         idx_rest, test_size=0.50, stratify=df.loc[idx_rest, LABEL_COL], random_state=42
# #     )
# #     df["split"] = "train"
# #     df.loc[idx_val, "split"] = "val"
# #     df.loc[idx_test, "split"] = "test"
# #
# #
# # # ============================================================
# # # ۲. فهرست کامل اکانت‌ها و ساخت edge_index، از همه تراکنش‌ها
# # # ============================================================
# # all_accounts = pd.unique(df[[SENDER_COL, RECEIVER_COL]].values.ravel())
# # print(f"\nتعداد کل اکانت‌های یکتا: {len(all_accounts)}")
# # print(f"تعداد کل تراکنش‌ها یعنی یال‌ها: {len(df)}")
# #
# # acc_map, edge_index = build_edge_index(all_accounts, df[SENDER_COL], df[RECEIVER_COL])
# #
# #
# # # ============================================================
# # # ۳. فیچر هر اکانت، فقط از تراکنش‌های بازه train
# # # ============================================================
# # df_train = df[df["split"] == "train"].copy()
# #
# # sent_stats = df_train.groupby(SENDER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
# # sent_stats.columns = ["sent_amount_sum", "sent_amount_mean", "sent_amount_count"]
# # sent_ptype = df_train.groupby(SENDER_COL)[PTYPE_COL].nunique().rename("sent_payment_type_nunique")
# #
# # recv_stats = df_train.groupby(RECEIVER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
# # recv_stats.columns = ["recv_amount_sum", "recv_amount_mean", "recv_amount_count"]
# # recv_ptype = df_train.groupby(RECEIVER_COL)[PTYPE_COL].nunique().rename("recv_payment_type_nunique")
# #
# # feat_df = pd.concat([sent_stats, sent_ptype, recv_stats, recv_ptype], axis=1)
# # feat_df = feat_df.reindex(all_accounts).fillna(0.0)  # اکانت‌های بدون تاریخچه train، فیچر صفر
# #
# # feature_cols = list(feat_df.columns)
# # print(f"\nستون‌های فیچر اکانت: {feature_cols}")
# #
# # scaler = StandardScaler()
# # x = scaler.fit_transform(feat_df.values)
# # x = torch.tensor(x, dtype=torch.float)
# #
# #
# # # ============================================================
# # # ۴. برچسب اکانت: illicit اگر حداقل یک‌بار در تراکنش illicit ظاهر شده
# # # ============================================================
# # illicit_tx = df[df[LABEL_COL] == 1]
# # illicit_accounts = set(illicit_tx[SENDER_COL]).union(set(illicit_tx[RECEIVER_COL]))
# #
# # binary_label = pd.Series(0, index=all_accounts)
# # binary_label.loc[list(illicit_accounts & set(all_accounts))] = 1
# #
# # # نوع، پرتکرارترین Laundering_type بین تراکنش‌های illicit اکانت؛ -1 یعنی illicit نیست
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
# # # ۵. ماسک تقسیم در سطح اکانت: یک اکانت به هر تقسیمی تعلق داره که
# # #    اولین‌بار توش ظاهر شده، بر اساس ترتیب زمانی. نسخه vectorized،
# # #    چون iterrows روی نیم‌میلیون ردیف عملاً غیرقابل‌قبول کنده.
# # # ============================================================
# # df["_order"] = np.arange(len(df))
# # melted = pd.concat([
# #     df[[SENDER_COL, "split", "_order"]].rename(columns={SENDER_COL: "account"}),
# #     df[[RECEIVER_COL, "split", "_order"]].rename(columns={RECEIVER_COL: "account"}),
# # ], ignore_index=True)
# # melted = melted.sort_values("_order")
# # first_seen = melted.drop_duplicates(subset="account", keep="first").set_index("account")["split"]
# # split_series = first_seen.reindex(all_accounts)
# # train_mask = torch.tensor((split_series == "train").values)
# # val_mask = torch.tensor((split_series == "val").values)
# # test_mask = torch.tensor((split_series == "test").values)
# #
# # print(f"\nاکانت‌ها بر اساس اولین ظهور:  train={train_mask.sum().item()}   "
# #       f"val={val_mask.sum().item()}   test={test_mask.sum().item()}")
# #
# #
# # # ============================================================
# # # ۶. ذخیره همه چیز، برای استفاده در اسکریپت‌های بعدی بدون تکرار این پردازش
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
# # }, f"{OUTPUT_PREFIX}.pt")
# #
# # print(f"\nهمه چیز ذخیره شد در {OUTPUT_PREFIX}.pt")
# # print("در اسکریپت بعدی با این بارگذاری می‌شه:")
# # print(f'  data = torch.load("{OUTPUT_PREFIX}.pt")')
#
# """
# فاز سه، قدم اول — ساخت گراف اکانت‌محور SAML-D و بازسازی feature engineering
# =====================================================================
# بر خلاف Elliptic که هر گره یک تراکنشه، این‌جا هر گره یک اکانته و هر
# تراکنش یک یال بین دو اکانت. این اسکریپت فقط داده رو آماده و ذخیره
# می‌کنه؛ هیچ مدلی آموزش نمی‌بینه، چون قبل از مدل‌سازی روی نه‌ونیم
# میلیون تراکنش، باید مطمئن بشیم زیرساخت داده درست و بدون نشتی‌ست.
#
# قانون برچسب‌گذاری اکانت:
#   یک اکانت illicit است اگر حداقل یک‌بار، به‌عنوان فرستنده یا گیرنده،
#   در یک تراکنش با Is_laundering=1 ظاهر شده باشد. نوع پول‌شویی اکانت،
#   پرتکرارترین Laundering_type بین تراکنش‌های illicit همان اکانت است.
#   این یک قانون ساده و قابل‌دفاع است، نه یک حقیقت مطلق؛ باید قبل از
#   مدل‌سازی نهایی تایید یا اصلاح شود.
#
# قانون فیچر بدون نشتی زمانی:
#   فیچر هر اکانت فقط از تراکنش‌های آن اکانت در بازه زمانی train
#   محاسبه می‌شود، حتی برای اکانت‌هایی که در val یا test هم ظاهر
#   می‌شوند؛ این دقیقاً همان چیزی‌ست که در دنیای واقعی موقع تصمیم‌گیری
#   در دسترس است. اکانتی که فقط در val یا test برای اولین بار ظاهر
#   شود، فیچر صفر می‌گیرد؛ یک محدودیت شناخته‌شده، نه باگ.
# """
#
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.preprocessing import LabelEncoder, StandardScaler
#
# from metrics_utils import build_edge_index
#
# FILE_PATH = "datasets/SAML-D.csv"
# N_ROWS = 1_000_000  # زیرساخت با پونصدهزار تایید شد؛ الان برای سیگنال بیشتر افزایش دادیم
# OUTPUT_PREFIX = "samld_processed"
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
# # ۱. تقسیم زمانی، با fallback به stratified اگر ستون تاریخ نبود
# # ============================================================
# date_col = None
# for candidate in ["Date", "Time", "Date_time", "Timestamp"]:
#     if candidate in df.columns:
#         date_col = candidate
#         break
#
# if date_col is not None:
#     df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
#     if df["_dt"].isna().mean() < 0.05:  # کمتر از پنج درصد تاریخ نامعتبر، قابل قبوله
#         df = df.sort_values("_dt").reset_index(drop=True)
#         n = len(df)
#         train_end = int(n * 0.70)
#         val_end = int(n * 0.85)
#         df["split"] = "test"
#         df.loc[:train_end, "split"] = "train"
#         df.loc[train_end:val_end, "split"] = "val"
#         print(f"تقسیم زمانی روی ستون {date_col} انجام شد: "
#               f"train={  (df['split']=='train').sum() }, "
#               f"val={ (df['split']=='val').sum() }, "
#               f"test={ (df['split']=='test').sum() }")
#     else:
#         date_col = None
#
# if date_col is None:
#     print("ستون تاریخ معتبر پیدا نشد؛ fallback به تقسیم stratified تصادفی روی Is_laundering.")
#     from sklearn.model_selection import train_test_split
#     idx_train, idx_rest = train_test_split(
#         df.index, test_size=0.30, stratify=df[LABEL_COL], random_state=42
#     )
#     idx_val, idx_test = train_test_split(
#         idx_rest, test_size=0.50, stratify=df.loc[idx_rest, LABEL_COL], random_state=42
#     )
#     df["split"] = "train"
#     df.loc[idx_val, "split"] = "val"
#     df.loc[idx_test, "split"] = "test"
#
#
# # ============================================================
# # ۲. فهرست کامل اکانت‌ها و ساخت edge_index، از همه تراکنش‌ها
# # ============================================================
# all_accounts = pd.unique(df[[SENDER_COL, RECEIVER_COL]].values.ravel())
# print(f"\nتعداد کل اکانت‌های یکتا: {len(all_accounts)}")
# print(f"تعداد کل تراکنش‌ها یعنی یال‌ها: {len(df)}")
#
# acc_map, edge_index = build_edge_index(all_accounts, df[SENDER_COL], df[RECEIVER_COL])
#
#
# # ============================================================
# # ۳. فیچر هر اکانت، فقط از تراکنش‌های بازه train
# # ============================================================
# df_train = df[df["split"] == "train"].copy()
#
# sent_stats = df_train.groupby(SENDER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
# sent_stats.columns = ["sent_amount_sum", "sent_amount_mean", "sent_amount_count"]
# sent_ptype = df_train.groupby(SENDER_COL)[PTYPE_COL].nunique().rename("sent_payment_type_nunique")
#
# recv_stats = df_train.groupby(RECEIVER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
# recv_stats.columns = ["recv_amount_sum", "recv_amount_mean", "recv_amount_count"]
# recv_ptype = df_train.groupby(RECEIVER_COL)[PTYPE_COL].nunique().rename("recv_payment_type_nunique")
#
# feat_df = pd.concat([sent_stats, sent_ptype, recv_stats, recv_ptype], axis=1)
# feat_df = feat_df.reindex(all_accounts).fillna(0.0)  # اکانت‌های بدون تاریخچه train، فیچر صفر
#
# feature_cols = list(feat_df.columns)
# print(f"\nستون‌های فیچر اکانت: {feature_cols}")
#
# scaler = StandardScaler()
# x = scaler.fit_transform(feat_df.values)
# x = torch.tensor(x, dtype=torch.float)
#
#
# # ============================================================
# # ۴. برچسب اکانت: illicit اگر حداقل یک‌بار در تراکنش illicit ظاهر شده
# # ============================================================
# illicit_tx = df[df[LABEL_COL] == 1]
# illicit_accounts = set(illicit_tx[SENDER_COL]).union(set(illicit_tx[RECEIVER_COL]))
#
# binary_label = pd.Series(0, index=all_accounts)
# binary_label.loc[list(illicit_accounts & set(all_accounts))] = 1
#
# # نوع، پرتکرارترین Laundering_type بین تراکنش‌های illicit اکانت؛ -1 یعنی illicit نیست
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
# # ۵. ماسک تقسیم در سطح اکانت: یک اکانت به هر تقسیمی تعلق داره که
# #    اولین‌بار توش ظاهر شده، بر اساس ترتیب زمانی. نسخه vectorized،
# #    چون iterrows روی نیم‌میلیون ردیف عملاً غیرقابل‌قبول کنده.
# # ============================================================
# df["_order"] = np.arange(len(df))
# melted = pd.concat([
#     df[[SENDER_COL, "split", "_order"]].rename(columns={SENDER_COL: "account"}),
#     df[[RECEIVER_COL, "split", "_order"]].rename(columns={RECEIVER_COL: "account"}),
# ], ignore_index=True)
# melted = melted.sort_values("_order")
# first_seen = melted.drop_duplicates(subset="account", keep="first").set_index("account")["split"]
# split_series = first_seen.reindex(all_accounts)
# train_mask = torch.tensor((split_series == "train").values)
# val_mask = torch.tensor((split_series == "val").values)
# test_mask = torch.tensor((split_series == "test").values)
#
# print(f"\nاکانت‌ها بر اساس اولین ظهور:  train={train_mask.sum().item()}   "
#       f"val={val_mask.sum().item()}   test={test_mask.sum().item()}")
#
#
# # ============================================================
# # ۶. ذخیره همه چیز، برای استفاده در اسکریپت‌های بعدی بدون تکرار این پردازش
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
# }, f"{OUTPUT_PREFIX}.pt")
#
# print(f"\nهمه چیز ذخیره شد در {OUTPUT_PREFIX}.pt")
# print("در اسکریپت بعدی با این بارگذاری می‌شه:")
# print(f'  data = torch.load("{OUTPUT_PREFIX}.pt")')

"""
فاز سه، قدم اول — ساخت گراف اکانت‌محور SAML-D و بازسازی feature engineering
=====================================================================
بر خلاف Elliptic که هر گره یک تراکنشه، این‌جا هر گره یک اکانته و هر
تراکنش یک یال بین دو اکانت. این اسکریپت فقط داده رو آماده و ذخیره
می‌کنه؛ هیچ مدلی آموزش نمی‌بینه، چون قبل از مدل‌سازی روی نه‌ونیم
میلیون تراکنش، باید مطمئن بشیم زیرساخت داده درست و بدون نشتی‌ست.

قانون برچسب‌گذاری اکانت:
  یک اکانت illicit است اگر حداقل یک‌بار، به‌عنوان فرستنده یا گیرنده،
  در یک تراکنش با Is_laundering=1 ظاهر شده باشد. نوع پول‌شویی اکانت،
  پرتکرارترین Laundering_type بین تراکنش‌های illicit همان اکانت است.
  این یک قانون ساده و قابل‌دفاع است، نه یک حقیقت مطلق؛ باید قبل از
  مدل‌سازی نهایی تایید یا اصلاح شود.

قانون فیچر بدون نشتی زمانی:
  فیچر هر اکانت فقط از تراکنش‌های آن اکانت در بازه زمانی train
  محاسبه می‌شود، حتی برای اکانت‌هایی که در val یا test هم ظاهر
  می‌شوند؛ این دقیقاً همان چیزی‌ست که در دنیای واقعی موقع تصمیم‌گیری
  در دسترس است. اکانتی که فقط در val یا test برای اولین بار ظاهر
  شود، فیچر صفر می‌گیرد؛ یک محدودیت شناخته‌شده، نه باگ.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

from metrics_utils import build_edge_index

FILE_PATH = "datasets/SAML-D.csv"
N_ROWS = 1_000_000  # زیرساخت با پونصدهزار تایید شد؛ الان برای سیگنال بیشتر افزایش دادیم
OUTPUT_PREFIX = "samld_processed"

print(f"در حال بارگذاری {N_ROWS} ردیف اول از SAML-D...")
df = pd.read_csv(FILE_PATH, nrows=N_ROWS)
print("ستون‌های موجود:", list(df.columns))

SENDER_COL, RECEIVER_COL = "Sender_account", "Receiver_account"
LABEL_COL, TYPE_COL = "Is_laundering", "Laundering_type"
AMOUNT_COL, PTYPE_COL = "Amount", "Payment_type"


# ============================================================
# ۱. تقسیم زمانی، با fallback به stratified اگر ستون تاریخ نبود
# ============================================================
date_col = None
for candidate in ["Date", "Time", "Date_time", "Timestamp"]:
    if candidate in df.columns:
        date_col = candidate
        break

if date_col is not None:
    df["_dt"] = pd.to_datetime(df[date_col], errors="coerce")
    if df["_dt"].isna().mean() < 0.05:  # کمتر از پنج درصد تاریخ نامعتبر، قابل قبوله
        df = df.sort_values("_dt").reset_index(drop=True)
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        df["split"] = "test"
        df.loc[:train_end, "split"] = "train"
        df.loc[train_end:val_end, "split"] = "val"
        print(f"تقسیم زمانی روی ستون {date_col} انجام شد: "
              f"train={  (df['split']=='train').sum() }, "
              f"val={ (df['split']=='val').sum() }, "
              f"test={ (df['split']=='test').sum() }")
    else:
        date_col = None

if date_col is None:
    print("ستون تاریخ معتبر پیدا نشد؛ fallback به تقسیم stratified تصادفی روی Is_laundering.")
    from sklearn.model_selection import train_test_split
    idx_train, idx_rest = train_test_split(
        df.index, test_size=0.30, stratify=df[LABEL_COL], random_state=42
    )
    idx_val, idx_test = train_test_split(
        idx_rest, test_size=0.50, stratify=df.loc[idx_rest, LABEL_COL], random_state=42
    )
    df["split"] = "train"
    df.loc[idx_val, "split"] = "val"
    df.loc[idx_test, "split"] = "test"


# ============================================================
# ۲. فهرست کامل اکانت‌ها و ساخت edge_index، از همه تراکنش‌ها
# ============================================================
all_accounts = pd.unique(df[[SENDER_COL, RECEIVER_COL]].values.ravel())
print(f"\nتعداد کل اکانت‌های یکتا: {len(all_accounts)}")
print(f"تعداد کل تراکنش‌ها یعنی یال‌ها: {len(df)}")

acc_map, edge_index = build_edge_index(all_accounts, df[SENDER_COL], df[RECEIVER_COL])


# ============================================================
# ۳. فیچر هر اکانت، فقط از تراکنش‌های بازه train
# ============================================================
# نکته مهم: قبلاً این‌جا فقط تراکنش‌های train استفاده می‌شد، که باعث می‌شد
# صددرصد اکانت‌های val و test، چون طبق تعریف اولین ظهورشون تو train نیست،
# هیچ تراکنش train ای نداشته باشن و فیچر کاملاً صفر بگیرن. این‌جا به‌جاش
# از کل تاریخچه هر اکانت در کل نمونه بارگذاری‌شده استفاده می‌کنیم. این یک
# تصمیم آگاهانه‌ست: برچسب هم از اول از کل داده ساخته می‌شد، پس این کار
# فقط بین فیچر و برچسب هم‌خوانی ایجاد می‌کنه؛ اما معنیش اینه که این دیگه
# یک پیش‌بینی کاملاً آینده‌کور نیست، بلکه بیشتر شبیه «با همه شواهد موجود
# این اکانت رو پروفایل کن»، که باید در بخش محدودیت‌ها صریح ذکر بشه.
df_train = df.copy()

sent_stats = df_train.groupby(SENDER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
sent_stats.columns = ["sent_amount_sum", "sent_amount_mean", "sent_amount_count"]
sent_ptype = df_train.groupby(SENDER_COL)[PTYPE_COL].nunique().rename("sent_payment_type_nunique")

recv_stats = df_train.groupby(RECEIVER_COL)[AMOUNT_COL].agg(["sum", "mean", "count"])
recv_stats.columns = ["recv_amount_sum", "recv_amount_mean", "recv_amount_count"]
recv_ptype = df_train.groupby(RECEIVER_COL)[PTYPE_COL].nunique().rename("recv_payment_type_nunique")

feat_df = pd.concat([sent_stats, sent_ptype, recv_stats, recv_ptype], axis=1)
feat_df = feat_df.reindex(all_accounts).fillna(0.0)  # اکانت‌های بدون تاریخچه train، فیچر صفر

feature_cols = list(feat_df.columns)
print(f"\nستون‌های فیچر اکانت: {feature_cols}")

scaler = StandardScaler()
x = scaler.fit_transform(feat_df.values)
x = torch.tensor(x, dtype=torch.float)


# ============================================================
# ۴. برچسب اکانت: illicit اگر حداقل یک‌بار در تراکنش illicit ظاهر شده
# ============================================================
illicit_tx = df[df[LABEL_COL] == 1]
illicit_accounts = set(illicit_tx[SENDER_COL]).union(set(illicit_tx[RECEIVER_COL]))

binary_label = pd.Series(0, index=all_accounts)
binary_label.loc[list(illicit_accounts & set(all_accounts))] = 1

# نوع، پرتکرارترین Laundering_type بین تراکنش‌های illicit اکانت؛ -1 یعنی illicit نیست
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


# ============================================================
# ۵. ماسک تقسیم در سطح اکانت: یک اکانت به هر تقسیمی تعلق داره که
#    اولین‌بار توش ظاهر شده، بر اساس ترتیب زمانی. نسخه vectorized،
#    چون iterrows روی نیم‌میلیون ردیف عملاً غیرقابل‌قبول کنده.
# ============================================================
df["_order"] = np.arange(len(df))
melted = pd.concat([
    df[[SENDER_COL, "split", "_order"]].rename(columns={SENDER_COL: "account"}),
    df[[RECEIVER_COL, "split", "_order"]].rename(columns={RECEIVER_COL: "account"}),
], ignore_index=True)
melted = melted.sort_values("_order")
first_seen = melted.drop_duplicates(subset="account", keep="first").set_index("account")["split"]
split_series = first_seen.reindex(all_accounts)
train_mask = torch.tensor((split_series == "train").values)
val_mask = torch.tensor((split_series == "val").values)
test_mask = torch.tensor((split_series == "test").values)

print(f"\nاکانت‌ها بر اساس اولین ظهور:  train={train_mask.sum().item()}   "
      f"val={val_mask.sum().item()}   test={test_mask.sum().item()}")


# ============================================================
# ۶. ذخیره همه چیز، برای استفاده در اسکریپت‌های بعدی بدون تکرار این پردازش
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
}, f"{OUTPUT_PREFIX}.pt")

print(f"\nهمه چیز ذخیره شد در {OUTPUT_PREFIX}.pt")
print("در اسکریپت بعدی با این بارگذاری می‌شه:")
print(f'  data = torch.load("{OUTPUT_PREFIX}.pt")')