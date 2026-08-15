# # import random
# #
# # import numpy as np
# # import pandas as pd
# # import torch
# # from sklearn.metrics import (
# #     average_precision_score, f1_score, matthews_corrcoef,
# #     precision_score, recall_score, roc_auc_score,
# # )
# #
# #
# # def set_seed(seed=42):
# #     random.seed(seed)
# #     np.random.seed(seed)
# #     torch.manual_seed(seed)
# #     torch.cuda.manual_seed_all(seed)
# #
# #
# # def evaluate_binary(name, y_true, y_pred, y_prob, verbose=True):
# #     results = {
# #         "model": name,
# #         "AUC": roc_auc_score(y_true, y_prob),
# #         "PR-AUC": average_precision_score(y_true, y_prob),
# #         "F1": f1_score(y_true, y_pred),
# #         "Precision": precision_score(y_true, y_pred),
# #         "Recall": recall_score(y_true, y_pred),
# #         "MCC": matthews_corrcoef(y_true, y_pred),
# #     }
# #     if verbose:
# #         print(f"\n--- {name} ---")
# #         for k, v in results.items():
# #             if k != "model":
# #                 print(f"{k:12s}: {v:.4f}")
# #     return results
# #
# #
# # def evaluate_gnn(name, model, x, edge_index, y, mask, two_class_softmax=False):
# #     model.eval()
# #     with torch.no_grad():
# #         out = model(x, edge_index)
# #         if two_class_softmax:
# #             probs = torch.softmax(out, dim=1)[mask, 1].cpu().numpy()
# #             preds = out[mask].argmax(dim=1).cpu().numpy()
# #         else:
# #             probs = torch.sigmoid(out[mask]).cpu().numpy()
# #             preds = (probs >= 0.5).astype(int)
# #         y_true = y[mask].cpu().numpy()
# #     return evaluate_binary(name, y_true, preds, probs)
# #
# #
# # # ============================================================
# # # فاز صفر: زیرساخت ارزیابی
# # # سه تابع زیر رو اضافه کردیم چون همه‌ی نتایج فعلی تک-seed و
# # # تک-threshold بودن. از این به بعد هر تجربه باید از این سه رد بشه.
# # # ============================================================
# #
# # def get_temporal_split_masks(time_step, y, train_end=27, val_end=34, device=None):
# #     """
# #     ورودی time_step باید همون ستون خام ۱ تا ۴۹ الیپتیک باشه، نه صفر-پایه.
# #     منطق: ۱ تا train_end برای train واقعی، train_end+1 تا val_end برای
# #     validation، بقیه یعنی بالای val_end برای test. چون test اصلی پروژه
# #     شما همیشه بالای ۳۴ بوده، val_end رو هم روی ۳۴ نگه داشتیم تا test
# #     دست‌نخورده بمونه و فقط تکه‌ی قدیمی train به train/val تقسیم بشه.
# #     """
# #     if not torch.is_tensor(time_step):
# #         time_step = torch.tensor(time_step)
# #     if device is not None:
# #         time_step = time_step.to(device)
# #         y = y.to(device)
# #
# #     labeled = y != -1
# #     train_mask = labeled & (time_step <= train_end)
# #     val_mask = labeled & (time_step > train_end) & (time_step <= val_end)
# #     test_mask = labeled & (time_step > val_end)
# #     return train_mask, val_mask, test_mask
# #
# #
# # def find_best_threshold(y_true, y_prob, thresholds=None):
# #     """
# #     این تابع رو فقط روی validation صدا بزن، هرگز روی test. خروجی
# #     threshold بهینه برای F1 کلاس illicit و F1 متناظرش هست.
# #     """
# #     if thresholds is None:
# #         thresholds = np.arange(0.05, 0.96, 0.01)
# #     best_t, best_f1 = 0.5, -1.0
# #     for t in thresholds:
# #         preds = (y_prob >= t).astype(int)
# #         f1 = f1_score(y_true, preds, zero_division=0)
# #         if f1 > best_f1:
# #             best_t, best_f1 = float(t), f1
# #     return best_t, best_f1
# #
# #
# # def run_multi_seed(run_one_seed_fn, seeds=(42, 1, 7, 123, 2024), name="model", verbose=True):
# #     """
# #     run_one_seed_fn باید تابعی با امضای run_one_seed_fn(seed) -> dict
# #     باشه که یک دیکشنری متریک مثل خروجی evaluate_binary برمی‌گردونه.
# #     خروجی این تابع میانگین و انحراف معیار روی همه seed هاست، به علاوه
# #     جدول کامل تک‌تک اجراها برای شفافیت.
# #     """
# #     rows = []
# #     for seed in seeds:
# #         set_seed(seed)
# #         metrics = run_one_seed_fn(seed)
# #         metrics["seed"] = seed
# #         rows.append(metrics)
# #
# #     df = pd.DataFrame(rows)
# #     numeric_cols = [c for c in df.columns if c not in ("model", "seed")]
# #     summary = df[numeric_cols].agg(["mean", "std"])
# #
# #     if verbose:
# #         print(f"\n=== {name} — نتیجه روی {len(seeds)} seed ===")
# #         print(df[["seed"] + numeric_cols].round(4).to_string(index=False))
# #         print("\nمیانگین ± انحراف معیار:")
# #         for c in numeric_cols:
# #             print(f"{c:12s}: {summary.loc['mean', c]:.4f} ± {summary.loc['std', c]:.4f}")
# #
# #     return df, summary
#
# import random
#
# import numpy as np
# import pandas as pd
# import torch
# from sklearn.metrics import (
#     average_precision_score, f1_score, matthews_corrcoef,
#     precision_score, recall_score, roc_auc_score,
# )
#
#
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
#
# def evaluate_binary(name, y_true, y_pred, y_prob, verbose=True):
#     results = {
#         "model": name,
#         "AUC": roc_auc_score(y_true, y_prob),
#         "PR-AUC": average_precision_score(y_true, y_prob),
#         "F1": f1_score(y_true, y_pred),
#         "Precision": precision_score(y_true, y_pred),
#         "Recall": recall_score(y_true, y_pred),
#         "MCC": matthews_corrcoef(y_true, y_pred),
#     }
#     if verbose:
#         print(f"\n--- {name} ---")
#         for k, v in results.items():
#             if k != "model":
#                 print(f"{k:12s}: {v:.4f}")
#     return results
#
#
# def evaluate_gnn(name, model, x, edge_index, y, mask, two_class_softmax=False):
#     model.eval()
#     with torch.no_grad():
#         out = model(x, edge_index)
#         if two_class_softmax:
#             probs = torch.softmax(out, dim=1)[mask, 1].cpu().numpy()
#             preds = out[mask].argmax(dim=1).cpu().numpy()
#         else:
#             probs = torch.sigmoid(out[mask]).cpu().numpy()
#             preds = (probs >= 0.5).astype(int)
#         y_true = y[mask].cpu().numpy()
#     return evaluate_binary(name, y_true, preds, probs)
#
#
# # ============================================================
# # فاز صفر: زیرساخت ارزیابی
# # سه تابع زیر رو اضافه کردیم چون همه‌ی نتایج فعلی تک-seed و
# # تک-threshold بودن. از این به بعد هر تجربه باید از این سه رد بشه.
# # ============================================================
#
# def get_temporal_split_masks(time_step, y, train_end=27, val_end=34, device=None):
#     """
#     ورودی time_step باید همون ستون خام ۱ تا ۴۹ الیپتیک باشه، نه صفر-پایه.
#     منطق: ۱ تا train_end برای train واقعی، train_end+1 تا val_end برای
#     validation، بقیه یعنی بالای val_end برای test. چون test اصلی پروژه
#     شما همیشه بالای ۳۴ بوده، val_end رو هم روی ۳۴ نگه داشتیم تا test
#     دست‌نخورده بمونه و فقط تکه‌ی قدیمی train به train/val تقسیم بشه.
#     """
#     if not torch.is_tensor(time_step):
#         time_step = torch.tensor(time_step)
#     if device is not None:
#         time_step = time_step.to(device)
#         y = y.to(device)
#
#     labeled = y != -1
#     train_mask = labeled & (time_step <= train_end)
#     val_mask = labeled & (time_step > train_end) & (time_step <= val_end)
#     test_mask = labeled & (time_step > val_end)
#     return train_mask, val_mask, test_mask
#
#
# def find_best_threshold(y_true, y_prob, thresholds=None):
#     """
#     این تابع رو فقط روی validation صدا بزن، هرگز روی test. خروجی
#     threshold بهینه برای F1 کلاس illicit و F1 متناظرش هست.
#     """
#     if thresholds is None:
#         thresholds = np.arange(0.05, 0.96, 0.01)
#     best_t, best_f1 = 0.5, -1.0
#     for t in thresholds:
#         preds = (y_prob >= t).astype(int)
#         f1 = f1_score(y_true, preds, zero_division=0)
#         if f1 > best_f1:
#             best_t, best_f1 = float(t), f1
#     return best_t, best_f1
#
#
# def run_multi_seed(run_one_seed_fn, seeds=(42, 1, 7, 123, 2024), name="model", verbose=True):
#     """
#     run_one_seed_fn باید تابعی با امضای run_one_seed_fn(seed) -> dict
#     باشه که یک دیکشنری متریک مثل خروجی evaluate_binary برمی‌گردونه.
#     خروجی این تابع میانگین و انحراف معیار روی همه seed هاست، به علاوه
#     جدول کامل تک‌تک اجراها برای شفافیت.
#     """
#     rows = []
#     for seed in seeds:
#         set_seed(seed)
#         metrics = run_one_seed_fn(seed)
#         metrics["seed"] = seed
#         rows.append(metrics)
#
#     df = pd.DataFrame(rows)
#     numeric_cols = [c for c in df.columns if c not in ("model", "seed")]
#     summary = df[numeric_cols].agg(["mean", "std"])
#
#     if verbose:
#         print(f"\n=== {name} — نتیجه روی {len(seeds)} seed ===")
#         print(df[["seed"] + numeric_cols].round(4).to_string(index=False))
#         print("\nمیانگین ± انحراف معیار:")
#         for c in numeric_cols:
#             print(f"{c:12s}: {summary.loc['mean', c]:.4f} ± {summary.loc['std', c]:.4f}")
#
#     return df, summary
#
#
# # ============================================================
# # فاز دو: MAD برای over-smoothing و آزمون معناداری آماری
# # ============================================================
#
# def compute_mad_neighbors(embeddings, edge_index):
#     """
#     میانگین فاصله کسینوسی بین گره‌های همسایه، برای سنجش over-smoothing.
#     فقط روی یال‌های واقعی حساب می‌شه نه همه جفت‌ها، چون روی گراف‌های
#     بزرگ محاسبه فاصله همه‌به‌همه عملاً غیرممکنه. عدد پایین یعنی
#     گره‌های همسایه بعد از عبور از لایه‌های گراف خیلی شبیه هم شدن.
#     embeddings باید خروجی یکی از لایه‌های میانی مدل باشه، نه خروجی
#     نهایی classifier.
#     """
#     src, dst = edge_index
#     h_src = embeddings[src]
#     h_dst = embeddings[dst]
#     cos_sim = torch.nn.functional.cosine_similarity(h_src, h_dst, dim=1)
#     distance = 1 - cos_sim
#     return distance.mean().item()
#
#
# def paired_significance_test(f1_a, f1_b, name_a="A", name_b="B", verbose=True):
#     """
#     برای مقایسه دو مدل که دقیقاً روی همون seed ها اجرا شدن. ورودی دو
#     لیست هم‌ترتیب از F1 هر seed. چون داده‌ها paired هستن، از
#     paired t-test استفاده می‌کنه نه t-test مستقل، که قدرت آماری
#     بیشتری با همین تعداد کم seed می‌ده.
#     """
#     from scipy import stats
#
#     f1_a = np.array(f1_a, dtype=float)
#     f1_b = np.array(f1_b, dtype=float)
#     diff = f1_a - f1_b
#     t_stat, p_value = stats.ttest_rel(f1_a, f1_b)
#
#     if verbose:
#         print(f"\n--- {name_a} در برابر {name_b} ---")
#         print(f"میانگین {name_a}: {f1_a.mean():.4f}   میانگین {name_b}: {f1_b.mean():.4f}")
#         print(f"میانگین تفاوت: {diff.mean():.4f}   t = {t_stat:.3f}   p = {p_value:.4f}")
#         if p_value < 0.05:
#             print("تفاوت از نظر آماری معنادار است، p کمتر از 0.05.")
#         else:
#             print("تفاوت از نظر آماری معنادار نیست، ممکنه فقط نویز seed باشه.")
#
#     return t_stat, p_value



import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_binary(name, y_true, y_pred, y_prob, verbose=True):
    results = {
        "model": name,
        "AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    if verbose:
        print(f"\n--- {name} ---")
        for k, v in results.items():
            if k != "model":
                print(f"{k:12s}: {v:.4f}")
    return results


def evaluate_gnn(name, model, x, edge_index, y, mask, two_class_softmax=False):
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        if two_class_softmax:
            probs = torch.softmax(out, dim=1)[mask, 1].cpu().numpy()
            preds = out[mask].argmax(dim=1).cpu().numpy()
        else:
            probs = torch.sigmoid(out[mask]).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
        y_true = y[mask].cpu().numpy()
    return evaluate_binary(name, y_true, preds, probs)


# ============================================================
# فاز صفر: زیرساخت ارزیابی
# سه تابع زیر رو اضافه کردیم چون همه‌ی نتایج فعلی تک-seed و
# تک-threshold بودن. از این به بعد هر تجربه باید از این سه رد بشه.
# ============================================================

def get_temporal_split_masks(time_step, y, train_end=27, val_end=34, device=None):
    """
    ورودی time_step باید همون ستون خام ۱ تا ۴۹ الیپتیک باشه، نه صفر-پایه.
    منطق: ۱ تا train_end برای train واقعی، train_end+1 تا val_end برای
    validation، بقیه یعنی بالای val_end برای test. چون test اصلی پروژه
    شما همیشه بالای ۳۴ بوده، val_end رو هم روی ۳۴ نگه داشتیم تا test
    دست‌نخورده بمونه و فقط تکه‌ی قدیمی train به train/val تقسیم بشه.
    """
    if not torch.is_tensor(time_step):
        time_step = torch.tensor(time_step)
    if device is not None:
        time_step = time_step.to(device)
        y = y.to(device)

    labeled = y != -1
    train_mask = labeled & (time_step <= train_end)
    val_mask = labeled & (time_step > train_end) & (time_step <= val_end)
    test_mask = labeled & (time_step > val_end)
    return train_mask, val_mask, test_mask


def find_best_threshold(y_true, y_prob, thresholds=None):
    """
    این تابع رو فقط روی validation صدا بزن، هرگز روی test. خروجی
    threshold بهینه برای F1 کلاس illicit و F1 متناظرش هست.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def run_multi_seed(run_one_seed_fn, seeds=(42, 1, 7, 123, 2024), name="model", verbose=True):
    """
    run_one_seed_fn باید تابعی با امضای run_one_seed_fn(seed) -> dict
    باشه که یک دیکشنری متریک مثل خروجی evaluate_binary برمی‌گردونه.
    خروجی این تابع میانگین و انحراف معیار روی همه seed هاست، به علاوه
    جدول کامل تک‌تک اجراها برای شفافیت.
    """
    rows = []
    for seed in seeds:
        set_seed(seed)
        metrics = run_one_seed_fn(seed)
        metrics["seed"] = seed
        rows.append(metrics)

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c not in ("model", "seed")]
    summary = df[numeric_cols].agg(["mean", "std"])

    if verbose:
        print(f"\n=== {name} — نتیجه روی {len(seeds)} seed ===")
        print(df[["seed"] + numeric_cols].round(4).to_string(index=False))
        print("\nمیانگین ± انحراف معیار:")
        for c in numeric_cols:
            print(f"{c:12s}: {summary.loc['mean', c]:.4f} ± {summary.loc['std', c]:.4f}")

    return df, summary


# ============================================================
# فاز دو: MAD برای over-smoothing و آزمون معناداری آماری
# ============================================================

def compute_mad_neighbors(embeddings, edge_index):
    """
    میانگین فاصله کسینوسی بین گره‌های همسایه، برای سنجش over-smoothing.
    فقط روی یال‌های واقعی حساب می‌شه نه همه جفت‌ها، چون روی گراف‌های
    بزرگ محاسبه فاصله همه‌به‌همه عملاً غیرممکنه. عدد پایین یعنی
    گره‌های همسایه بعد از عبور از لایه‌های گراف خیلی شبیه هم شدن.
    embeddings باید خروجی یکی از لایه‌های میانی مدل باشه، نه خروجی
    نهایی classifier.
    """
    src, dst = edge_index
    h_src = embeddings[src]
    h_dst = embeddings[dst]
    cos_sim = torch.nn.functional.cosine_similarity(h_src, h_dst, dim=1)
    distance = 1 - cos_sim
    return distance.mean().item()


def paired_significance_test(f1_a, f1_b, name_a="A", name_b="B", verbose=True):
    """
    برای مقایسه دو مدل که دقیقاً روی همون seed ها اجرا شدن. ورودی دو
    لیست هم‌ترتیب از F1 هر seed. چون داده‌ها paired هستن، از
    paired t-test استفاده می‌کنه نه t-test مستقل، که قدرت آماری
    بیشتری با همین تعداد کم seed می‌ده.
    """
    from scipy import stats

    f1_a = np.array(f1_a, dtype=float)
    f1_b = np.array(f1_b, dtype=float)
    diff = f1_a - f1_b
    t_stat, p_value = stats.ttest_rel(f1_a, f1_b)

    if verbose:
        print(f"\n--- {name_a} در برابر {name_b} ---")
        print(f"میانگین {name_a}: {f1_a.mean():.4f}   میانگین {name_b}: {f1_b.mean():.4f}")
        print(f"میانگین تفاوت: {diff.mean():.4f}   t = {t_stat:.3f}   p = {p_value:.4f}")
        if p_value < 0.05:
            print("تفاوت از نظر آماری معنادار است، p کمتر از 0.05.")
        else:
            print("تفاوت از نظر آماری معنادار نیست، ممکنه فقط نویز seed باشه.")

    return t_stat, p_value


# ============================================================
# بسته یک از نقاط ضعف باقی‌مانده: زیرساخت آموزش و گزارش‌دهی
# ردیف‌های ۳، ۴، ۱۴، ۱۸، ۱۹ از جدول محدودیت‌ها، به‌علاوه ۱۷
# ============================================================

class EarlyStopper:
    """
    ردیف ۱۴. نظارت روی معیار validation در حین آموزش؛ اگر بعد از
    patience epoch بهبود معناداری دیده نشد، پرچم should_stop بالا
    می‌ره. بهترین state_dict مدل جداگانه نگه داشته می‌شه، مستقل از
    آخرین epoch، تا overfit روی خطای آموزش دیرهنگام نادیده گرفته
    نشه؛ دقیقاً همون مشکلی که در step3 با افت پیوسته loss تا epoch
    ۳۰۰ دیدیم.
    """

    def __init__(self, patience=20, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.best_state = None
        self.best_epoch = None
        self.counter = 0
        self.should_stop = False

    def step(self, score, model, epoch=None):
        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


def save_checkpoint(model, path, extra=None):
    """ردیف ۱۹. ذخیره وزن‌های مدل به‌علاوه هر متادیتای دلخواه، برای استفاده بعدی در inference داشبورد."""
    payload = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"checkpoint ذخیره شد در {path}")


def load_checkpoint(model, path, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    return payload


def report_at_percentile_thresholds(y_true, y_prob, percentiles=(90, 99, 99.9), verbose=True):
    """
    ردیف ۳. به‌جای یک threshold بهینه F1، می‌گه اگر فقط به‌اندازه
    ظرفیت واقعی بازرسی، مثلا فقط بالای صدک نودونه یا نودونونه امتیاز
    مشکوک‌ترین تراکنش‌ها رو پرچم بزنیم، Precision و Recall چقدر
    می‌شه. این همون رویکردیه که Dang و همکاران برای Elliptic
    توصیه کردن، چون threshold ثابت روی این دیتاست بی‌معنیه.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    rows = []
    for pct in percentiles:
        cutoff = np.percentile(y_prob, pct)
        preds = (y_prob >= cutoff).astype(int)
        row = {
            "percentile": pct,
            "threshold": cutoff,
            "n_flagged": int(preds.sum()),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
        }
        rows.append(row)
        if verbose:
            print(f"صدک {pct:5.1f} | threshold={cutoff:.4f} | flagged={row['n_flagged']:6d} | "
                  f"Precision={row['precision']:.4f} | Recall={row['recall']:.4f}")
    return pd.DataFrame(rows)


def bootstrap_test_ci(y_true, y_prob, n_iterations=100, sample_frac=0.5, threshold=0.5, seed=42, verbose=True):
    """
    ردیف ۴. برخلاف چند-seed که واریانس آموزش رو می‌سنجه، این تابع
    واریانس ناشی از خود نمونه‌گیری مجموعه test رو می‌سنجه: test رو
    n_iterations بار با جایگذاری نمونه‌برداری می‌کنه و فاصله اطمینان
    ۹۵ درصد F1 رو برمی‌گردونه. دقیقاً همون رویکرد Dang و همکاران،
    مکمل چند-seed نه جایگزینش.
    """
    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    n = len(y_true)
    scores = []
    for _ in range(n_iterations):
        idx = rng.choice(n, size=int(n * sample_frac), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        preds = (y_prob[idx] >= threshold).astype(int)
        scores.append(f1_score(y_true[idx], preds, zero_division=0))
    scores = np.array(scores)
    lower, upper = np.percentile(scores, [2.5, 97.5])
    if verbose:
        print(f"Bootstrap {len(scores)} تکرار معتبر: میانگین F1={scores.mean():.4f}   "
              f"فاصله اطمینان ۹۵٪ = [{lower:.4f}, {upper:.4f}]")
    return scores.mean(), lower, upper


def log_experiment(log_path, run_info):
    """
    ردیف ۱۸. یک ردیف شامل هایپرپارامترها و متریک‌های نهایی به یک
    فایل csv اضافه می‌کنه، فایل رو اگه نبود می‌سازه. جایگزین ساده و
    بدون وابستگی بیرونی برای ابزارهای experiment tracking.
    """
    import os
    df_row = pd.DataFrame([run_info])
    if os.path.exists(log_path):
        df_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_path, mode="w", header=True, index=False)
    print(f"لاگ شد در {log_path}: {run_info}")


def build_edge_index(node_ids, edge_src_ids, edge_dst_ids):
    """
    ردیف ۱۷. شناسه‌های خام تراکنش رو به اندیس‌های پیوسته صفر تا
    N-1 برای torch_geometric تبدیل می‌کنه. این منطق تا الان داخل
    هر اسکریپت جدا کپی می‌شد؛ جدا کردنش این‌جا هم تکرار کد رو حذف
    می‌کنه هم قابل تست‌شدن می‌کندش، چون تست واحد روی کد داخل یک
    اسکریپت اجرایی عملاً ممکن نیست.
    """
    map_id = {node_id: i for i, node_id in enumerate(node_ids)}
    src_idx = [map_id[s] for s in edge_src_ids]
    dst_idx = [map_id[d] for d in edge_dst_ids]
    return map_id, torch.tensor([src_idx, dst_idx], dtype=torch.long)