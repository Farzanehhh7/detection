"""
باسلاین جدولی SAML-D، بدون هیچ گراف — دقیقاً همون کاری که باید از اول می‌کردیم
=====================================================================
همون فیچرهای هشت‌ستونی و همون برچسب و همون تقسیم train/val/test که
در step18 ساختیم، این‌بار فقط با Logistic Regression و Random Forest،
بدون هیچ SAGEConv یا نمونه‌برداری همسایگی. چند ثانیه طول می‌کشه.

این نقطه مرجعیه که تا الان نداشتیم: اگه GraphSAGE از این دو بهتر
نشه، دقیقاً همون داستان Elliptic تکرار می‌شه و باید صادقانه بپذیریم
که ارزش افزوده گراف این‌جا هم هنوز اثبات نشده؛ اگه بهتر شد، خیالمون
راحت می‌شه که این مسیر ارزش وقت‌گذاشتن بیشتر رو داره.
"""

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from metrics_utils import evaluate_binary, find_best_threshold

# DATA_PATH = "samld_processed.pt"
DATA_PATH = "samld_processed_v3.pt"
print("در حال بارگذاری داده پردازش‌شده SAML-D...")
data_dict = torch.load(DATA_PATH, weights_only=False)
x = data_dict["x"].numpy()
y = data_dict["y_binary"].numpy()
train_mask = data_dict["train_mask"].numpy()
val_mask = data_dict["val_mask"].numpy()
test_mask = data_dict["test_mask"].numpy()

X_train, y_train = x[train_mask], y[train_mask]
X_val, y_val = x[val_mask], y[val_mask]
X_test, y_test = x[test_mask], y[test_mask]

print(f"Train: {len(X_train)}   Val: {len(X_val)}   Test: {len(X_test)}")
print(f"illicit در train: {y_train.sum()}   در val: {y_val.sum()}   در test: {y_test.sum()}")


def run_and_report(name, model):
    model.fit(X_train, y_train)
    probs_val = model.predict_proba(X_val)[:, 1]
    best_t, _ = find_best_threshold(y_val, probs_val)

    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= best_t).astype(int)
    evaluate_binary(name, y_test, preds_test, probs_test)
    print(f"threshold انتخابی: {best_t:.2f}")


print("\n=== Logistic Regression، بدون گراف ===")
run_and_report("LR بدون گراف", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))

print("\n=== Random Forest، بدون گراف ===")
run_and_report("RF بدون گراف", RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
))

print("\n\nبرای مقایسه، GraphSAGE با نمونه‌برداری، پونصدهزار ردیف:")
print("  raw_weight     F1=0.2273  PR-AUC=0.1017")
print("  capped_weight  F1=0.2857  PR-AUC=0.0930")