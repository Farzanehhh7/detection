"""
فاز سه — بررسی توزیع نوع پول‌شویی قبل از طراحی سر multi-task
=====================================================================
قبل از ساخت سر ۲۸/۱۷ کلاسه، باید بدونیم هر نوع چند نمونه illicit
داره، جداگانه در train و val و test. اگه بعضی نوع‌ها فقط چند نمونه
دارن، باید قبل از مدل‌سازی تصمیم بگیریم که ادغامشون کنیم یا نه.
"""

import torch
from collections import Counter

# DATA_PATH = "samld_processed.pt"

DATA_PATH = "samld_processed_v3.pt"

data_dict = torch.load(DATA_PATH, weights_only=False)
y_type = data_dict["y_type"]
train_mask = data_dict["train_mask"]
val_mask = data_dict["val_mask"]
test_mask = data_dict["test_mask"]
num_types = data_dict["num_types"]

print(f"تعداد کل انواع پول‌شویی: {num_types}\n")

for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
    types_in_split = y_type[mask]
    illicit_types = types_in_split[types_in_split != -1].tolist()
    counts = Counter(illicit_types)
    print(f"=== {name}، تعداد کل illicit با نوع مشخص: {len(illicit_types)} ===")
    for type_id in range(num_types):
        c = counts.get(type_id, 0)
        flag = "  ⚠ کمتر از ۵ نمونه" if c < 5 else ""
        print(f"  نوع {type_id:2d}: {c:4d} نمونه{flag}")
    print()

print("قاعده تصمیم پیشنهادی: هر نوعی که در train کمتر از ۱۰ نمونه داره،")
print("عملاً قابل یادگیری نیست؛ باید یا با نزدیک‌ترین نوع مشابه ادغام بشه")
print("یا در یک کلاس «سایر» جمع بشه، قبل از آموزش سر type.")