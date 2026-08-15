"""
جمع‌بندی فاز دو — آزمون معناداری روی همه مقایسه‌های مهم فاز یک و دو
=====================================================================
این اسکریپت به دیتاست یا GPU نیازی نداره؛ اعداد F1 هر seed مستقیم
از خروجی‌هایی که تا الان گرفتیم کپی شدن. همه‌شون دقیقاً روی همون
پنج seed اجرا شدن، ۴۲ و ۱ و ۷ و ۱۲۳ و ۲۰۲۴، پس مقایسه paired
کاملاً معتبره.
"""

from metrics_utils import paired_significance_test

# ترتیب seed ها همه‌جا: 42, 1, 7, 123, 2024
structural_only = [0.4092, 0.4186, 0.4846, 0.4677, 0.4334]
structural_temporal = [0.3401, 0.3754, 0.3815, 0.3857, 0.4320]
structural_global_fixed = [0.3764, 0.4096, 0.3679, 0.3278, 0.4313]
all_three_fixed = [0.3028, 0.2820, 0.3117, 0.4378, 0.3389]
film_modulated = [0.3273, 0.3407, 0.4969, 0.4825, 0.3629]
skip_gcn = [0.4019, 0.3857, 0.3732, 0.3741, 0.3964]
tie_baseline = [0.4247, 0.3704, 0.4552, 0.5417, 0.3223]

print("=" * 60)
print("مهم‌ترین سوال: آیا گراف واقعاً بهتر از دونستن فقط زمانه؟")
print("=" * 60)
paired_significance_test(structural_only, tie_baseline, "GraphSAGE ساختاری", "TIE بدون گراف")

print("\n" + "=" * 60)
print("آیا GraphSAGE از Skip-GCN بهتره؟")
print("=" * 60)
paired_significance_test(structural_only, skip_gcn, "GraphSAGE ساختاری", "Skip-GCN")

print("\n" + "=" * 60)
print("آیا افزودن جریان کلی، حتی نسخه اصلاح‌شده، واقعاً ضرر می‌زنه؟")
print("=" * 60)
paired_significance_test(structural_only, structural_global_fixed, "GraphSAGE ساختاری", "ساختاری + کلی اصلاح‌شده")

print("\n" + "=" * 60)
print("آیا FiLM زمانی واقعاً ضرر می‌زنه؟")
print("=" * 60)
paired_significance_test(structural_only, film_modulated, "GraphSAGE ساختاری", "FiLM + کلی")

print("\n" + "=" * 60)
print("آیا نسخه اول سه‌جریانی، با gate رقابتی خام، واقعاً بدترین حالته؟")
print("=" * 60)
paired_significance_test(structural_only, all_three_fixed, "GraphSAGE ساختاری", "هر سه جریان خام")