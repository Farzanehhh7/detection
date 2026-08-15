from django.db import models


class AnalysisRun(models.Model):
    """One pipeline run -- either the pretrained SAML-D result set, or a
    later run on newly-uploaded data (Phase D)."""
    STATUS_CHOICES = [
        ("pending", "در انتظار"),
        ("running", "در حال اجرا"),
        ("done", "تمام‌شده"),
        ("failed", "خطا"),
    ]
    name = models.CharField(max_length=200)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)
    current_step = models.CharField(max_length=200, blank=True, default="")
    progress_pct = models.IntegerField(default=0)
    source_file_name = models.CharField(max_length=200, blank=True, default="")

    def __str__(self):
        return f"{self.name} ({self.status})"


class Account(models.Model):
    """One account/node, with its raw features and the model's predictions.
    Mirrors exactly the 8 feature columns in samld_processed_v3.pt, plus
    the binary/type/family predictions from step22/step25/step31."""
    run = models.ForeignKey(AnalysisRun, related_name="accounts", on_delete=models.CASCADE)
    node_id = models.IntegerField(db_index=True)

    sent_amount_sum = models.FloatField()
    sent_amount_mean = models.FloatField()
    sent_amount_count = models.FloatField()
    sent_payment_type_nunique = models.FloatField()
    recv_amount_sum = models.FloatField()
    recv_amount_mean = models.FloatField()
    recv_amount_count = models.FloatField()
    recv_payment_type_nunique = models.FloatField()

    prob_illicit = models.FloatField(db_index=True)
    is_flagged = models.BooleanField(default=False, db_index=True)
    actual_label = models.IntegerField(null=True, blank=True)  # ground truth, if known (demo/eval only)

    predicted_type = models.CharField(max_length=64, blank=True)
    predicted_family = models.CharField(max_length=64, blank=True)

    class Meta:
        ordering = ["-prob_illicit"]
        unique_together = [("run", "node_id")]

    def __str__(self):
        return f"Account {self.node_id} (P={self.prob_illicit:.3f})"

    @property
    def feature_dict(self):
        return {
            "sent_amount_sum": self.sent_amount_sum,
            "sent_amount_mean": self.sent_amount_mean,
            "sent_amount_count": self.sent_amount_count,
            "sent_payment_type_nunique": self.sent_payment_type_nunique,
            "recv_amount_sum": self.recv_amount_sum,
            "recv_amount_mean": self.recv_amount_mean,
            "recv_amount_count": self.recv_amount_count,
            "recv_payment_type_nunique": self.recv_payment_type_nunique,
        }


class FeatureAttribution(models.Model):
    """Own-feature Grad x Input attribution, from step33."""
    account = models.ForeignKey(Account, related_name="attributions", on_delete=models.CASCADE)
    feature_name = models.CharField(max_length=64)
    value = models.FloatField()
    attribution = models.FloatField()
    rank = models.IntegerField()

    class Meta:
        ordering = ["rank"]


class NeighborInfluence(models.Model):
    """Which other accounts (up to 2 hops) most influenced this account's
    score, from step33's neighbor-contribution analysis."""
    account = models.ForeignKey(Account, related_name="neighbor_influences", on_delete=models.CASCADE)
    neighbor_node_id = models.IntegerField()
    influence = models.FloatField()
    neighbor_is_flagged = models.BooleanField(default=False)
    rank = models.IntegerField()

    class Meta:
        ordering = ["rank"]


class GraphEdge(models.Model):
    """Real transaction edges among imported accounts, for visualization."""
    run = models.ForeignKey(AnalysisRun, related_name="edges", on_delete=models.CASCADE)
    source_node_id = models.IntegerField(db_index=True)
    target_node_id = models.IntegerField(db_index=True)
