from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("detector", "0001_initial")]
    operations = [
        migrations.AddField(
            model_name="analysisrun",
            name="current_step",
            field=models.CharField(max_length=200, blank=True, default=""),
        ),
        migrations.AddField(
            model_name="analysisrun",
            name="progress_pct",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="analysisrun",
            name="source_file_name",
            field=models.CharField(max_length=200, blank=True, default=""),
        ),
    ]
