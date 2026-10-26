from django.db import models

# Create your models here.


class ModelVersion(models.Model):
    STATUS_CHOICES = [
        ("candidate", "Candidate"),
        ("approved", "Approved"),
        ("rejected", "Rejected"),
    ]

    name = models.CharField(max_length=100)
    version = models.CharField(max_length=50)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default="candidate",
    )
    artifact_path = models.CharField(max_length=500)
    accuracy = models.FloatField(null=True, blank=True)
    roc_auc = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["name", "version"],
                name="unique_model_name_version",
            )
        ]
        ordering = ["name", "-created_at"]

    def __str__(self):
        return f"{self.name}:{self.version} ({self.status})"