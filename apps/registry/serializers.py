
from rest_framework import serializers
from apps.registry.models import ModelVersion


class ModelVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelVersion
        fields = [
            "id",
            "name",
            "version",
            "status",
            "artifact_path",
            "accuracy",
            "roc_auc",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]

    def validate_name(self, value):
        value = value.strip()
        if not value:
            raise serializers.ValidationError(
                "Model name cannot be empty."
            )
        return value

    def validate_version(self, value):
        value = value.strip()
        if not value:
            raise serializers.ValidationError(
                "Model version cannot be empty."
            )
        return value

    def validate_artifact_path(self, value):
        value = value.strip()
        if not value:
            raise serializers.ValidationError(
                "Artifact path cannot be empty."
            )
        return value

    def validate_accuracy(self, value):
        if value is not None and not 0 <= value <= 1:
            raise serializers.ValidationError(
                "Accuracy must be between 0 and 1."
            )
        return value

    def validate_roc_auc(self, value):
        if value is not None and not 0 <= value <= 1:
            raise serializers.ValidationError(
                "ROC-AUC must be between 0 and 1."
            )
        return value
