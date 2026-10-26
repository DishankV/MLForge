from rest_framework import status
from rest_framework.test import APITestCase
from apps.registry.models import ModelVersion


class ModelVersionAPITests(APITestCase):
    url = "/api/models/"

    def setUp(self):
        self.valid_payload = {
            "name": "churn-logreg",
            "version": "v1",
            "status": "candidate",
            "artifact_path": (
                "ml/registry/models/churn-logreg-v1/model.joblib"
            ),
            "accuracy": 0.8055,
            "roc_auc": 0.8419,
        }

    def test_register_model_version(self):
        response = self.client.post(
            self.url, self.valid_payload, format="json"
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(ModelVersion.objects.count(), 1)
        self.assertEqual(response.json()["name"], "churn-logreg")

    def test_list_model_versions(self):
        ModelVersion.objects.create(**self.valid_payload)

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.json()), 1)

    def test_reject_accuracy_above_one(self):
        payload = {
            **self.valid_payload,
            "version": "v2",
            "accuracy": 1.5,
        }

        response = self.client.post(
            self.url, payload, format="json"
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(ModelVersion.objects.count(), 0)

    def test_reject_empty_artifact_path(self):
        payload = {
            **self.valid_payload,
            "artifact_path": "   ",
        }

        response = self.client.post(
            self.url, payload, format="json"
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(ModelVersion.objects.count(), 0)
