from django.urls import path
from apps.registry.views import (
    health_check,
    ModelVersionListCreateView,
)

urlpatterns = [
    path("health/", health_check, name="health-check"),
    path("models/", ModelVersionListCreateView.as_view(), name="model-list-create"),
]