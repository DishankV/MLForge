from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import generics
from apps.registry.models import ModelVersion
from apps.registry.serializers import ModelVersionSerializer

@api_view(["GET"])
def health_check(request):
    return Response({
        "service": "MLForge Control Plane",
        "status": "healthy",
    })

class ModelVersionListCreateView(generics.ListCreateAPIView):
    queryset = ModelVersion.objects.all()
    serializer_class = ModelVersionSerializer
