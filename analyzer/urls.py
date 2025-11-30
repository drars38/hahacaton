from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import TextDocumentViewSet, TextEntryViewSet, ValidationResultViewSet

router = DefaultRouter()
router.register(r'documents', TextDocumentViewSet, basename='document')
router.register(r'entries', TextEntryViewSet, basename='entry')
router.register(r'validations', ValidationResultViewSet, basename='validation')

urlpatterns = [
    path('', include(router.urls)),
]

