from django.conf.urls import url, include
from rest_framework.routers import DefaultRouter

from .views import SentimentAnalysisView

router = DefaultRouter()

# router.register('sentiment', SentimentAnalysisView, base_name="sentiments")

urlpatterns = [
    url(r'sentiment', SentimentAnalysisView.as_view()),
]
