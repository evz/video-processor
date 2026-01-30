from django.urls import path
from . import views

urlpatterns = [
    path('', views.video_list, name='video_list'),
    path('video/<int:video_id>/', views.video_detail, name='video_detail'),
    path('video/<int:video_id>/add-tag/', views.add_tag, name='add_tag'),
    path('video/<int:video_id>/remove-tag/<int:tag_id>/', views.remove_tag, name='remove_tag'),
]
