from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='trader-home'),
    path('dow_home', views.dow_home, name='dow-home'),
    path('tehran_home', views.tehran_home, name='tehran-home'),
]