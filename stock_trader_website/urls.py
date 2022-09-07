from django.contrib import admin
from django.urls import path, include
from trader.views import SignUp, Login, Logout


urlpatterns = [
    path('admin/', admin.site.urls),
    path('stocks/', include('trader.urls')),
    path('register/', SignUp.as_view(), name='register'),
    path('login/', Login.as_view(), name='login'),
    path('logout/', Logout.as_view(), name='logout'),
]
