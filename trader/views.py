from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views import View
from django.contrib import messages
from django.http import HttpResponseRedirect


class SignUp(View):
    def get(self, request):
        return render(request, 'trader/register.html')

    def post(self, request):
        email = request.POST.get('email').strip()
        first_name = request.POST.get('first_name').strip()
        last_name = request.POST.get('last_name').strip()
        password = request.POST.get('password').strip()
        password_repeat = request.POST.get('password_repeat').strip()

        if password != password_repeat:
            messages.error(request, 'Passwords do not match')
            return HttpResponseRedirect(request.path_info)  # redirect to the same page

        user_already_exists = User.objects.filter(username=email).exists()
        if user_already_exists:
            messages.error(request, 'User already exists')
            return HttpResponseRedirect(request.path_info)  # redirect to the same page

        user = User.objects.create_user(username=email, email=email, password=password,
                                        first_name=first_name, last_name=last_name)
        user.save()
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect(reverse('trader-home'))


class Login(View):
    def get(self, request):
        return render(request, 'trader/login.html')

    def post(self, request):
        username = request.POST.get('email').strip()
        password = request.POST.get('password').strip()
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect(reverse('trader-home'))
        else:
            messages.error(request, 'Email or Password is incorrect')
            return HttpResponseRedirect(request.path_info)  # redirect to the same page


class Logout(View):
    def get(self, request):
        if request.user.is_authenticated:
            logout(request)
        return redirect(reverse('login'))


def home(request):
    if not request.user.is_authenticated:
        messages.warning(request, 'Login First!')
        return redirect(reverse('login'))
    return render(request, 'trader/home.html')


def dow_home(request):
    return render(request, 'trader/dow_home.html')


def tehran_home(request):
    return render(request, 'trader/tehran_home.html')
