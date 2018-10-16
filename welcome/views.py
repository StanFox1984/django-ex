import os
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse

from . import database
from .models import PageView
from predict import predict

# Create your views here.

def index(request):
    hostname = os.getenv('HOSTNAME', 'unknown')
    PageView.objects.create(hostname=hostname)
#ff
    return render(request, 'welcome/index.html', {
        'hostname': hostname,
        'database': database.info(),
        'count': PageView.objects.count()
    })

def mainview(request):
    s = render(request, 'welcome/MainView.htm')
    s += run_all_tests()
    return s

def health(request):
    return HttpResponse(PageView.objects.count())
