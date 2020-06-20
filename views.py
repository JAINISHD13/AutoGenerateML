import simplejson as simplejson
from django.shortcuts import render, render_to_response
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
import json
from django.views.decorators.csrf import requires_csrf_token, csrf_exempt

def index(request):
    return render(request,'index.html',{})

def registerUser(request):
    context ={}
    return render(request,'register.html',context)

def loginUser(request):
    context ={}
    return render(request,'login.html',context)

def hello(request):
    return HttpResponse('Hello World!')

def home(request):
    return render_to_response('response.html', {'variable': 'world'})