import simplejson as simplejson
from django.core.serializers import json
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.http import JsonResponse
from django.template import RequestContext
from django.views.decorators.csrf import csrf_protect, csrf_exempt

from .models import registerUser

def index(request):
    return render(request,'index.html',{})

def registerUser(request):
    if request.method == 'POST':
        register1_data = request.POST.dict()
        firstname = register1_data.get('inputFirstName')
        lastname = register1_data.get('inputLastName')
        email_id = register1_data.get('inputEmailAddress')
        password = register1_data.get('inputPassword')
        confirmPassowrd = register1_data.get('inputConfirmPassword')
        print(firstname,lastname,email_id,password)
        messages.success(request,f'Account created for {firstname}!')
        return redirect('loginUser')
    else:
        return render(request,"register.html")

def loginUser(request):
    context ={}
    return render(request,'login.html',context)

@csrf_exempt
def getModelAttribute(request):
    if request.method == "POST" and request.is_ajax():
        arry1D = request.POST.get('arry1D')
        count = request.POST.get('count')
        print(arry1D,count)
        return JsonResponse({'arry1D':arry1D})

def hello(request):
    return HttpResponse('Hello World!')

def home(request):
    return render(request,'response.html', {'variable': 'world'})