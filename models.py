from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class registerUser(models.Model):
    firstname = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    email_id = models.CharField(primary_key=True,max_length=100)
    password = models.CharField(max_length=100)
    confirmPassowrd = models.CharField(max_length=100)