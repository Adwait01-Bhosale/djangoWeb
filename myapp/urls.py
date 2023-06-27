from django.urls import path
from myapp.views import kyc, homePage, videoCam, grayscale


urlpatterns = [
     path('', homePage, name= 'home'),
     path('kyc/', kyc, name='KYC'),
     path('tries/', videoCam, name="tries"),
     path('cam/', grayscale, name="cam")
]
