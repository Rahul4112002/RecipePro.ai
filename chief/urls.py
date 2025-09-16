from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  # Main page
    path('recipe/', views.generate_recipe, name='recipe'),  # Recipe generation page
    path('saved-recipes/', views.saved_recipes, name='saved_recipes')  # Saved recipes page
]