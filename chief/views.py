from django.shortcuts import render
from chief.langchain import ask  # Import the ask function
import os
from django.conf import settings

# Ensure the upload directory exists
UPLOAD_FOLDER = os.path.join(settings.BASE_DIR, 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def home(request):
    return render(request, 'home.html')  # Render the home page

def generate_recipe(request):
    if request.method == "POST":
        # Get the recipe input from the form
        recipe_message = request.POST.get("recipe_message", "")
        recipe_image = request.FILES.get("recipe_image", None)  # Get the uploaded image

        session_data = request.session.get("recipe_data", {})
        recipe = None

        if recipe_image:
            # Save the uploaded image to the server
            image_path = os.path.join(UPLOAD_FOLDER, recipe_image.name)
            with open(image_path, 'wb') as f:
                for chunk in recipe_image.chunks():
                    f.write(chunk)

            # Generate a recipe from the image
            recipe = ask(image_path=image_path)
        elif recipe_message:
            # Check if the input is already in the session
            if recipe_message in session_data:
                # Use the stored response if available
                recipe = session_data[recipe_message]
            else:
                # Generate a recipe from text input
                recipe = ask(recipe_message=recipe_message)
                session_data[recipe_message] = recipe
                request.session["recipe_data"] = session_data  # Save to session

        # Pass the generated recipe to the template
        return render(request, 'recipe.html', {'recipe': recipe, 'input_data': recipe_message})
    else:
        # Render the recipe page when the user first visits or performs a GET request
        return render(request, 'recipe.html', {})
