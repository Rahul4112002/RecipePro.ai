from django.shortcuts import render
from chief.langchain import ask  # Import the ask function from langchain.py
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

def home(request):
    return render(request, 'home.html')  # Render the home page

def generate_recipe(request):
    if request.method == "POST":
        recipe_message = request.POST.get("recipe_message", "")
        image_file = request.FILES.get("food_image")
        
        image_url = None  # Default to None if no image is uploaded
        generated_image_url = None  # For AI-generated images

        # Check if both inputs are empty
        if not recipe_message and not image_file:
            return render(request, 'recipe.html', {'recipe': "No input provided.", 'input_data': ""})

        # Generate a unique cache key based on input text and image
        cache_key = recipe_message
        if image_file:
            cache_key += f"_image_{image_file.name}"

            # Save the image to the "uploads" directory
            image_path = f"uploads/{image_file.name}"
            default_storage.save(image_path, ContentFile(image_file.read()))

            # Get the image URL for display in the template
            image_url = default_storage.url(image_path)

        # Check if cached recipe exists
        session_data = request.session.get("recipe_data", {})
        if cache_key in session_data:
            cached_data = session_data[cache_key]
            # Handle backward compatibility
            if isinstance(cached_data, dict):
                recipe = cached_data.get('recipe', '')
                generated_image_url = cached_data.get('generated_image_url', None)
            else:
                # Old format was just a string
                recipe = cached_data
                generated_image_url = None
        else:
            # Call the AI API with both text and image (if available)
            result = ask(recipe_message, image_file)
            
            # Handle the new return format (recipe, generated_image_url)
            if isinstance(result, tuple):
                recipe, generated_image_url = result
            else:
                recipe = result
                generated_image_url = None
            
            # Save to session
            session_data[cache_key] = {
                'recipe': recipe,
                'generated_image_url': generated_image_url
            }
            request.session["recipe_data"] = session_data

        # Pass the recipe and image URLs to the template
        return render(request, 'recipe.html', {
            'recipe': recipe,
            'input_data': recipe_message,
            'image_url': image_url,  # User uploaded image
            'generated_image_url': generated_image_url  # AI generated image
        })

    else:
        return render(request, 'recipe.html', {})

def saved_recipes(request):
    """View for displaying saved recipes from localStorage"""
    return render(request, 'saved_recipes.html')
