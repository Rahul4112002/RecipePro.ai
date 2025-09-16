from django.shortcuts import render
from chief.langchain import generate_recipe_from_text, generate_recipe_from_image, generate_recipe_image
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import json

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
            # Generate recipe based on input type
            if image_file:
                # Generate recipe from image
                result = generate_recipe_from_image(image_file)
            else:
                # Generate recipe from text ingredients
                ingredients_list = [ingredient.strip() for ingredient in recipe_message.split(',')]
                result = generate_recipe_from_text(ingredients_list)
            
            # Check if there was an error
            if isinstance(result, dict) and 'error' in result:
                recipe = f"Error: {result['error']}"
                generated_image_url = None
            else:
                # Format the recipe output
                if isinstance(result, dict):
                    recipe_parts = []
                    if result.get('recipe_name'):
                        recipe_parts.append(f"**{result['recipe_name']}**\n")
                    
                    if result.get('ingredients'):
                        recipe_parts.append("**Ingredients:**")
                        for ingredient in result['ingredients']:
                            recipe_parts.append(f"• {ingredient}")
                        recipe_parts.append("")
                    
                    if result.get('instructions'):
                        recipe_parts.append("**Instructions:**")
                        for i, instruction in enumerate(result['instructions'], 1):
                            recipe_parts.append(f"{i}. {instruction}")
                        recipe_parts.append("")
                    
                    if result.get('tips'):
                        recipe_parts.append("**Tips:**")
                        for tip in result['tips']:
                            recipe_parts.append(f"• {tip}")
                    
                    recipe = "\n".join(recipe_parts)
                    
                    # Generate image for the recipe
                    recipe_name = result.get('recipe_name', 'Generated Recipe')
                    generated_image_url = generate_recipe_image(recipe_name)
                else:
                    recipe = str(result)
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
