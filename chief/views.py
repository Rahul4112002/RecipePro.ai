from django.shortcuts import render
from chief.langchain import ask  # Import the ask function from langchain.py

def home(request):
    return render(request, 'home.html')  # Render the home page

def generate_recipe(request):
    if request.method == "POST":
        # Get the recipe input from the form
        recipe_message = request.POST.get("recipe_message", "")

        # Check if the input is already in the session
        session_data = request.session.get("recipe_data", {})
        if recipe_message in session_data:
            # Use the stored response if available
            recipe = session_data[recipe_message]
        else:
            # Call the AI API and store the response in the session
            recipe = ask(recipe_message) if recipe_message else "No input provided."
            session_data[recipe_message] = recipe
            request.session["recipe_data"] = session_data  # Save to session

        # Pass the generated recipe to the template
        return render(request, 'recipe.html', {'recipe': recipe, 'input_data': recipe_message})
    else:
        # Render the recipe page when the user first visits or performs a GET request
        return render(request, 'recipe.html', {})
