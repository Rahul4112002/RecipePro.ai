from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from decouple import config
from PIL import Image

def ask(recipe_message=None, image_path=None):
    try:
        # Initialize the API client
        SECRET_KEY = config('API_KEY')
        print(SECRET_KEY)
        genai = GoogleGenerativeAI(model="gemini-pro", api_key=SECRET_KEY)

        if image_path:
            # Generate a recipe based on the uploaded image
            image = Image.open(image_path)  # Open the image file
            prompt_template = PromptTemplate.from_template(
                "Your name is RecipePro, an AI-powered culinary expert specializing in recipes.\n\n\
                Analyze the uploaded image and generate the recipe in the following format:\n\n\
                **Recipe Name**\n\
                - Ingredients with Quantity\n\
                **Instructions**\n\
                **Tips** (if required)\n\n"
            )
            formatted_prompt = prompt_template.format(user_query="Analyze this image and generate a recipe.")
            response = genai.invoke([formatted_prompt, image])  # Pass both prompt and image
        elif recipe_message:
            # Generate a recipe based on the provided message
            prompt_template = PromptTemplate.from_template(
                "Your name is RecipePro, an AI-powered culinary expert specializing in quick and easy recipes.\n\n\
                Generate the recipe in the following format:\n\n\
                **Recipe Name**\n\
                - Ingredients with Quantity\n\
                **Instructions**\n\
                **Tips** (if required)\n\n\
                User's query: {user_query}"
            )
            formatted_prompt = prompt_template.format(user_query=recipe_message)
            response = genai.invoke(formatted_prompt)
        else:
            return "No input provided."

        # Convert the response content to string
        if isinstance(response, list):
            recipe_text = "\n".join(response)  # Join list into a string
        else:
            recipe_text = str(response)  # Ensure it's a string

        # Format the response for cleaner output
        recipe_text_lines = recipe_text.split('\n')
        formatted_text = []

        for line in recipe_text_lines:
            if line.startswith("**") and line.endswith("**"):  # For bold titles
                formatted_text.append(f"<b>{line.strip('**')}</b>")
            else:
                formatted_text.append(line)

        return "\n".join(formatted_text)

    except Exception as e:
        return f"Error: {str(e)}"
