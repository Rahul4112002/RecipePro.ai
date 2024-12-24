from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from decouple import config

def ask(recipe_message):
    try:
        # Initialize the API client
        SECRET_KEY = config('API_KEY')
        genai = GoogleGenerativeAI(model="gemini-pro", api_key=SECRET_KEY)

        prompt_template = PromptTemplate.from_template(
            "Your name is RecipePro, an AI-powered culinary expert specializing in quick and easy recipes that can be prepared in 5 minutes.\n\n\
            Generate the recipe in the following format:\n\n\
             **Recipe Name**\n\
            - Ingredients with Quantity\n\
             **Instructions**\n\
             **Tips** (if required)\n\n\
            If the user’s query is not related to recipes or if you don’t know the answer, respond with 'I don't know the answer.\n\n\
            User's query: {user_query}'"

        )

        formatted_prompt = prompt_template.format(user_query=recipe_message)

        # Call the AI API and get the response
        response = genai.invoke(formatted_prompt)

        # Convert the response content to string
        if isinstance(response, list):
            recipe_text = "\n".join(response)  # Join list into a string
        else:
            recipe_text = str(response)  # Ensure it's a string
            
        # Remove any '*' symbols to avoid unwanted markdown formatting
        recipe_text = recipe_text.replace('*', '')  # Removes all '*' symbols

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


