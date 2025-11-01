import os
import requests
from dotenv import load_dotenv

load_dotenv()

def generate_recipe_from_text(ingredients_list):
    try:
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            return {'error': 'API key not found'}
        
        ingredients_str = ', '.join(ingredients_list)
        ingredients_formatted = '\n• '.join(ingredients_list)
        
        # Enhanced prompt to generate recipe in specific format
        prompt = f"""You are a professional chef. Create a simple and delicious recipe using these exact ingredients: {ingredients_str}

Format your response EXACTLY like this:

**Recipe Title**
[Write a short, attractive recipe name]

**Ingredients:**
• {ingredients_formatted}

**Instructions:**
1. [First step in very simple English]
2. [Second step in very simple English]
3. [Continue with each step, one by one]
[Add as many steps as needed, keep each step clear and easy to follow]

**Tips:**
• [Tip 1 to make the recipe more delicious]
• [Tip 2 to make the recipe more delicious]
• [Tip 3 to make the recipe more delicious]

IMPORTANT: 
- Use VERY SIMPLE English that anyone can understand
- Keep the ingredients list exactly as provided
- Make each instruction step clear and detailed
- Number all instruction steps
- Provide practical tips"""
        
        headers = {'Authorization': f'Bearer {groq_api_key}', 'Content-Type': 'application/json'}
        data = {'model': 'llama-3.3-70b-versatile', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 2000}
        
        response = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            recipe_text = result['choices'][0]['message']['content']
            return {'recipe_name': 'Generated Recipe', 'ingredients': ingredients_list, 'instructions': [recipe_text], 'tips': []}
        else:
            # Log more details for debugging
            print(f"API Error Details: Status {response.status_code}, Response: {response.text}")
            return {'error': f'API Error: {response.status_code}'}
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return {'error': f'Error: {str(e)}'}

def generate_recipe_image(recipe_name):
    return f'https://image.pollinations.ai/prompt/{recipe_name.replace(" ", "%20")}%20food'

def generate_recipe_from_image(image_file):
    return generate_recipe_from_text(['chicken', 'vegetables', 'spices'])
