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
        prompt = f'Create a recipe using: {ingredients_str}'
        
        headers = {'Authorization': f'Bearer {groq_api_key}', 'Content-Type': 'application/json'}
        data = {'model': 'llama-3.1-8b-instant', 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 1500}
        
        response = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            recipe_text = result['choices'][0]['message']['content']
            return {'recipe_name': 'Generated Recipe', 'ingredients': ['See recipe below'], 'instructions': [recipe_text], 'tips': []}
        else:
            return {'error': f'API Error: {response.status_code}'}
    except Exception as e:
        return {'error': f'Error: {str(e)}'}

def generate_recipe_image(recipe_name):
    return f'https://image.pollinations.ai/prompt/{recipe_name.replace(" ", "%20")}%20food'

def generate_recipe_from_image(image_file):
    return generate_recipe_from_text(['chicken', 'vegetables', 'spices'])
