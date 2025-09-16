import os
import base64
import requests
import json
from PIL import Image
import io
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List, Optional

# Load environment variables
load_dotenv()

# Define the output schema using Pydantic
class RecipeOutput(BaseModel):
    recipe_name: str = Field(description="The name of the recipe")
    ingredients: List[str] = Field(description="List of ingredients with quantities")
    instructions: List[str] = Field(description="Step by step instructions")
    tips: Optional[List[str]] = Field(default=None, description="Optional tips for the recipe")

# Initialize the output parser
output_parser = PydanticOutputParser(pydantic_object=RecipeOutput)

# Create the prompt template
recipe_template = """You are RecipePro, a friendly cooking helper. You make easy recipes for everyone.

Make a simple recipe using these ingredients. Use VERY SIMPLE words and short sentences:

🍽️ **[WRITE A SIMPLE, TASTY FOOD NAME]**

⏰ **Time to Prepare:** [X minutes] | **Time to Cook:** [X minutes] | **Total Time:** [X minutes]
👥 **Serves:** [X people]

🛒 **WHAT YOU NEED:**
• [How much] [Simple ingredient name]
• [How much] [Simple ingredient name]
[List all ingredients with easy measurements like "1 cup" or "2 spoons"]

👨‍🍳 **HOW TO MAKE IT:**
1. [Very simple step using easy words]
2. [Very simple step using easy words]
[Keep steps short and easy to understand]

💡 **HELPFUL TIPS:**
• [Simple tip to make it better]
• [Easy substitute if something is missing]

✨ **ABOUT THIS FOOD:**
[2-3 simple sentences about how it tastes and why people will like it]

IMPORTANT: Use simple words only! No hard cooking words. Write like you're teaching a beginner.

If the request is not about food, say 'I only help make food recipes! Tell me what ingredients you have or what food you want to make.'

What you want to cook: {query}
"""

# Create prompt template for image-based recipe generation
image_recipe_template = """You are RecipePro, a friendly cooking helper. You make easy recipes for everyone.

Look at this food picture and make a simple recipe. Use VERY SIMPLE words and short sentences:

🍽️ **[WRITE A SIMPLE NAME FOR THIS FOOD]**

⏰ **Time to Prepare:** [X minutes] | **Time to Cook:** [X minutes] | **Total Time:** [X minutes]
👥 **Serves:** [X people]

🛒 **WHAT YOU NEED:**
• [How much] [Simple ingredient name]
• [How much] [Simple ingredient name]
[List all ingredients with easy measurements like "1 cup" or "2 spoons"]

👨‍🍳 **HOW TO MAKE IT:**
1. [Very simple step using easy words]
2. [Very simple step using easy words]
[Keep steps short and easy to understand - explain how to make what you see in the picture]

💡 **HELPFUL TIPS:**
• [Simple tip to make it better]
• [Easy substitute if something is missing]

✨ **ABOUT THIS FOOD:**
[2-3 simple sentences about how it looks and tastes]

IMPORTANT: Use simple words only! No hard cooking words. Write like you're teaching a beginner.

If the picture doesn't show food, say 'I only help with food pictures! Please show me a picture of food you want to make.'
"""

# Create prompt template for image+text recipe generation
combined_recipe_template = """Your name is RecipePro, an AI-powered culinary expert who creates delicious and creative recipes.

Analyze this food image and consider the user's input: {query}

Generate a complete recipe with the following EXACT format:

🍽️ **[CREATE AN ATTRACTIVE, CATCHY RECIPE NAME THAT COMBINES THE IMAGE AND USER REQUEST]**

⏰ **Prep Time:** [X minutes] | **Cook Time:** [X minutes] | **Total Time:** [X minutes]
👥 **Servings:** [X people]

🛒 **INGREDIENTS:**
• [Quantity] [Ingredient name]
• [Quantity] [Ingredient name]
[Continue for all ingredients with proper measurements, incorporating user's requested ingredients where possible]

👨‍🍳 **INSTRUCTIONS:**
1. [Clear, detailed step]
2. [Clear, detailed step]
[Continue with all steps, adapting based on both the image and user preferences]

💡 **CHEF'S TIPS:**
• [Helpful tip for better results]
• [Substitution suggestion or cooking tip]

✨ **RECIPE DESCRIPTION:**
[2-3 sentences describing how this recipe combines what's shown in the image with the user's preferences]

If the image doesn't show food or the request is unrelated to cooking, respond with 'I can only create recipes! Please share ingredients or upload a food image.'
"""

# Create the prompt templates
text_prompt = PromptTemplate(
    template=recipe_template,
    input_variables=["query"]
)

# Initialize the LLM
def get_llm():
    try:
        API_KEY = os.getenv('GROQ_API_KEY')
        if not API_KEY:
            raise ValueError("Groq API key not found. Please set the GROQ_API_KEY environment variable.")
        
        llm = ChatGroq(
            model="llama-3.1-8b-instant",  # Fast and reliable model
            groq_api_key=API_KEY,
            temperature=0.7,
            max_tokens=2048
        )
        return llm
    except Exception as e:
        raise Exception(f"Error initializing LLM: {str(e)}")

# Function to generate recipe images using Hugging Face API
def generate_recipe_image(recipe_name, ingredients_list):
    """
    Generate an appetizing image of the recipe using Hugging Face's Stable Diffusion
    """
    try:
        # Create a detailed prompt for the image generation
        prompt = f"A professional, appetizing photo of {recipe_name}, beautifully plated, high quality food photography, studio lighting, garnished, delicious looking, restaurant quality presentation"
        
        # Hugging Face API endpoint for Stable Diffusion
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        
        # You can get a free API token from huggingface.co
        hf_token = os.getenv('HUGGINGFACE_API_TOKEN')
        if not hf_token:
            print("No Hugging Face API token found. Image generation skipped.")
            return None
            
        headers = {"Authorization": f"Bearer {hf_token}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "num_inference_steps": 20,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            # Save the image
            image_data = response.content
            image = Image.open(io.BytesIO(image_data))
            
            # Save to media directory
            import uuid
            image_filename = f"generated_{uuid.uuid4().hex[:8]}.jpg"
            image_path = f"media/generated/{image_filename}"
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            
            # Save the image
            image.save(image_path, "JPEG", quality=95)
            
            return f"/media/generated/{image_filename}"
        else:
            print(f"Image generation failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None

# Alternative free image generation using Pollinations API
def generate_recipe_image_pollinations(recipe_name):
    """
    Generate recipe image using Pollinations.ai (free, no API key required)
    """
    try:
        # Create a detailed prompt for better image quality
        prompt = f"professional food photography of {recipe_name}, appetizing, high quality, restaurant style plating, well lit, delicious looking, gourmet presentation"
        
        # URL encode the prompt
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        
        # Pollinations API URL with better parameters
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&model=flux&enhance=true&nologo=true"
        
        print(f"Generating image for: {recipe_name}")
        print(f"Image URL: {image_url}")
        
        # Return the URL directly (Pollinations serves images directly)
        return image_url
        
    except Exception as e:
        print(f"Error with Pollinations image generation: {str(e)}")
        return None

# Alternative image generation using Unsplash API (free with attribution)
def generate_recipe_image_unsplash(recipe_name):
    """
    Generate recipe image using Unsplash API (free, requires API key)
    """
    try:
        unsplash_access_key = os.getenv('UNSPLASH_ACCESS_KEY')
        if not unsplash_access_key:
            print("No Unsplash API key found. Skipping Unsplash image generation.")
            return None
            
        # Search for food images on Unsplash
        search_query = f"{recipe_name} food"
        url = f"https://api.unsplash.com/search/photos"
        
        headers = {
            "Authorization": f"Client-ID {unsplash_access_key}"
        }
        
        params = {
            "query": search_query,
            "per_page": 5,
            "orientation": "square"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                # Get the first image URL
                image_url = data['results'][0]['urls']['regular']
                
                # Download and save the image
                img_response = requests.get(image_url, timeout=30)
                if img_response.status_code == 200:
                    import uuid
                    image_filename = f"unsplash_{uuid.uuid4().hex[:8]}.jpg"
                    image_path = f"media/generated/{image_filename}"
                    
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    
                    # Save the image
                    with open(image_path, 'wb') as f:
                        f.write(img_response.content)
                    
                    return f"/media/generated/{image_filename}"
        
        return None
            
    except Exception as e:
        print(f"Error with Unsplash image generation: {str(e)}")
        return None

# AI Image generation using free APIs like Prodia or others
def generate_recipe_image_free_ai(recipe_name):
    """
    Generate recipe image using free AI APIs like Prodia
    """
    try:
        # Check if Prodia API key is available
        prodia_api_key = os.getenv('PRODIA_API_KEY')
        if not prodia_api_key:
            print("No Prodia API key found. Skipping Prodia.")
            return None
            
        print(f"Trying Prodia for: {recipe_name}")
        return None  # Skip Prodia for now, go straight to Pollinations
        
    except Exception as e:
        print(f"Error with Prodia: {str(e)}")
        return None

# Alternative free image generation using Pollinations API
def generate_recipe_image_pollinations(recipe_name):
    """
    Generate recipe image using Pollinations.ai (free, no API key required)
    """
    try:
        # Create a detailed prompt for better image quality
        prompt = f"professional food photography of {recipe_name}, appetizing, high quality, restaurant style plating, well lit, delicious looking, gourmet presentation"
        
        # URL encode the prompt
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        
        # Pollinations API URL with better parameters
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&model=flux&enhance=true&nologo=true"
        
        print(f"Generating image for: {recipe_name}")
        print(f"Image URL: {image_url}")
        
        # Return the URL directly (Pollinations serves images directly)
        return image_url
        
    except Exception as e:
        print(f"Error with Pollinations image generation: {str(e)}")
        return None

# Create formatter for output
def format_recipe_output(recipe_text):
    """Format the recipe text with HTML for better display"""
    if not recipe_text:
        return recipe_text
    
    # Split into lines for processing
    lines = recipe_text.split('\n')
    formatted_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        
        # Format emoji headers and bold sections
        if stripped_line.startswith('🍽️') and '**' in stripped_line:
            # Recipe title - remove all asterisks properly
            title = stripped_line.replace('🍽️', '').replace('**', '').strip()
            formatted_lines.append(f'<div class="recipe-title">🍽️ <strong>{title}</strong></div>')
        
        elif stripped_line.startswith('⏰'):
            # Time and serving info - clean up asterisks
            clean_line = stripped_line.replace('**', '')
            formatted_lines.append(f'<div class="recipe-meta">{clean_line}</div>')
        
        elif stripped_line.startswith('👥'):
            # Servings info - clean up asterisks
            clean_line = stripped_line.replace('**', '')
            formatted_lines.append(f'<div class="recipe-meta">{clean_line}</div>')
        
        elif stripped_line.startswith('🛒'):
            # Ingredients header
            formatted_lines.append(f'<div class="section-header">🛒 <strong>INGREDIENTS:</strong></div>')
            
        elif stripped_line.startswith('👨‍🍳'):
            # Instructions header
            formatted_lines.append(f'<div class="section-header">👨‍🍳 <strong>INSTRUCTIONS:</strong></div>')
            
        elif stripped_line.startswith('💡'):
            # Tips header
            formatted_lines.append(f'<div class="section-header">💡 <strong>CHEF\'S TIPS:</strong></div>')
            
        elif stripped_line.startswith('✨'):
            # Description header
            formatted_lines.append(f'<div class="section-header">✨ <strong>RECIPE DESCRIPTION:</strong></div>')
        
        elif stripped_line.startswith('•'):
            # Ingredient or tip items
            content = stripped_line[1:].strip()
            formatted_lines.append(f'<div class="recipe-item">• {content}</div>')
        
        elif stripped_line and stripped_line[0].isdigit() and '. ' in stripped_line:
            # Numbered instructions
            formatted_lines.append(f'<div class="recipe-step">{stripped_line}</div>')
        
        elif stripped_line and not stripped_line.startswith(('🍽️', '⏰', '👥', '🛒', '👨‍🍳', '💡', '✨')):
            # Regular text
            formatted_lines.append(f'<div class="recipe-text">{stripped_line}</div>')
        
        else:
            # Empty lines or unformatted content
            formatted_lines.append('<br>' if not stripped_line else f'<div>{stripped_line}</div>')
    
    return '\n'.join(formatted_lines)

# Main function
def ask(recipe_message, image_file=None):
    try:
        # Get the LLM
        llm = get_llm()
        
        # Handle image-based recipe generation
        if image_file:
            # For now, we'll use text-based generation with a note about the image
            # Since Groq doesn't support image inputs, we'll create a recipe based on user text
            if recipe_message:
                enhanced_message = f"Create a recipe considering that the user has uploaded a food image and mentioned: {recipe_message}"
            else:
                enhanced_message = "Create a delicious recipe for a dish that would look great in a food photo"
            
            # Use text-based generation
            chain = LLMChain(
                llm=llm,
                prompt=text_prompt
            )
            
            response = chain.run(query=enhanced_message)
            recipe_text = response
        else:
            # For text-based queries, use the LangChain
            chain = LLMChain(
                llm=llm,
                prompt=text_prompt
            )
            
            # Execute the chain
            response = chain.run(query=recipe_message)
            recipe_text = response
        
        # Generate recipe image for all recipes
        generated_image_url = None
        if recipe_text and "I can only help you" not in recipe_text:
            try:
                # Extract recipe name from the generated text for image generation
                lines = recipe_text.split('\n')
                recipe_name = ""
                for line in lines:
                    if '🍽️' in line:
                        # Extract the recipe name and clean it thoroughly
                        recipe_name = line.replace('🍽️', '').replace('**', '').replace('*', '').strip()
                        break
                    elif line.strip() and ('Delight' in line or 'Recipe' in line or 'Salad' in line or 'Soup' in line or 'Curry' in line):
                        # Fallback: look for common recipe words in the first few lines
                        recipe_name = line.replace('**', '').replace('*', '').strip()[:50]  # Take first 50 chars
                        break
                
                # If still no recipe name, use a generic one based on ingredients or query
                if not recipe_name and recipe_message:
                    recipe_name = f"delicious {recipe_message[:30]}"
                elif not recipe_name:
                    recipe_name = "delicious homemade dish"
                    
                print(f"Extracted recipe name: '{recipe_name}'")
                
                if recipe_name:
                    print(f"Attempting to generate image for recipe: {recipe_name}")
                    # Try multiple image generation services in order of preference
                    # 1. First try Prodia (free AI with API key)
                    generated_image_url = generate_recipe_image_free_ai(recipe_name)
                    print(f"Prodia result: {generated_image_url}")
                    
                    # 2. If that fails, try Pollinations (free, no API key required)
                    if not generated_image_url:
                        print("Trying Pollinations...")
                        generated_image_url = generate_recipe_image_pollinations(recipe_name)
                        print(f"Pollinations result: {generated_image_url}")
                    
                    # 3. If that fails, try Unsplash (free with API key)
                    if not generated_image_url:
                        print("Trying Unsplash...")
                        generated_image_url = generate_recipe_image_unsplash(recipe_name)
                        print(f"Unsplash result: {generated_image_url}")
                    
                    # 4. If everything fails, try Hugging Face (requires API token)
                    if not generated_image_url:
                        print("Trying Hugging Face...")
                        generated_image_url = generate_recipe_image(recipe_name, [])
                        print(f"Hugging Face result: {generated_image_url}")
                        
                    print(f"Final generated_image_url: {generated_image_url}")
                else:
                    print("No recipe name found for image generation")
                        
            except Exception as img_gen_error:
                print(f"Image generation error: {str(img_gen_error)}")
                # Continue without generated image
        
        # Format the output for display
        formatted_recipe = format_recipe_output(recipe_text)
        
        return formatted_recipe, generated_image_url
        
    except Exception as e:
        if "429" in str(e):
            return "Error: You've reached the API quota limit. Please try again later.", None
        else:
            return f"Error: {str(e)}", None