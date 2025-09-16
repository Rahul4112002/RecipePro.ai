import osimport os

import requestsimport requests

import jsonimport json

from PIL import Imagefrom PIL import Image

import ioimport io

from dotenv import load_dotenvfrom dotenv import load_dotenv

from groq import Groq

# Load environment variablesfrom typing import List, Optional

load_dotenv()

# Load environment variables

def get_groq_response(prompt, groq_api_key):load_dotenv()

    """Get response from Groq API directly"""    ingredients: List[str] = Field(description="List of ingredients with quantities")

    try:    instructions: List[str] = Field(description="Step by step instructions")

        headers = {    tips: Optional[List[str]] = Field(default=None, description="Optional tips for the recipe")

            'Authorization': f'Bearer {groq_api_key}',

            'Content-Type': 'application/json'# Initialize the output parser

        }output_parser = PydanticOutputParser(pydantic_object=RecipeOutput)

        

        data = {# Create the prompt template

            'model': 'llama-3.1-8b-instant',recipe_template = """You are RecipePro, a friendly cooking helper. You make easy recipes for everyone.

            'messages': [

                {Make a simple recipe using these ingredients. Use VERY SIMPLE words and short sentences:

                    'role': 'user',

                    'content': prompt🍽️ **[WRITE A SIMPLE, TASTY FOOD NAME]**

                }

            ],⏰ **Time to Prepare:** [X minutes] | **Time to Cook:** [X minutes] | **Total Time:** [X minutes]

            'temperature': 0.7,👥 **Serves:** [X people]

            'max_tokens': 2000

        }🛒 **WHAT YOU NEED:**

        • [How much] [Simple ingredient name]

        response = requests.post(• [How much] [Simple ingredient name]

            'https://api.groq.com/openai/v1/chat/completions',[List all ingredients with easy measurements like "1 cup" or "2 spoons"]

            headers=headers,

            json=data,👨‍🍳 **HOW TO MAKE IT:**

            timeout=301. [Very simple step using easy words]

        )2. [Very simple step using easy words]

        [Keep steps short and easy to understand]

        if response.status_code == 200:

            result = response.json()💡 **HELPFUL TIPS:**

            return result['choices'][0]['message']['content']• [Simple tip to make it better]

        else:• [Easy substitute if something is missing]

            print(f"Groq API Error: {response.status_code} - {response.text}")

            return None✨ **ABOUT THIS FOOD:**

            [2-3 simple sentences about how it tastes and why people will like it]

    except Exception as e:

        print(f"Error calling Groq API: {str(e)}")IMPORTANT: Use simple words only! No hard cooking words. Write like you're teaching a beginner.

        return None

If the request is not about food, say 'I only help make food recipes! Tell me what ingredients you have or what food you want to make.'

def format_recipe_output(raw_output):

    """Format the raw output from LLM into structured recipe"""What you want to cook: {query}

    if not raw_output:"""

        return None

    # Create prompt template for image-based recipe generation

    # Remove asterisks and clean up the textimage_recipe_template = """You are RecipePro, a friendly cooking helper. You make easy recipes for everyone.

    clean_output = raw_output.replace('*', '').replace('#', '')

    Look at this food picture and make a simple recipe. Use VERY SIMPLE words and short sentences:

    # Try to parse the structured output

    try:🍽️ **[WRITE A SIMPLE NAME FOR THIS FOOD]**

        lines = clean_output.strip().split('\n')

        recipe_name = ""⏰ **Time to Prepare:** [X minutes] | **Time to Cook:** [X minutes] | **Total Time:** [X minutes]

        ingredients = []👥 **Serves:** [X people]

        instructions = []

        tips = []🛒 **WHAT YOU NEED:**

        • [How much] [Simple ingredient name]

        current_section = None• [How much] [Simple ingredient name]

        [List all ingredients with easy measurements like "1 cup" or "2 spoons"]

        for line in lines:

            line = line.strip()👨‍🍳 **HOW TO MAKE IT:**

            if not line:1. [Very simple step using easy words]

                continue2. [Very simple step using easy words]

                [Keep steps short and easy to understand - explain how to make what you see in the picture]

            # Detect sections

            if 'recipe name:' in line.lower() or 'name:' in line.lower():💡 **HELPFUL TIPS:**

                recipe_name = line.split(':', 1)[1].strip() if ':' in line else line• [Simple tip to make it better]

                current_section = 'name'• [Easy substitute if something is missing]

            elif 'ingredients:' in line.lower():

                current_section = 'ingredients'✨ **ABOUT THIS FOOD:**

                continue[2-3 simple sentences about how it looks and tastes]

            elif 'instructions:' in line.lower() or 'steps:' in line.lower():

                current_section = 'instructions'IMPORTANT: Use simple words only! No hard cooking words. Write like you're teaching a beginner.

                continue

            elif 'tips:' in line.lower():If the picture doesn't show food, say 'I only help with food pictures! Please show me a picture of food you want to make.'

                current_section = 'tips'"""

                continue

            elif line.startswith(('-', '•', '*')) or line[0].isdigit():# Create prompt template for image+text recipe generation

                # This is a list itemcombined_recipe_template = """Your name is RecipePro, an AI-powered culinary expert who creates delicious and creative recipes.

                cleaned_line = line.lstrip('-•*0123456789. ').strip()

                if current_section == 'ingredients':Analyze this food image and consider the user's input: {query}

                    ingredients.append(cleaned_line)

                elif current_section == 'instructions':Generate a complete recipe with the following EXACT format:

                    instructions.append(cleaned_line)

                elif current_section == 'tips':🍽️ **[CREATE AN ATTRACTIVE, CATCHY RECIPE NAME THAT COMBINES THE IMAGE AND USER REQUEST]**

                    tips.append(cleaned_line)

            else:⏰ **Prep Time:** [X minutes] | **Cook Time:** [X minutes] | **Total Time:** [X minutes]

                # Regular text line👥 **Servings:** [X people]

                if current_section == 'name' and not recipe_name:

                    recipe_name = line🛒 **INGREDIENTS:**

                elif current_section == 'ingredients':• [Quantity] [Ingredient name]

                    ingredients.append(line)• [Quantity] [Ingredient name]

                elif current_section == 'instructions':[Continue for all ingredients with proper measurements, incorporating user's requested ingredients where possible]

                    instructions.append(line)

                elif current_section == 'tips':👨‍🍳 **INSTRUCTIONS:**

                    tips.append(line)1. [Clear, detailed step]

        2. [Clear, detailed step]

        # If we couldn't parse properly, try a simpler approach[Continue with all steps, adapting based on both the image and user preferences]

        if not recipe_name and not ingredients:

            # Look for the first line as recipe name💡 **CHEF'S TIPS:**

            lines = [l.strip() for l in clean_output.split('\n') if l.strip()]• [Helpful tip for better results]

            if lines:• [Substitution suggestion or cooking tip]

                recipe_name = lines[0]

                # Rest as ingredients and instructions✨ **RECIPE DESCRIPTION:**

                for line in lines[1:]:[2-3 sentences describing how this recipe combines what's shown in the image with the user's preferences]

                    if any(word in line.lower() for word in ['cup', 'tbsp', 'tsp', 'oz', 'lb', 'kg', 'g']):

                        ingredients.append(line)If the image doesn't show food or the request is unrelated to cooking, respond with 'I can only create recipes! Please share ingredients or upload a food image.'

                    else:"""

                        instructions.append(line)

        # Create the prompt templates

        return {text_prompt = PromptTemplate(

            'recipe_name': recipe_name or "Delicious Recipe",    template=recipe_template,

            'ingredients': ingredients or ["No ingredients specified"],    input_variables=["query"]

            'instructions': instructions or ["No instructions provided"],)

            'tips': tips or []

        }# Initialize the LLM

        def get_llm():

    except Exception as e:    try:

        print(f"Error parsing recipe: {str(e)}")        API_KEY = os.getenv('GROQ_API_KEY')

        return {        if not API_KEY:

            'recipe_name': "Generated Recipe",            raise ValueError("Groq API key not found. Please set the GROQ_API_KEY environment variable.")

            'ingredients': ["Check the raw output for details"],        

            'instructions': [clean_output],        llm = ChatGroq(

            'tips': []            model="llama-3.1-8b-instant",  # Fast and reliable model

        }            groq_api_key=API_KEY,

            temperature=0.7,

def generate_recipe_from_text(ingredients_list):            max_tokens=2048

    """Generate recipe from ingredients using Groq API"""        )

    try:        return llm

        # Get API key    except Exception as e:

        groq_api_key = os.getenv('GROQ_API_KEY')        raise Exception(f"Error initializing LLM: {str(e)}")

        if not groq_api_key:

            return {# Function to generate recipe images using Hugging Face API

                'error': 'Groq API key not found. Please set GROQ_API_KEY in your environment variables.'def generate_recipe_image(recipe_name, ingredients_list):

            }    """

            Generate an appetizing image of the recipe using Hugging Face's Stable Diffusion

        # Create ingredients string    """

        ingredients_str = ', '.join(ingredients_list)    try:

                # Create a detailed prompt for the image generation

        # Create prompt        prompt = f"A professional, appetizing photo of {recipe_name}, beautifully plated, high quality food photography, studio lighting, garnished, delicious looking, restaurant quality presentation"

        prompt = f"""        

Create a complete recipe using these ingredients: {ingredients_str}        # Hugging Face API endpoint for Stable Diffusion

        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"

Please format your response exactly like this:        

        # You can get a free API token from huggingface.co

Recipe Name: [Name of the dish]        hf_token = os.getenv('HUGGINGFACE_API_TOKEN')

        if not hf_token:

Ingredients:            print("No Hugging Face API token found. Image generation skipped.")

- [ingredient 1 with quantity]            return None

- [ingredient 2 with quantity]            

- [ingredient 3 with quantity]        headers = {"Authorization": f"Bearer {hf_token}"}

        

Instructions:        payload = {

1. [Step 1]            "inputs": prompt,

2. [Step 2]            "parameters": {

3. [Step 3]                "num_inference_steps": 20,

                "guidance_scale": 7.5,

Tips:                "width": 512,

- [Tip 1]                "height": 512

- [Tip 2]            }

        }

Make it practical and delicious!        

"""        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

                

        # Get response from Groq        if response.status_code == 200:

        raw_output = get_groq_response(prompt, groq_api_key)            # Save the image

                    image_data = response.content

        if raw_output:            image = Image.open(io.BytesIO(image_data))

            # Format the output            

            formatted_recipe = format_recipe_output(raw_output)            # Save to media directory

            return formatted_recipe            import uuid

        else:            image_filename = f"generated_{uuid.uuid4().hex[:8]}.jpg"

            return {            image_path = f"media/generated/{image_filename}"

                'error': 'Failed to generate recipe. Please try again.'            

            }            # Create directory if it doesn't exist

                        os.makedirs(os.path.dirname(image_path), exist_ok=True)

    except Exception as e:            

        print(f"Error in generate_recipe_from_text: {str(e)}")            # Save the image

        return {            image.save(image_path, "JPEG", quality=95)

            'error': f'An error occurred: {str(e)}'            

        }            return f"/media/generated/{image_filename}"

        else:

def extract_text_from_image(image_file):            print(f"Image generation failed: {response.status_code}")

    """Extract ingredients from uploaded image"""            return None

    try:            

        # For now, return a placeholder response    except Exception as e:

        # In a real implementation, you would use OCR or image recognition        print(f"Error generating image: {str(e)}")

        return "tomato, onion, garlic, chicken, rice"        return None

    except Exception as e:

        print(f"Error in extract_text_from_image: {str(e)}")# Alternative free image generation using Pollinations API

        return "Error processing image"def generate_recipe_image_pollinations(recipe_name):

    """

def generate_recipe_image(recipe_name):    Generate recipe image using Pollinations.ai (free, no API key required)

    """Generate an image for the recipe using free APIs"""    """

    try:    try:

        # Try Pollinations.ai first (completely free)        # Create a detailed prompt for better image quality

        pollinations_url = f"https://image.pollinations.ai/prompt/{recipe_name.replace(' ', '%20')}%20food%20recipe%20delicious"        prompt = f"professional food photography of {recipe_name}, appetizing, high quality, restaurant style plating, well lit, delicious looking, gourmet presentation"

                

        try:        # URL encode the prompt

            response = requests.get(pollinations_url, timeout=10)        import urllib.parse

            if response.status_code == 200:        encoded_prompt = urllib.parse.quote(prompt)

                print(f"✅ Successfully generated image using Pollinations.ai")        

                return pollinations_url        # Pollinations API URL with better parameters

        except Exception as e:        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=512&height=512&model=flux&enhance=true&nologo=true"

            print(f"❌ Pollinations.ai failed: {str(e)}")        

                print(f"Generating image for: {recipe_name}")

        # Fallback: Try a different Pollinations endpoint        print(f"Image URL: {image_url}")

        try:        

            fallback_url = f"https://image.pollinations.ai/prompt/delicious%20{recipe_name.replace(' ', '%20')}%20food%20photography"        # Return the URL directly (Pollinations serves images directly)

            response = requests.get(fallback_url, timeout=10)        return image_url

            if response.status_code == 200:        

                print(f"✅ Successfully generated image using Pollinations.ai (fallback)")    except Exception as e:

                return fallback_url        print(f"Error with Pollinations image generation: {str(e)}")

        except Exception as e:        return None

            print(f"❌ Pollinations.ai fallback failed: {str(e)}")

        # Alternative image generation using Unsplash API (free with attribution)

        # Final fallback: Use a placeholder food image servicedef generate_recipe_image_unsplash(recipe_name):

        placeholder_url = f"https://source.unsplash.com/800x600/?{recipe_name.replace(' ', ',')},food"    """

        print(f"⚠️ Using Unsplash placeholder image")    Generate recipe image using Unsplash API (free, requires API key)

        return placeholder_url    """

            try:

    except Exception as e:        unsplash_access_key = os.getenv('UNSPLASH_ACCESS_KEY')

        print(f"❌ All image generation methods failed: {str(e)}")        if not unsplash_access_key:

        # Return a generic food image            print("No Unsplash API key found. Skipping Unsplash image generation.")

        return "https://source.unsplash.com/800x600/?food,recipe"            return None

            

def generate_recipe_from_image(image_file):        # Search for food images on Unsplash

    """Generate recipe from uploaded image using Groq API"""        search_query = f"{recipe_name} food"

    try:        url = f"https://api.unsplash.com/search/photos"

        # Get API key        

        groq_api_key = os.getenv('GROQ_API_KEY')        headers = {

        if not groq_api_key:            "Authorization": f"Client-ID {unsplash_access_key}"

            return {        }

                'error': 'Groq API key not found. Please set GROQ_API_KEY in your environment variables.'        

            }        params = {

                    "query": search_query,

        # For now, we'll use a simplified approach since direct image analysis requires special models            "per_page": 5,

        # In production, you would use vision models or OCR to extract ingredients            "orientation": "square"

                }

        # Simulate ingredient extraction (replace with actual OCR/vision API)        

        detected_ingredients = ["chicken", "vegetables", "rice", "spices"]        response = requests.get(url, headers=headers, params=params, timeout=10)

                

        # Generate recipe from detected ingredients        if response.status_code == 200:

        return generate_recipe_from_text(detected_ingredients)            data = response.json()

                    if data['results']:

    except Exception as e:                # Get the first image URL

        print(f"Error in generate_recipe_from_image: {str(e)}")                image_url = data['results'][0]['urls']['regular']

        return {                

            'error': f'An error occurred: {str(e)}'                # Download and save the image

        }                img_response = requests.get(image_url, timeout=30)
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