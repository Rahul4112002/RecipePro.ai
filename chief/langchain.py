import os
import base64
from PIL import Image
import io
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
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
recipe_template = """Your name is RecipePro, an AI-powered culinary expert specializing in quick and easy recipes that can be prepared in 5 minutes.

Generate the recipe in the following format:

**Recipe Name**
- Ingredients with Quantity
**Instructions**
**Tips** (if required)

If the user's query is not related to recipes or if you don't know the answer, respond with 'I don't know the answer.'

User's query: {query}
"""

# Create prompt template for image-based recipe generation
image_recipe_template = """Your name is RecipePro, an AI-powered culinary expert specializing in quick and easy recipes that can be prepared in 5 minutes.

Generate the recipe in the following format:

**Recipe Name**
- Ingredients with Quantity
**Instructions**
**Tips** (if required)

If the user's query is not related to recipes or if you don't know the answer, respond with 'I don't know the answer.'

Look at this food image and generate a recipe based on what you see.
"""

# Create prompt template for image+text recipe generation
combined_recipe_template = """Your name is RecipePro, an AI-powered culinary expert specializing in quick and easy recipes that can be prepared in 5 minutes.

Generate the recipe in the following format:

**Recipe Name**
- Ingredients with Quantity
**Instructions**
**Tips** (if required)

If the user's query is not related to recipes or if you don't know the answer, respond with 'I don't know the answer.'

Look at this food image and consider the following user ingredients/request: {query}
Generate a recipe based on what you see in the image and the user's input.
"""

# Create the prompt templates
text_prompt = PromptTemplate(
    template=recipe_template,
    input_variables=["query"]
)

# Initialize the LLM
def get_llm():
    try:
        API_KEY = os.getenv('GOOGLE_API_KEY')
        if not API_KEY:
            raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=API_KEY,
            temperature=0.7
        )
        return llm
    except Exception as e:
        raise Exception(f"Error initializing LLM: {str(e)}")

# Function to convert uploaded image to data URI for Gemini
def image_to_data_uri(image_file):
    try:
        # Read the image file
        image = Image.open(image_file)
        
        # Convert to bytes
        image_byte_arr = io.BytesIO()
        image.save(image_byte_arr, format=image.format or 'JPEG')
        image_bytes = image_byte_arr.getvalue()
        
        # Convert to base64 and create data URI
        encoded = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = f"image/{image.format.lower() if image.format else 'jpeg'}"
        data_uri = f"data:{mime_type};base64,{encoded}"
        
        return data_uri
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

# Create formatter for output
def format_recipe_output(recipe_text):
    recipe_text_lines = recipe_text.split('\n')
    formatted_text = []
    
    for line in recipe_text_lines:
        if line.startswith("**") and line.endswith("**"):  # For bold titles
            formatted_text.append(f"<b>{line.strip('**')}</b>")
        else:
            formatted_text.append(line)
    
    return "\n".join(formatted_text)

# Main function
def ask(recipe_message, image_file=None):
    try:
        # Get the LLM
        llm = get_llm()
        
        # Handle image-based or text-based recipe generation
        if image_file:
            try:
                # Convert image to data URI format for Gemini
                data_uri = image_to_data_uri(image_file)
                
                # For multimodal input (image + text if available)
                from langchain_core.messages import HumanMessage
                
                if recipe_message:
                    # Both image and text are provided
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": combined_recipe_template.format(query=recipe_message)},
                            {"type": "image_url", "image_url": data_uri}
                        ]
                    )
                else:
                    # Only image is provided
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": image_recipe_template},
                            {"type": "image_url", "image_url": data_uri}
                        ]
                    )
                
                # Get response directly from LLM
                response = llm.invoke([message])
                recipe_text = response.content
                
            except Exception as img_error:
                return f"Error processing image: {str(img_error)}"
        else:
            # For text-based queries, use the LangChain
            chain = LLMChain(
                llm=llm,
                prompt=text_prompt
            )
            
            # Execute the chain
            response = chain.run(query=recipe_message)
            recipe_text = response
        
        # Format the output for display
        return format_recipe_output(recipe_text)
        
    except Exception as e:
        if "429" in str(e):
            return "Error: You've reached the Google API quota limit. Please try again later or check your API usage limits in the Google Cloud Console."
        else:
            return f"Error: {str(e)}"