import os
from dotenv import load_dotenv
from lib.search_utils import PROMPTS_PATH

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

from google import genai
model = "gemini-2.5-flash"

client = genai.Client(api_key=api_key)

def augment_prompt(query, type):
    with open(PROMPTS_PATH/f'{type}.md', 'r') as f:
        prompt = f.read()
    return generate_content(prompt, query)

def generate_content(prompt, query):
    prompt = prompt.format(query=query)
    response = client.models.generate_content(model=model, contents=prompt)
    return(response.text)

def correct_spelling(query):
    augment_prompt(query, "spelling")

def rewrite_query(query):
    augment_prompt(query, "rewrite")

def expand_query(query):
    augment_prompt(query, "expand")

if __name__ == "__main__":
    generate_content()