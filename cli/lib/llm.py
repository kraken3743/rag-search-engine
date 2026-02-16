import os
from dotenv import load_dotenv
from lib.search_utils import PROMPTS_PATH

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

from google import genai
model = "gemini-2.5-flash"

client = genai.Client(api_key=api_key)

def generate_content(prompt, query):
    prompt = prompt.format(query=query)
    prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
    response = client.models.generate_content(model=model, contents=prompt)
    return(response.text)

def correct_spelling(query):
    with open(PROMPTS_PATH/'spelling.md', 'r') as f:
        prompt = f.read()
    return generate_content(prompt, query)

if __name__ == "__main__":
    generate_content()