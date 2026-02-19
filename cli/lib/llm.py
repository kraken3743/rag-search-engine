from google import genai
import os
import json
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

def generate_content(prompt, query, **kwargs):
    prompt = prompt.format(query=query, **kwargs)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text

def llm_judge(query, formatted_results):
    with open(PROMPTS_PATH/'llm_judge.md', 'r') as f:
        prompt = f.read()
    results =  generate_content(prompt, query,formatted_results=formatted_results)
    results = json.loads(results)
    return results

def _rag(query, documents, prompt_fname):
    with open(PROMPTS_PATH/'answer_question.md', 'r') as f:
        prompt = f.read()
    results = generate_content(prompt, query=query, docs=documents)
    return results

def answer_question(query, documents):
    return _rag(query, documents, 'answer_question.md')
    
def summarize_documents(query, documents):
    return _rag(query, documents, 'summarization.md')

def doc_citations(query, documents):
    return _rag(query, documents, 'answer_with_citations.md')

def detailed_question_answering(query, documents):
    return _rag(query, documents, 'answer_question_detailed.md')



