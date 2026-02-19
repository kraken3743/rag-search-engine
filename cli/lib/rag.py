__all__ =  ['query_answering', 'doc_summarization', 'doc_citation', 'answer_detailed_question']

from lib.llm import answer_question, summarize_documents , doc_citations, detailed_question_answering
from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def query_answering(query):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=5)
    print(f"Search Results:")
    for res in rrf_results:
        print(f" - {res['title']}")
    rag_results = answer_question(query,rrf_results)
    print("RAG Response:")
    print(rag_results)

def doc_summarization(query, limit=5):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    print(f"Search Results:")
    for res in rrf_results:
        print(f" - {res['title']}")
    rag_results = answer_question(query,rrf_results)
    print("LLM Summary:")
    print(rag_results)

def doc_citation(query, limit=5):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    print(f"Search Results:")
    for res in rrf_results:
        print(f" - {res['title']}")
    rag_results = doc_citations(query, rrf_results)
    print("LLM Answer:")
    print(rag_results)

def answer_detailed_question(query, limit=5):
    movies = load_movies()
    hs = HybridSearch(movies)
    rrf_results = hs.rrf_search(query, k=60, limit=limit)
    print(f"Search Results:")
    for res in rrf_results:
        print(f" - {res['title']}")
    rag_results = detailed_question_answering(query,rrf_results)
    print("Answer:")
    print(rag_results)