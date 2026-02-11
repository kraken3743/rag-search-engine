#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embedding,
    embed_query_text,
    search
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("verify", help="Verify the Embedding Model loads properly")

    embed_parser = subparsers.add_parser("embed_text", help="Encode text with embedding model ")
    embed_parser.add_argument("text", type=str, help="text to be encoded")

    embed_parser = subparsers.add_parser("verify_embeddings", help="verify embedding model loads properly")

    embedquery_parser = subparsers.add_parser("embedquery", help="Encode query with embedding model")
    embedquery_parser.add_argument("query", type=str, help="User query to be encoded")

    search_parser = subparsers.add_parser("search", help="Search for relevant documents based on query")
    search_parser.add_argument("query", type=str, help="User query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")

    args = parser.parse_args()

    match args.command:
        case "search":
            search(args.query, args.limit)
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embedding()
        case "embed_text":
            embed_text(args.text)
        case "verify":
            verify_model()
        
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()