#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embedding,
    embed_query_text,
    search,
    chunk_text,
    chunk_text_semantic,
    semantic_chunking,
    embed_chunks,
    search_chunked
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

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for relevant documents based on query")
    search_chunked_parser.add_argument("query", type=str, help="User query to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of search results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into smaller pieces") 
    chunk_parser.add_argument("text", type=str, help="Document to be chunked") 
    chunk_parser.add_argument("--overlap", type=int, default=50, help="Number of overlapping words between chunks")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Number of words per chunk")

    chunk_text_semantic_parser = subparsers.add_parser("semantic_chunk", help="Chunk text into semantically coherent pieces")
    chunk_text_semantic_parser.add_argument("text", type=str, help="Document to be chunked") 
    chunk_text_semantic_parser.add_argument("--overlap", type=int, default=0, help="Number of overlapping words between chunks")
    chunk_text_semantic_parser.add_argument("--max-chunk-size", type=int, default=4, help="Number of words per chunk")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed chunks of documents")


    args = parser.parse_args()

    match args.command:
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case "embed_chunks": 
            embed_chunks()
        case "semantic_chunk":
            chunk_text_semantic(args.text, args.overlap, args.max_chunk_size)
        case "chunk":
            chunk_text(args.text, args.overlap, args.chunk_size)
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