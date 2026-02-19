import argparse
from lib.rag import query_answering, doc_summarization, doc_citation

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    qa_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    qa_parser.add_argument("query", type=str, help="Search query for RAG")

    sum_parser = subparsers.add_parser(
        "summarize", help="Perform RAG (search + summarize)"
    )
    sum_parser.add_argument("query", type=str, help="Search query for RAG")
    sum_parser.add_argument("--limit", type=int, default=5, help="docs to be summarize")

    cit_parser = subparsers.add_parser(
        "citations", help="Perform RAG (search + summarize + citation)"
    )
    cit_parser.add_argument("query", type=str, help="Search query for RAG")
    cit_parser.add_argument("--limit", type=int, default=5, help="docs to be summarize w/ citations")   

    args = parser.parse_args()

    match args.command:
        case "rag":
            query_answering(args.query)
        case "summarize":
            doc_summarization(args.query, args.limit)
        case "citations":
            doc_citation(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()