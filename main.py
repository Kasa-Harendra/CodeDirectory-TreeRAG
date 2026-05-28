import argparse
import asyncio
import sys
import time
from treerag.builder import TreeBuilder
from treerag.retriever import TreeRAGRetriever, TreeRAGQA

def handle_index(args):
    """Handle the indexing command."""
    dir_path = args.directory
    index_file = args.index_file
    concurrency = args.concurrency

    start_time = time.time()
    print("=" * 60)
    print(f"TreeRAG: Indexing directory: '{dir_path}'")
    print(f"   Output file: '{index_file}'")
    print(f"   LLM Concurrency Limit: {concurrency}")
    print("=" * 60)

    # Initialize builder
    builder = TreeBuilder(
        root_dir=dir_path,
        cache_path=index_file,
        concurrency_limit=concurrency
    )

    # Run build loop asynchronously
    try:
        root_id = asyncio.run(builder.build())
        duration = time.time() - start_time
        print("\n" + "=" * 60)
        print("Indexing Completed Successfully!")
        print(f"   Root Node ID: {root_id}")
        print(f"   Total Nodes Built: {len(builder.nodes)}")
        print(f"   Time Elapsed: {duration:.2f} seconds")
        print("=" * 60)
    except Exception as e:
        print(f"\nError during indexing: {e}")
        sys.exit(1)


def handle_query(args):
    """Handle the querying command."""
    index_file = args.index_file
    query_str = args.query
    silent = args.silent

    # Load retriever and QA
    try:
        retriever = TreeRAGRetriever(
            index_path=index_file,
            max_depth=4,
            verbose=not silent
        )
        qa_system = TreeRAGQA(retriever=retriever)
    except Exception as e:
        print(f"Error loading index: {e}")
        sys.exit(1)

    start_time = time.time()
    
    # Run the query
    result = qa_system.query(query_str)
    duration = time.time() - start_time

    # Output the final synthesized answer
    print("\n" + "=== Answer ===" + "\n" + "=" * 60)
    print(result["answer"])
    print("=" * 60)
    
    print("\nSources Used:")
    for metadata in result["sources"]:
        print(f"- {metadata['path']} (ID: {metadata['node_id']})")
    
    print(f"\nQuery took: {duration:.2f} seconds")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="TreeRAG: A Vectorless, Flat-Node Directory RAG Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Commands to execute")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a directory to build a flattened Tree JSON")
    index_parser.add_argument("directory", type=str, help="Absolute or relative path to the directory to index")
    index_parser.add_argument("--index-file", type=str, default="treerag_index.json", help="Path to save the JSON index")
    index_parser.add_argument("--concurrency", type=int, default=4, help="Maximum concurrent requests to Ollama")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the TreeRAG index to retrieve context and answer questions")
    query_parser.add_argument("index_file", type=str, help="Path to the built JSON index file")
    query_parser.add_argument("query", type=str, help="User question/prompt to pose to the code/documents")
    query_parser.add_argument("--silent", action="store_true", help="Hide intermediate directory traversal details")

    args = parser.parse_args()

    if args.command == "index":
        handle_index(args)
    elif args.command == "query":
        handle_query(args)


if __name__ == "__main__":
    main()
