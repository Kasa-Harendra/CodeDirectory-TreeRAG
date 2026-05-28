import os
import json
from typing import Dict, Any, List, Set, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from treerag.config import get_tree_retrieval_llm, count_tokens

# Prompts for Retrieval and QA
NODE_SELECTOR_PROMPT = """You are an expert technical navigation assistant. Your task is to analyze the contents of a directory and select which files or subdirectories are relevant to help answer the user's query.

User Query: {query}

Current Folder: {current_folder_name} (Path: {current_folder_path})
Folder Summary: {current_folder_summary}

Below are the direct sub-elements (files and folders) inside this folder, each with its ID, type, title, and summary:
{children_details}

Based on the query and summaries, choose the sub-elements that are likely to contain the answers or context. You can select multiple items, one item, or none if nothing is relevant.
Your response MUST be a JSON object containing a "selected_ids" key with a list of selected node IDs, and a "reasoning" key explaining why you chose them.

Example output format:
```json
{{
  "reasoning": "The 'api' folder likely contains the routing files, and 'config.py' holds the ports.",
  "selected_ids": ["0002", "0005"]
}}
```

Ensure your response is valid JSON enclosed in ```json ... ``` blocks."""

QA_SYNTHESIS_PROMPT = """You are an expert software engineer and technical assistant. Your task is to answer the user's query using only the technical context retrieved from the directory tree index.

User Query: {query}

Retrieved Technical Context (from relevant files in the directory):
{context}

Provide a comprehensive, high-fidelity, and well-structured technical answer based on the retrieved files. If code blocks are helpful, write clean, commented code. If the answer cannot be found in the context, state that clearly."""


def extract_json(content: str) -> Dict[str, Any]:
    """Helper to extract JSON from a markdown code block."""
    try:
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            json_content = content.strip()
            
        return json.loads(json_content)
    except Exception:
        # Fallback manual search
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(content[start:end])
        except Exception:
            pass
        return {}


class TreeRAGRetriever(BaseRetriever):
    """
    A custom LangChain retriever that navigates a directory structure represented as a
    flat graph of nodes. The LLM decides at each node which children to traverse based
    on the query and node summaries.
    """
    root_dir: str
    nodes: Dict[str, Dict[str, Any]]
    root_id: str
    max_depth: int = 4
    verbose: bool = True
    llm: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, index_path: str, max_depth: int = 4, verbose: bool = True):
        # Load the index
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
            
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        super().__init__(
            root_dir=data["root_dir"],
            nodes=data["nodes"],
            root_id=data["root_id"],
            max_depth=max_depth,
            verbose=verbose,
            llm=get_tree_retrieval_llm()
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Traverse the node hierarchy recursively using LLM decisions and return
        the content of selected files as LangChain Documents.
        """
        retrieved_docs: List[Document] = []
        visited_nodes: Set[str] = set()

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"TreeRAG: Beginning reasoning-based traversal for query: '{query}'")
            print("=" * 60)

        def _traverse(node_id: str, depth: int):
            if node_id in visited_nodes or depth > self.max_depth:
                return
            visited_nodes.add(node_id)

            node = self.nodes.get(node_id)
            if not node:
                return

            # If we hit a file (leaf), retrieve its full content
            if node["type"] == "file":
                file_path = os.path.join(self.root_dir, node["path"])
                if self.verbose:
                    print(f"[Retrieving File]: '{node['path']}' (ID: {node_id})")
                
                try:
                    if os.path.exists(file_path):
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                    else:
                        content = f"[File not found on system: {node['path']}]"
                except Exception as e:
                    content = f"[Error reading file: {e}]"

                doc = Document(
                    page_content=content,
                    metadata={
                        "node_id": node_id,
                        "title": node["title"],
                        "path": node["path"],
                        "type": "file",
                        "summary": node["summary"]
                    }
                )
                retrieved_docs.append(doc)
                return

            # If we hit a directory, decide which children to explore
            if node["type"] == "directory":
                children_ids = node.get("children", [])
                if not children_ids:
                    return

                # Build children descriptions for the prompt
                children_details = []
                for cid in children_ids:
                    cnode = self.nodes.get(cid)
                    if cnode:
                        children_details.append(
                            f"- ID: {cid} | Type: {cnode['type']} | Title: {cnode['title']}\n"
                            f"  Summary: {cnode['summary']}\n"
                        )
                
                children_details_str = "\n".join(children_details)
                
                prompt = NODE_SELECTOR_PROMPT.format(
                    query=query,
                    current_folder_name=node["title"],
                    current_folder_path=node["path"],
                    current_folder_summary=node["summary"],
                    children_details=children_details_str
                )

                if self.verbose:
                    print(f"\n[Exploring Folder]: '{node['path']}' (ID: {node_id})")
                    print(f"   Total sub-elements to evaluate: {len(children_ids)}")

                # Invoke the LLM
                try:
                    response = self.llm.invoke(prompt)
                    parsed_res = extract_json(response.content)
                    
                    reasoning = parsed_res.get("reasoning", "No reasoning provided.")
                    selected_ids = parsed_res.get("selected_ids", [])
                    
                    if self.verbose:
                        print(f"   [LLM Reasoning]: {reasoning}")
                        print(f"   [LLM Decision]: Chose to explore IDs: {selected_ids}")
                        
                    # Recursively traverse selected nodes
                    for cid in selected_ids:
                        if cid in children_ids:
                            _traverse(cid, depth + 1)
                        else:
                            if self.verbose:
                                print(f"   Warning: Selected ID '{cid}' is not a valid child of this folder.")
                except Exception as e:
                    if self.verbose:
                        print(f"   Error during node selection traversal: {e}")

        # Start traversal from the root directory node
        _traverse(self.root_id, 0)
        
        # De-duplicate docs in case paths are visited multiple times
        unique_docs = []
        seen_paths = set()
        for doc in retrieved_docs:
            p = doc.metadata["path"]
            if p not in seen_paths:
                seen_paths.add(p)
                unique_docs.append(doc)

        if self.verbose:
            print(f"\nTraversal complete. Retrieved {len(unique_docs)} relevant file(s).")
            print("=" * 60 + "\n")
            
        return unique_docs


class TreeRAGQA:
    """
    Coordinates context retrieval using TreeRAGRetriever and answers queries
    via local LLM synthesis.
    """
    def __init__(self, retriever: TreeRAGRetriever):
        self.retriever = retriever
        self.llm = get_tree_retrieval_llm()

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Retrieves context using TreeRAG and synthesizes a high-fidelity answer.
        """
        # 1. Retrieve the relevant documents using our reasoning-based tree traversal
        docs = self.retriever._get_relevant_documents(user_query)
        
        if not docs:
            return {
                "answer": "I traversed the directory structure but could not identify any files relevant to your query.",
                "sources": []
            }
            
        # 2. Format the context blocks for the QA synthesizer
        context_blocks = []
        for i, doc in enumerate(docs, 1):
            block = f"--- [{i}] File: {doc.metadata['path']} ---\n{doc.page_content}\n"
            context_blocks.append(block)
            
        context_str = "\n".join(context_blocks)
        
        # 3. Call synthesis prompt
        prompt = QA_SYNTHESIS_PROMPT.format(
            query=user_query,
            context=context_str
        )
        
        if self.retriever.verbose:
            print("Synthesizing final cohesive response...")
            
        response = self.llm.invoke(prompt)
        
        return {
            "answer": response.content.strip(),
            "sources": [doc.metadata for doc in docs]
        }
