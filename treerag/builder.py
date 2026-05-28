import os
import hashlib
import json
import asyncio
from typing import Dict, Any, List, Set, Tuple
from treerag.config import (
    get_tree_build_llm,
    IGNORED_DIRS,
    BINARY_EXTENSIONS,
    TEXT_EXTENSIONS,
    count_tokens,
)

# Prompts for LLM
FILE_SUMMARY_PROMPT = """You are an expert technical writer and software architect. Your task is to analyze the source code or document file provided and generate a precise, high-fidelity, and exhaustive summary. Focus on:
1. The primary purpose and role of this file in the project.
2. Key classes, functions, variables, or configurations defined, along with their core behaviors.
3. Main inputs, outputs, side effects, and dependencies.
4. Any architectural patterns or design choices.
Ensure your summary is concise but dense with technical details, avoiding generic phrases.

File Name: {filename}
Relative Path: {path}
File Contents:
{contents}"""

DIR_SUMMARY_PROMPT = """You are an expert technical architect. Your task is to analyze the contents and structures of a directory and generate a highly accurate, cohesive summary that captures the overall purpose of the folder and how its contents work together.
You are given the details and summaries of all files and subdirectories directly within this folder.

Folder Name: {foldername}
Relative Path: {path}
Direct Contents of this folder:
{children_metadata_and_summaries}

Provide a dense, structured summary detailing:
1. The architectural role of this folder.
2. The main themes, subsystems, or responsibilities represented by the files and subfolders inside.
3. How these files interact with each other (e.g. key entrypoints vs utilities).
Do not just list the files; provide a unified overview."""


def get_file_md5(file_path: str) -> str:
    """Calculate the MD5 hash of a file's content."""
    hasher = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return ""


class TreeBuilder:
    def __init__(self, root_dir: str, cache_path: str = None, concurrency_limit: int = 4):
        self.root_dir = os.path.abspath(root_dir)
        self.cache_path = cache_path or os.path.join(self.root_dir, "treerag_index.json")
        self.concurrency_limit = concurrency_limit
        
        self.nodes: Dict[str, Dict[str, Any]] = {}  # Global flat map of node_id -> node
        self.path_to_id: Dict[str, str] = {}  # Map of relative_path -> node_id
        self.id_counter = 1
        
        # Cache of previous run
        self.cache: Dict[str, Any] = {}
        self.cache_file_hits: Set[str] = set()  # Tracks relative paths that had cache hits
        self.cache_dir_hits: Set[str] = set()   # Tracks relative paths of dirs that had cache hits
        
        self.load_cache()

    def get_next_id(self) -> str:
        """Generate a sequential padded node ID."""
        node_id = str(self.id_counter).zfill(4)
        self.id_counter += 1
        return node_id

    def load_cache(self):
        """Load index cache from previous run if exists."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                print(f"Loaded existing index cache from {self.cache_path}")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}. Starting fresh.")
                self.cache = {}

    def save_index(self, root_id: str):
        """Save the flat nodes map to a JSON file."""
        output = {
            "root_id": root_id,
            "root_dir": self.root_dir,
            "nodes": self.nodes
        }
        output_dir = os.path.dirname(self.cache_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Tree index saved successfully to {self.cache_path}")

    def is_ignored(self, path: str) -> bool:
        """Check if a path or any of its parent segments are ignored."""
        parts = os.path.relpath(path, self.root_dir).split(os.sep)
        for part in parts:
            if part in IGNORED_DIRS or part.startswith("."):
                return True
        return False

    async def summarize_file(
        self,
        file_path: str,
        rel_path: str,
        file_hash: str,
        semaphore: asyncio.Semaphore,
        llm: Any,
    ) -> Dict[str, Any]:
        """Summarize a single text file concurrently."""
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # Check cache
        cached_nodes = self.cache.get("nodes", {})
        cached_node = None
        for cid, cnode in cached_nodes.items():
            if cnode.get("path") == rel_path and cnode.get("type") == "file":
                if cnode.get("hash") == file_hash:
                    cached_node = cnode
                    break
        
        if cached_node:
            self.cache_file_hits.add(rel_path)
            # Create a clean copy of the cached node but assign a new node_id later or preserve it
            return {
                "title": filename,
                "type": "file",
                "path": rel_path,
                "file_size_bytes": file_size,
                "hash": file_hash,
                "summary": cached_node["summary"],
                "_cached": True
            }

        # Cache miss: Read and summarize
        async with semaphore:
            print(f"Summarizing file (Cache Miss): {rel_path} ...")
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                # Limit tokens by truncating extremely long files to first 25,000 characters
                if len(content) > 25000:
                    content = content[:25000] + "\n\n[Content truncated for summary generation...]"
                
                prompt = FILE_SUMMARY_PROMPT.format(
                    filename=filename,
                    path=rel_path,
                    contents=content
                )
                
                # Call Ollama asynchronously using langchain ChatOllama
                response = await llm.ainvoke(prompt)
                summary = response.content.strip()
                
            except Exception as e:
                print(f"Error summarizing {rel_path}: {e}")
                summary = f"Error generating summary: {e}"
                
            return {
                "title": filename,
                "type": "file",
                "path": rel_path,
                "file_size_bytes": file_size,
                "hash": file_hash,
                "summary": summary,
                "_cached": False
            }

    async def build(self) -> str:
        """
        Build the entire flattened tree structure.
        Returns the root_id.
        """
        print(f"Scanning directory structure: {self.root_dir}")
        llm = get_tree_build_llm()
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        
        # 1. Collect all files and folders
        all_files: List[Tuple[str, str]] = []  # List of (absolute_path, relative_path)
        all_dirs: List[Tuple[str, str]] = []   # List of (absolute_path, relative_path)
        
        for root, dirs, files in os.walk(self.root_dir):
            # Skip ignored directories in place
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
            
            for d in dirs:
                dir_path = os.path.join(root, d)
                rel_path = os.path.relpath(dir_path, self.root_dir)
                all_dirs.append((dir_path, rel_path))
                
            for f in files:
                if f.startswith("."):
                    continue
                file_path = os.path.join(root, f)
                rel_path = os.path.relpath(file_path, self.root_dir)
                all_files.append((file_path, rel_path))

        # Include root directory
        root_rel_path = "."
        all_dirs.append((self.root_dir, root_rel_path))

        print(f"Found {len(all_files)} files and {len(all_dirs)} directories (excluding ignored items).")

        # 2. Separate text files for async summarization and process binaries
        summarization_tasks = []
        file_nodes_draft: Dict[str, Dict[str, Any]] = {}  # rel_path -> node draft

        for file_path, rel_path in all_files:
            _, ext = os.path.splitext(rel_path.lower())
            
            # Check if binary/media file
            if ext in BINARY_EXTENSIONS or (ext not in TEXT_EXTENSIONS and not rel_path.endswith((".dockerfile", "dockerfile"))):
                file_size = os.path.getsize(file_path)
                file_nodes_draft[rel_path] = {
                    "title": os.path.basename(file_path),
                    "type": "file",
                    "path": rel_path,
                    "file_size_bytes": file_size,
                    "hash": "",
                    "summary": f"Binary or media file ({ext}). Content skipped in detailed RAG traversal.",
                    "_cached": True
                }
            else:
                # Text file - calculate hash
                file_hash = get_file_md5(file_path)
                task = self.summarize_file(file_path, rel_path, file_hash, semaphore, llm)
                summarization_tasks.append((rel_path, task))

        # Run file summarizations concurrently
        if summarization_tasks:
            print(f"Processing {len(summarization_tasks)} text files concurrently...")
            rel_paths, tasks = zip(*summarization_tasks)
            summarized_results = await asyncio.gather(*tasks)
            
            for rel_path, node_data in zip(rel_paths, summarized_results):
                file_nodes_draft[rel_path] = node_data

        # Register File Nodes and assign IDs
        for rel_path, node_data in file_nodes_draft.items():
            node_id = self.get_next_id()
            cached_flag = node_data.pop("_cached", False)
            
            node = {
                "node_id": node_id,
                **node_data
            }
            self.nodes[node_id] = node
            self.path_to_id[rel_path] = node_id

        print(f"File summarization complete. Cache hits: {len(self.cache_file_hits)}/{len(all_files)}")

        # 3. Post-Order (Bottom-Up) traversal to build and summarize directory nodes
        # Sort directories by path depth (longest first, ensuring subdirectories are processed before parent directories)
        sorted_dirs = sorted(all_dirs, key=lambda x: len(x[1].split(os.sep)), reverse=True)

        for dir_path, rel_path in sorted_dirs:
            dirname = os.path.basename(dir_path) if rel_path != "." else os.path.basename(self.root_dir)
            
            # List immediate children of this directory
            try:
                items = os.listdir(dir_path)
            except Exception:
                items = []

            children_ids: List[str] = []
            all_children_cached = True

            for item in items:
                child_path = os.path.join(dir_path, item)
                child_rel = os.path.relpath(child_path, self.root_dir)
                
                # Retrieve child node id
                child_id = self.path_to_id.get(child_rel)
                if child_id:
                    children_ids.append(child_id)
                    child_node = self.nodes[child_id]
                    # Check if this child had a cache hit
                    if child_node["type"] == "file":
                        if child_rel not in self.cache_file_hits:
                            all_children_cached = False
                    elif child_node["type"] == "directory":
                        if child_rel not in self.cache_dir_hits:
                            all_children_cached = False

            # Assign ID to this directory node
            dir_node_id = self.get_next_id()
            self.path_to_id[rel_path] = dir_node_id

            # Check if directory node can be loaded from cache
            cached_dir_node = None
            if all_children_cached and self.cache:
                cached_nodes = self.cache.get("nodes", {})
                for cid, cnode in cached_nodes.items():
                    if cnode.get("path") == rel_path and cnode.get("type") == "directory":
                        cached_dir_node = cnode
                        break

            if cached_dir_node:
                self.cache_dir_hits.add(rel_path)
                dir_summary = cached_dir_node["summary"]
            else:
                # Cache miss: Summarize the directory using children summaries
                children_summaries = []
                for cid in children_ids:
                    child_node = self.nodes[cid]
                    children_summaries.append(
                        f"- Name: {child_node['title']}\n"
                        f"  Type: {child_node['type']}\n"
                        f"  Summary: {child_node['summary']}\n"
                    )
                
                if children_summaries:
                    children_info_str = "\n".join(children_summaries)
                    prompt = DIR_SUMMARY_PROMPT.format(
                        foldername=dirname,
                        path=rel_path,
                        children_metadata_and_summaries=children_info_str
                    )
                    
                    print(f"Summarizing directory: {rel_path} ...")
                    try:
                        response = await llm.ainvoke(prompt)
                        dir_summary = response.content.strip()
                    except Exception as e:
                        print(f"Error summarizing directory {rel_path}: {e}")
                        dir_summary = f"Error generating folder summary: {e}"
                else:
                    dir_summary = "Empty directory."

            # Register Directory Node
            self.nodes[dir_node_id] = {
                "node_id": dir_node_id,
                "title": dirname,
                "type": "directory",
                "path": rel_path,
                "summary": dir_summary,
                "children": children_ids
            }

        print(f"Directory summarization complete. Cache hits: {len(self.cache_dir_hits)}/{len(all_dirs)}")

        # Root directory is at rel_path "."
        root_id = self.path_to_id["."]
        self.save_index(root_id)
        
        return root_id
