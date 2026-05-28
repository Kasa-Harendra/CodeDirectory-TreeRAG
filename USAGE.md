# TreeRAG

### Building Tree Index
```
python main.py index ./mock_project --index-file treerag_index.json --concurrency 4
```

### Quering from Generated Index 
```
python main.py query treerag_index.json "What is the important settings to be configured"
```