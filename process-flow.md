## OpenParse Analysis Steps

#### Analyze flow accuracy:

- PDFMiner extracts text chunks ✓
- Initial chunks become nodes ✓
- Nodes get embedded ✓
- Embeddings used for similarity ✓
- Similar chunks combined ✓

### 1. Text Extraction
- PDFMiner splits PDF into initial text chunks
- Creates TextElements with position/formatting
- TextElements converted to initial Nodes

### 2. Processing Pipeline
- Nodes go through BasicIngestionPipeline or SemanticIngestionPipeline
- Basic: Spatial/layout-based combining
- Semantic: Uses embeddings for intelligent merging

### 3. Semantic Processing
- Each node's text sent to embeddings provider
  - Supports Ollama (local)
  - Cloudflare AI (cloud)
  - OpenAI (cloud)
- Embeddings used to calculate similarity
- Similar adjacent nodes combined if:
  - Similarity above threshold (default 0.6)
  - Combined tokens below limit
  
### 4. Final Output
- Merged nodes form final document
- Embeddings discarded after merging
- Returns clean, semantically grouped text

``` mermaid
graph TD
    A[PDF File] --> B[PDFMiner]
    B --> C[Text Chunks]
    C --> D[Initial Nodes]
    D --> E[Processing Pipeline]
    E --> F[Basic Processing]
    E --> G[Semantic Processing]
    G --> H[Generate Embeddings]
    H --> I[Compare Similarity]
    I --> J[Merge Similar Nodes]
    F --> K[Final Document]
    J --> K
```

#### Key Points
- PDFMiner handles initial text extraction
- Text chunks become nodes for processing
- Embeddings temporarily generated for comparison
- Multiple embedding providers supported
- Final output contains merged text without embeddings
- Processing is configurable (basic vs semantic)
