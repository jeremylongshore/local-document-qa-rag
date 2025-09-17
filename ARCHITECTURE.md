# 🏗 NEXUS Architecture Documentation

## System Architecture

```mermaid
graph TB
    subgraph "🎯 User Interface Layer"
        UI[Web Interface<br/>Streamlit]
        API[REST API<br/>FastAPI]
    end

    subgraph "🧠 Intelligence Layer"
        ORCH[Orchestrator<br/>LangChain]
        AGENT[RAG Agent<br/>Autonomous Pipeline]
        CACHE[Cache Layer<br/>LRU/Redis]
    end

    subgraph "📚 Document Processing"
        LOADER[Document Loaders<br/>PDF/TXT/MD/DOCX]
        SPLITTER[Text Splitter<br/>Semantic Chunking]
        EMBED[Embedder<br/>Ollama Embeddings]
    end

    subgraph "💾 Storage Layer"
        VECTOR[Vector Store<br/>ChromaDB]
        DOC[Document Store<br/>File System]
        META[Metadata Store<br/>SQLite]
    end

    subgraph "🤖 AI Model Layer"
        LLM[Local LLM<br/>Ollama Server]
        MODELS[Model Zoo<br/>Llama3/Mistral/Phi3]
    end

    UI --> ORCH
    API --> ORCH
    ORCH --> AGENT
    AGENT --> CACHE
    CACHE --> LOADER
    LOADER --> SPLITTER
    SPLITTER --> EMBED
    EMBED --> VECTOR
    AGENT --> VECTOR
    VECTOR --> LLM
    LLM --> MODELS
    LOADER --> DOC
    AGENT --> META
```

## Component Details

### 🎯 User Interface Layer
- **Streamlit UI**: Interactive chat interface with real-time responses
- **REST API**: (Future) RESTful endpoints for programmatic access

### 🧠 Intelligence Layer
- **Orchestrator**: LangChain-based pipeline coordination
- **RAG Agent**: Autonomous retrieval and generation logic
- **Cache Layer**: Multi-level caching for performance

### 📚 Document Processing
- **Document Loaders**: Extensible format support
- **Text Splitter**: Intelligent chunking with overlap
- **Embedder**: Local embedding generation via Ollama

### 💾 Storage Layer
- **Vector Store**: ChromaDB with HNSW indexing
- **Document Store**: Raw document preservation
- **Metadata Store**: Query history and analytics

### 🤖 AI Model Layer
- **Ollama Server**: Local LLM inference engine
- **Model Zoo**: Support for multiple models

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant UI
    participant NEXUS
    participant VectorDB
    participant LLM

    User->>UI: Submit Query
    UI->>NEXUS: Process Query
    NEXUS->>NEXUS: Check Cache
    alt Cache Miss
        NEXUS->>VectorDB: Semantic Search
        VectorDB-->>NEXUS: Top-k Results
        NEXUS->>LLM: Generate Response
        LLM-->>NEXUS: AI Response
        NEXUS->>NEXUS: Update Cache
    end
    NEXUS-->>UI: Final Response
    UI-->>User: Display Answer
```

## Performance Architecture

```mermaid
graph LR
    subgraph "Optimization Layers"
        A[Query Cache<br/>100ms] --> B[Embedding Cache<br/>500ms]
        B --> C[Vector Search<br/>200ms]
        C --> D[LLM Inference<br/>2-5s]
    end

    subgraph "Parallel Processing"
        E[Doc Loader 1]
        F[Doc Loader 2]
        G[Doc Loader N]
        E --> H[Batch Embedder]
        F --> H
        G --> H
    end
```

## Scalability Design

| Component | Current | Scalable To | Method |
|-----------|---------|-------------|--------|
| Documents | 1K | 100K+ | Incremental indexing |
| Queries/sec | 10 | 1000+ | Horizontal scaling |
| Users | 1 | 50+ | Async processing |
| Vector DB | 1GB | 100GB+ | Sharding |

## Security Architecture

```mermaid
graph TB
    subgraph "Security Layers"
        A[Input Validation] --> B[Path Sanitization]
        B --> C[Query Filtering]
        C --> D[Response Sanitization]
    end

    subgraph "Privacy Controls"
        E[Local Processing]
        F[No External APIs]
        G[Encrypted Storage]
        H[Audit Logging]
    end
```

## Deployment Patterns

### Standalone
```
Single Machine
├── NEXUS Application
├── Ollama Server
├── ChromaDB
└── File Storage
```

### Distributed
```
Load Balancer
├── NEXUS Node 1
├── NEXUS Node 2
└── NEXUS Node N
    ├── Shared ChromaDB
    ├── Shared Ollama Pool
    └── Distributed Cache
```

### Containerized
```yaml
version: '3.8'
services:
  nexus:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./documents:/app/documents
      - ./chroma_db:/app/chroma_db

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Streamlit | Web UI |
| **Backend** | Python 3.9+ | Core logic |
| **Orchestration** | LangChain | Pipeline management |
| **Vector DB** | ChromaDB | Semantic search |
| **LLM Runtime** | Ollama | Model inference |
| **Caching** | LRU/Redis | Performance |
| **Monitoring** | Prometheus | Metrics |
| **Logging** | Structured Logging | Debugging |

## Extension Points

### Adding New Document Types
```python
# loaders/custom_loader.py
class CustomLoader(BaseLoader):
    def load(self, path: str) -> List[Document]:
        # Custom loading logic
        pass
```

### Custom Embedding Models
```python
# embeddings/custom_embedder.py
class CustomEmbedder(BaseEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Custom embedding logic
        pass
```

### Plugin System (Future)
```python
# plugins/custom_plugin.py
class CustomPlugin(NEXUSPlugin):
    def on_query(self, query: str) -> str:
        # Pre-process query
        pass

    def on_response(self, response: str) -> str:
        # Post-process response
        pass
```

## Performance Metrics

```mermaid
graph LR
    subgraph "Key Metrics"
        A[Query Latency<br/>P50: 500ms<br/>P99: 2s]
        B[Throughput<br/>100 docs/min<br/>50 queries/min]
        C[Memory<br/>Base: 500MB<br/>+100MB/1K docs]
        D[Cache Hit<br/>Rate: 70%<br/>TTL: 1hr]
    end
```

## Error Handling

```mermaid
stateDiagram-v2
    [*] --> QueryReceived
    QueryReceived --> Validation
    Validation --> Processing: Valid
    Validation --> ErrorResponse: Invalid
    Processing --> CacheCheck
    CacheCheck --> CacheHit: Found
    CacheCheck --> VectorSearch: Miss
    VectorSearch --> LLMGeneration: Success
    VectorSearch --> Fallback: Error
    LLMGeneration --> Response: Success
    LLMGeneration --> Fallback: Error
    Fallback --> ErrorResponse
    CacheHit --> Response
    Response --> [*]
    ErrorResponse --> [*]
```

---

*NEXUS Architecture v1.0 - Autonomous Document Intelligence System*