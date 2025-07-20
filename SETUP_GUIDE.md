# 🛠️ Local Document Q&A RAG - Setup Guide

## 📋 Prerequisites Checklist

### 1. **Python 3.9+ Installation**
- [ ] Download from [python.org](https://www.python.org/downloads/)
- [ ] Verify: `python --version`
- [ ] Ensure pip is available: `pip --version`

### 2. **Ollama Installation & Model Download**
- [ ] Download Ollama from [ollama.com](https://ollama.com/download)
- [ ] Install Ollama on your system
- [ ] **CRITICAL**: Download LLM model:
  ```bash
  ollama run llama3
  # OR for smaller model:
  ollama run mistral
  ```
- [ ] Wait for complete download (several GB)
- [ ] Verify model works: Test in Ollama interface

### 3. **GPU Requirements**
- [ ] NVIDIA GPU with 8GB+ VRAM (recommended)
- [ ] CUDA drivers installed
- [ ] Verify GPU detection in Ollama

## 🚀 Step-by-Step Setup

### Step 1: Project Environment
```bash
# Navigate to project directory
cd "C:\Users\jeremy\Local-Document-QA-RAG"

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Create documents folder
mkdir documents
```

### Step 2: Install Dependencies
```bash
# Install required packages
pip install langchain_community
pip install langchain_chroma
pip install langchain
pip install ollama
pip install streamlit
pip install pypdf

# OR install from requirements.txt (when created)
pip install -r requirements.txt
```

### Step 3: Verify Ollama Connection
```bash
# Test Ollama is running
ollama list

# Should show your downloaded models
# Example output:
# NAME     ID       SIZE     MODIFIED
# llama3   abc123   4.7GB    2 days ago
```

### Step 4: Add Sample Documents
- Place test documents in `documents/` folder:
  - PDFs: research papers, manuals, reports
  - TXT files: notes, documentation
  - MD files: markdown documentation

### Step 5: Run the Application
```bash
# Start Streamlit app
streamlit run app.py

# Should open browser at: http://localhost:8501
```

## ⚙️ Configuration Options

### Model Selection
Edit `app.py` to change LLM model:
```python
OLLAMA_MODEL = "llama3"  # Change to: mistral, llama2, etc.
```

### Performance Tuning
```python
# Adjust chunk size for documents
chunk_size=1000  # Larger = more context, slower processing
chunk_overlap=200  # Overlap between chunks

# Retrieval settings
search_kwargs={"k": 3}  # Number of relevant chunks to retrieve
```

### Database Location
```python
CHROMA_DB_PATH = "./chroma_db"  # Local vector database path
```

## 🧪 Testing & Validation

### Basic Functionality Test
1. **Document Loading**: Check console for successful document processing
2. **Embedding Creation**: Verify ChromaDB folder creation
3. **Query Response**: Ask simple questions about your documents
4. **Performance**: Monitor response times and GPU usage

### Sample Test Questions
- "What is the main topic of this document?"
- "Summarize the key points from [specific document]"
- "What are the requirements mentioned in the documentation?"

## 🚨 Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```bash
# Check if Ollama is running
ollama list
# Restart Ollama service if needed
```

**2. Model Not Found**
```bash
# Download the model specified in app.py
ollama run llama3
```

**3. GPU Not Detected**
- Verify CUDA installation
- Check Ollama GPU configuration
- Monitor GPU usage during queries

**4. Documents Not Loading**
- Check file permissions in `documents/` folder
- Verify supported file formats (PDF, TXT, MD)
- Check file encoding (UTF-8 recommended)

**5. Streamlit Port Issues**
```bash
# Use different port if 8501 is busy
streamlit run app.py --server.port 8502
```

## 📊 Performance Optimization

### Hardware Recommendations
- **Minimum**: 16GB RAM, 4GB GPU VRAM
- **Recommended**: 32GB RAM, 8GB+ GPU VRAM
- **Optimal**: High-end GPU (RTX 4080/4090)

### Software Optimization
- Use smaller models (mistral) for faster responses
- Adjust chunk size based on document complexity
- Limit document collection size for initial testing
- Monitor system resources during operation

## 🔄 Maintenance

### Regular Tasks
- [ ] Update Ollama models periodically
- [ ] Clear ChromaDB when documents change significantly
- [ ] Monitor disk space (vector database growth)
- [ ] Update Python dependencies

### Backup Strategy
- Backup `documents/` folder
- Export important chat conversations
- Save custom configurations

---
**Next**: Create the main application file (`app.py`) following the provided code template