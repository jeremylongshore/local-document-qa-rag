# 📄 Local Document Q&A RAG Chatbot

A privacy-first document question-answering system that runs entirely on your local machine using GPU acceleration.

## 🚀 Features

- **100% Local Processing** - No data leaves your machine
- **GPU Accelerated** - Fast inference with local LLM
- **Multi-Format Support** - PDFs, TXT, MD files
- **Chat Interface** - Interactive document Q&A
- **Vector Database** - Efficient document retrieval
- **Cost-Free** - No API fees or cloud dependencies

## 🛠️ Technology Stack

- **Ollama** - Local LLM serving (Llama 3, Mistral)
- **LangChain** - RAG pipeline framework
- **Streamlit** - Web interface
- **ChromaDB** - Vector database
- **PyPDF** - Document processing

## 📦 Installation

1. **Install Ollama**
   ```bash
   # Download from https://ollama.com/download
   # Then download a model:
   ollama run llama3
   ```

2. **Setup Python Environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

3. **Add Documents**
   - Place your documents in the `documents/` folder
   - Supported formats: PDF, TXT, MD

4. **Run Application**
   ```bash
   streamlit run app.py
   ```

## 🎯 Usage

1. Start the application: `streamlit run app.py`
2. Wait for document processing (first run only)
3. Ask questions about your documents in the chat interface
4. Get instant answers based on document content

## ⚙️ Configuration

Edit `app.py` to customize:

```python
OLLAMA_MODEL = "llama3"  # Change model
chunk_size=1000          # Document chunk size
search_kwargs={"k": 3}   # Number of retrieved chunks
```

## 📊 System Requirements

- **Minimum**: 16GB RAM, 4GB GPU VRAM
- **Recommended**: 32GB RAM, 8GB+ GPU VRAM
- **Python**: 3.9+
- **GPU**: NVIDIA with CUDA support

## 🔧 Troubleshooting

**Ollama Connection Issues**
```bash
ollama list  # Check available models
ollama run llama3  # Re-download if needed
```

**Document Loading Problems**
- Check file permissions in `documents/` folder
- Verify file formats (PDF, TXT, MD only)
- Ensure UTF-8 encoding for text files

**Performance Issues**
- Use smaller models (mistral vs llama3)
- Reduce chunk size for faster processing
- Monitor GPU memory usage

## 📁 Project Structure

```
Local-Document-QA-RAG/
├── documents/           # Your documents go here
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── chroma_db/          # Vector database (auto-created)
├── PROJECT_OVERVIEW.md # Project details
├── SETUP_GUIDE.md      # Detailed setup instructions
└── README.md           # This file
```

## 🎯 Use Cases

- **Research Assistant** - Query academic papers and documents
- **Corporate Knowledge Base** - Private document search
- **Legal Document Analysis** - Compliance-safe document Q&A
- **Technical Documentation** - Interactive manual assistant
- **Personal Document Archive** - Search personal files

## 💼 Portfolio Value

This project demonstrates:
- **Local AI/LLM expertise** (high-demand skill)
- **RAG implementation** (modern AI technique)
- **Privacy-first solutions** (enterprise requirement)
- **Full-stack AI development** (comprehensive skill set)
- **Cost-effective AI** (no ongoing operational costs)

## 🔄 Maintenance

- Update Ollama models periodically
- Clear ChromaDB when documents change significantly
- Monitor disk space (vector database growth)
- Backup important documents and configurations

## 📝 License

This project is for educational and portfolio purposes.

---

**Built with ❤️ for privacy-conscious AI applications**