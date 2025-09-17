#!/bin/bash

# NEXUS Quick Installer Script
# One-line installation: curl -sSL https://raw.githubusercontent.com/jeremylongshore/nexus-rag/main/install.sh | bash

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║                                                      ║"
echo "║        🧠 NEXUS - Local RAG AI Agent                ║"
echo "║     Autonomous Document Intelligence System         ║"
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to print colored messages
print_status() { echo -e "${BLUE}[*]${NC} $1"; }
print_success() { echo -e "${GREEN}[✓]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }

# Check system requirements
print_status "Checking system requirements..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    echo "Please install Python 3.9+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
print_success "Python $PYTHON_VERSION detected"

# Check Git
if ! command -v git &> /dev/null; then
    print_error "Git is required but not installed."
    echo "Please install Git and try again."
    exit 1
fi
print_success "Git detected"

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    print_warning "Ollama not detected. Installing Ollama..."

    # Install Ollama based on OS
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
        print_success "Ollama installed successfully"
    else
        print_error "Automatic Ollama installation not supported for your OS."
        echo "Please install Ollama manually from: https://ollama.ai/download"
        exit 1
    fi
else
    print_success "Ollama detected"
fi

# Clone repository if not in NEXUS directory
if [ ! -f "app.py" ]; then
    print_status "Cloning NEXUS repository..."
    git clone https://github.com/jeremylongshore/nexus-rag.git nexus-rag
    cd nexus-rag
    print_success "Repository cloned"
else
    print_success "Already in NEXUS directory"
fi

# Create virtual environment
print_status "Creating Python virtual environment..."
python3 -m venv venv
print_success "Virtual environment created"

# Activate virtual environment
print_status "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip -q
print_success "Pip upgraded"

# Install requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt -q
print_success "Dependencies installed"

# Create documents directory if it doesn't exist
if [ ! -d "documents" ]; then
    print_status "Creating documents directory..."
    mkdir documents

    # Create sample document
    cat > documents/sample.txt << 'EOF'
Welcome to NEXUS!

This is a sample document to get you started. NEXUS is an autonomous RAG (Retrieval-Augmented Generation) agent that can answer questions about your documents.

Key Features:
- 100% local processing - no data leaves your machine
- Support for PDF, TXT, MD, and more formats
- Intelligent semantic search using vector embeddings
- Context-aware responses using local LLMs
- Zero API costs - runs entirely on your hardware

To get started:
1. Add your documents to the 'documents' folder
2. Ask questions about your documents
3. NEXUS will provide accurate, contextual answers

Enjoy using NEXUS for private, intelligent document analysis!
EOF
    print_success "Documents directory created with sample"
fi

# Download Ollama model if not already present
print_status "Checking Ollama models..."
if ! ollama list | grep -q "llama3"; then
    print_status "Downloading llama3 model (this may take a few minutes)..."
    ollama pull llama3
    print_success "Model downloaded"
else
    print_success "llama3 model already available"
fi

# Start Ollama service if not running
print_status "Starting Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > /dev/null 2>&1 &
    sleep 3
    print_success "Ollama service started"
else
    print_success "Ollama service already running"
fi

# Success message
echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}       🎉 NEXUS Installation Complete! 🎉${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}\n"

echo -e "${BLUE}To start NEXUS:${NC}"
echo -e "  ${YELLOW}cd nexus-rag${NC} (if not already there)"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo -e "  ${YELLOW}streamlit run app.py${NC}\n"

echo -e "${BLUE}NEXUS will open in your browser at:${NC}"
echo -e "  ${GREEN}http://localhost:8501${NC}\n"

echo -e "${BLUE}Add your documents to:${NC}"
echo -e "  ${YELLOW}nexus-rag/documents/${NC}\n"

# Optional: Auto-start NEXUS
read -p "Would you like to start NEXUS now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Starting NEXUS..."
    echo -e "\n${GREEN}Opening NEXUS in your browser...${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop NEXUS${NC}\n"
    streamlit run app.py
fi