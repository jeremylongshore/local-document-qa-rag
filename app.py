import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# --- Configuration ---
# Make sure this matches the model you downloaded via `ollama run <model_name>`
OLLAMA_MODEL = "llama3" # e.g., "llama3", "mistral", "llama2-uncensored"
DOCUMENTS_DIR = "documents"
CHROMA_DB_PATH = "./chroma_db" # Local directory to store vector DB

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Local Document Q&A Chatbot", layout="centered")
st.title("📚 Local Document Q&A Chatbot")
st.markdown("Ask questions about your documents! All processing happens locally.")

# --- Document Loading and Processing ---
@st.cache_resource # Cache this function to avoid re-running on every interaction
def setup_rag_pipeline():
    try:
        # Check if ChromaDB already exists to avoid reprocessing
        if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
            st.info("Loading existing vector database...")
            embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
            vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            llm = Ollama(model=OLLAMA_MODEL)
        else:
            # 1. Load documents
            st.info(f"Loading documents from {DOCUMENTS_DIR}/...")
            documents = []
            for filename in os.listdir(DOCUMENTS_DIR):
                file_path = os.path.join(DOCUMENTS_DIR, filename)
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                elif filename.endswith((".txt", ".md")):
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                # Add more loaders for other file types as needed (e.g., CSVLoader, DOCXLoader)
            
            if not documents:
                st.warning(f"No documents found in '{DOCUMENTS_DIR}'. Please add some files to proceed.")
                return None, None

            # 2. Split documents into chunks
            st.info("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)

            # 3. Create embeddings and store in ChromaDB
            st.info(f"Creating embeddings and storing in ChromaDB at {CHROMA_DB_PATH}...")
            embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DB_PATH)
            
            # 4. Initialize the Retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

            # 5. Initialize the Local LLM
            llm = Ollama(model=OLLAMA_MODEL)

        # 6. Define the RAG Prompt Template
        template = """
        You are an AI assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 7. Build the RAG Chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        st.success("RAG pipeline setup complete! You can now ask questions.")
        return rag_chain, vectorstore # Return vectorstore to manage persistence
    
    except Exception as e:
        if "ConnectionError" in str(type(e)) or "connection" in str(e).lower():
            st.error(f"❌ Cannot connect to Ollama. Please ensure Ollama is running and the model '{OLLAMA_MODEL}' is available.")
            st.info("To fix this:\n1. Start Ollama: `ollama serve`\n2. Download model: `ollama run {OLLAMA_MODEL}`")
        else:
            st.error(f"❌ Error setting up RAG pipeline: {str(e)}")
        return None, None

# --- Main Application Flow ---
rag_chain, vectorstore_instance = setup_rag_pipeline()

if rag_chain:
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents:"):
        # Input validation
        if not prompt.strip():
            st.warning("Please enter a valid question.")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = rag_chain.invoke(prompt)
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"❌ Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Optional: Button to clear chat history and re-setup (e.g., if documents change)
    if st.sidebar.button("Clear Chat & Re-index Documents"):
        st.session_state.messages = []
        # Delete ChromaDB to force re-indexing
        if os.path.exists(CHROMA_DB_PATH):
            import shutil
            shutil.rmtree(CHROMA_DB_PATH)
        st.cache_resource.clear() # Clear Streamlit cache
        st.rerun() # Rerun the app to re-setup RAG pipeline
else:
    st.error("Please add documents to the 'documents/' folder and restart the application.")