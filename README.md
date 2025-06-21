# 🧠 RAG Pipeline with LangChain, Ollama, and Chroma

This project implements a Retrieval-Augmented Generation (RAG) system using **LangChain**, **Chroma vector store**, and **Ollama-powered LLMs**. It processes local documents, chunks and embeds them, and enables intelligent question answering over your data.

---

## 🚀 Features

- Hybrid document chunking (semantic + token-based)
- Multi-query retriever with LLM-generated reformulations
- Context-aware RAG chain construction
- Supports `.pdf`, `.txt`, `.md`, and `.docx` files
- Local embeddings via Ollama + ChromaDB
- Conversational Q&A loop via CLI

---

## 📁 Directory Structure

├── data/ # Source documents
├── embedder.py # Embedding and vector store logic
├── file_process.py # Document loading and chunking
├── retriever.py # Multi-query retriever logic
├── main.py # App entry point
├── constants.py # Configurable constants
├── index/ # ChromaDB persistent storage
└── README.md # You're here

---

## ⚙️ Setup & Installation

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd <your-project-folder>

   ```

2. **Install dependencies**
   pip install -r requirements.txt

3. **Install Ollama & run your LLM**
   ollama run _your-llm_

4. **Add documents to your /data folder**

## 🧩 Running the RAG APP

    python main.py

## 🛠 Key Components

- Chunking: Uses unstructured, chunk_by_title, and LangChain's RecursiveCharacterTextSplitter
- Vector Store: ChromaDB with OllamaEmbeddings (e.g., bge-m3)
- Retriever: MultiQueryRetriever for better context coverage
- LLM: ChatOllama using qwen3 model or your prefered llm

## 🧠 Prompt Flow

    User Query
        ↓
    MultiQuery Prompt
        ↓
    Relevant Docs Retrieved
        ↓
    Injected into Prompt
        ↓
    LLM Response

## 📌 Configuration

All critical settings are in constants.py:

    DATA_PATH = "./data"
    MODEL_NAME = "qwen3" or your-llm
    EMBEDDING_MODEL = "bge-m3" or your-embedder
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}

## 📓 Notes

    Ensure Ollama is running before starting the app

    All vector data is stored in index/ and reused across sessions
