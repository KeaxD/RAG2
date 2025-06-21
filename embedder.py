from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import logging
from constants import EMBEDDING_MODEL
def store_embeddings(chunks, persist_directory="index/"):
    embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma.from_documents(chunks, embedding=embedder, persist_directory=persist_directory)
    logging.info("Vector database created.")
    return db
