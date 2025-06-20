import logging
from pathlib import Path
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from embedder import store_embeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from retriever import create_retriever


DATA_PATH = "./data"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

def check_file_support(file_path:Path):
    if file_path.suffix.lower() not in [".pdf", ".txt", ".md", ".docx"]:
            return False
    return True

def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

def load_and_chunk(file_path: Path):
   
        #File checks 

        if not file_path.is_file():
            logger.debug("  → skipping (not a file)")
            return []

        if not check_file_support(file_path):
            logger.debug(f"Skipping unsupported file type: {file_path}")
            return []

        logger.info(f"Processing file: {file_path}")

        try:
            with open(file_path, "rb") as f:
                elements = partition(file=f, include_page_breaks=True)
            
            docs = []
            for i, e in enumerate(elements):
                if hasattr(e, "text") and e.text.strip():
                    docs.append(Document(
                        page_content=e.text,
                        metadata={
                            "source": str(file_path),
                            "chunk_index": i
                        }
                    ))
                logger.info(f"Extracted {len(docs)} text elements from {file_path}")
            for doc in docs:
                logger.debug(f"[CHUNK] {doc.page_content[:150]!r}")
            return docs                
        except Exception as e:
            logger.error(f"Partition failed for {file_path}: {e}")
            return []
        


def create_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template("""Answer the question based ONLY on the following context:
{context}
Question: {question}
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    def wrapped_chain(inputs):
        result = chain.invoke(inputs)
        docs = retriever.invoke(inputs["query"])
        return {
            "result": result.content if hasattr(result, "content") else str(result),
            "source_documents": docs
        }

    return wrapped_chain



def main(folder:str = DATA_PATH):

    all_docs = []
    
    logger.info(f"Looking for files under {folder}")

    for file_path in Path(folder).rglob("*"):
        docs = load_and_chunk(file_path)
        if docs:
            all_docs.extend(docs)
    
    if not all_docs:
        logger.warning("No documents found to process.")
        return

    logger.info(f"Total extracted documents: {len(all_docs)}")

    chunks = split_documents(all_docs)
    logger.info(f"Total text chunks: {len(chunks)}")

    # Create the vector database
    db = store_embeddings(chunks)

    #Initialize LLM 
    llm = ChatOllama(model= MODEL_NAME, temperature=0.1)

    #Create retriever
    retriever = create_retriever(db,llm)

    #Create chain
    chain = create_chain(retriever,llm)

    print("RAG system is ready. Ask your questions (type 'exit' to quit).")

    while True:
        query = input("\n> ")
        if query.strip().lower() in ["exit", "quit"]:
            break

        response = chain({"query": query})
        print("\nAnswer:")
        print(response["result"])

        print("\nSources:")
        for doc in response["source_documents"]:
            print(f"- {doc.metadata['source']} (chunk {doc.metadata.get('chunk_index')})")


if __name__ == "__main__":
    main()