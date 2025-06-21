import logging
from pathlib import Path
from embedder import store_embeddings
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from file_process import hybrid_chunk 
from retriever import create_retriever

from constants import SUPPORTED_EXTENSIONS, MODEL_NAME, DATA_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

def is_supported_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS
 


def create_chain(retriever, llm):
    prompt_template = ChatPromptTemplate.from_template("""Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """)

    def wrapped_chain(inputs):
        query = inputs.get("query")
        if not query:
            raise ValueError("Missing 'query' in inputs")

        # Retrieve relevant documents
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Format the final prompt
        filled_prompt = prompt_template.format(context=context, question=query)

        # Generate response
        result = llm.invoke(filled_prompt, think=False)

        return {
            "result": getattr(result, "content", str(result)),
            "source_documents": docs
        }

    return wrapped_chain




def main(folder:str = DATA_PATH):

    all_docs = []
    
    logger.info(f"Looking for files under {folder}")

    # Recursively find, load, and chunk documents inside the specified folder
    for file_path in Path(folder).rglob("*"):

        #Skip directories and unsupported files
        if not file_path.is_file() or not is_supported_file(file_path):
            continue

        try:
            chunks = hybrid_chunk(file_path)
            if chunks:
                all_docs.extend(chunks)
        except Exception as e:
            logger.error(f"Chunking failed for {file_path}: {e}")

    
    # If no documents were found, log a warning and exit
    if not all_docs:    
        logger.warning("No valid documents found to process.")
        return

    logger.info(f"Total chunks collected: {len(all_docs)}")

    # Create the vector database using the chunks 
    db = store_embeddings(all_docs)

    #Initialize LLM 
    llm = ChatOllama(model= MODEL_NAME, temperature=0.4)

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