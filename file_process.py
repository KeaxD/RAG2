from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def hybrid_chunk(file_path: Path):
    """Partition a document, then chunk it semantically and token-efficiently."""
    logger.info(f"Processing {file_path}")

    # Step 1: Partition the file into elements
    with open(file_path, "rb") as f:
        elements = partition(file=f, include_page_breaks=True)

    # Step 2: Chunk the elements by title/semantic groupings
    structured_chunks = chunk_by_title(elements)
    logger.info(f"chunk_by_title() produced {len(structured_chunks)} semantic chunks")

    # Step 3: Wrap each chunk into a LangChain Document
    docs = []
    for i, chunk in enumerate(structured_chunks):
        text = chunk.text.strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "filename": file_path.stem.lower(),
                "chunk_index": i
            }
        ))

    # Step 4: Apply token-aware splitting to keep chunks manageable
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    final_chunks = text_splitter.split_documents(docs)
    logger.info(f"Final chunk count: {len(final_chunks)}")
    return final_chunks