from pathlib import Path

from core.loaders import (
    load_document,
    generate_document_id,
    add_document_metadata,
)

from core.splitter import split_documents

from core.embeddings import create_embedding_model

from core.vectorstore import create_vectorstore

from core.retriever import create_retriever


def process_document(
    file_path: str | Path,
    filename: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    top_k: int = 5,
):
    """
    Process a document through the complete ingestion pipeline.

    Pipeline:
        Load
        → Metadata
        → Split
        → Embeddings
        → Chroma Vector Store
        → Retriever

    Args:
        file_path: Actual path of the document being processed.
        filename: Original uploaded filename used for metadata
                  and source citations.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        top_k: Number of chunks retrieved for each query.

    Returns:
        A dictionary containing the document ID,
        processed chunks, vector store, and retriever.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    # --------------------------------------------------
    # 1. Determine the display filename
    # --------------------------------------------------

    # Use the original uploaded filename when available.
    # Fall back to the actual file name for direct pipeline usage.
    display_filename = filename or file_path.name

    # --------------------------------------------------
    # 2. Generate unique document ID
    # --------------------------------------------------

    document_id = generate_document_id()

    # --------------------------------------------------
    # 3. Load document
    # --------------------------------------------------

    documents = load_document(file_path)

    if not documents:
        raise ValueError(
            f"No content could be extracted from: "
            f"{display_filename}"
        )

    # --------------------------------------------------
    # 4. Add metadata
    # --------------------------------------------------

    documents = add_document_metadata(
        documents=documents,
        document_id=document_id,
        filename=display_filename,
    )

    # --------------------------------------------------
    # 5. Split document into chunks
    # --------------------------------------------------

    chunks = split_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(
            f"No text chunks were created from: "
            f"{display_filename}"
        )

    # --------------------------------------------------
    # 6. Create embedding model
    # --------------------------------------------------

    embeddings = create_embedding_model()

    # --------------------------------------------------
    # 7. Create document-specific vector store
    # --------------------------------------------------

    vectorstore = create_vectorstore(
        documents=chunks,
        embedding_function=embeddings,
        document_id=document_id,
    )

    # --------------------------------------------------
    # 8. Create document-specific retriever
    # --------------------------------------------------

    retriever = create_retriever(
        embedding_function=embeddings,
        document_id=document_id,
        top_k=top_k,
    )

    # --------------------------------------------------
    # 9. Return pipeline results
    # --------------------------------------------------

    return {
        "document_id": document_id,
        "filename": display_filename,
        "documents": documents,
        "chunks": chunks,
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "retriever": retriever,
    }