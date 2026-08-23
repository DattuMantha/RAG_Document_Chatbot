from pathlib import Path

from langchain_chroma import Chroma

from config.settings import VECTORSTORE_DIR


def get_collection_name(document_id: str) -> str:
    """
    Generate a unique Chroma collection name
    for a document.
    """

    return f"document_{document_id}"


def create_vectorstore(
    documents,
    embedding_function,
    document_id: str,
):
    """
    Create and persist a Chroma vector store
    for a specific document.
    """

    collection_name = get_collection_name(document_id)

    persist_directory = (
        Path(VECTORSTORE_DIR) / document_id
    )

    persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        collection_name=collection_name,
        persist_directory=str(persist_directory),
    )

    return vectorstore


def load_vectorstore(
    embedding_function,
    document_id: str,
):
    """
    Load an existing Chroma vector store
    for a specific document.
    """

    collection_name = get_collection_name(document_id)

    persist_directory = (
        Path(VECTORSTORE_DIR) / document_id
    )

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Vector store not found for document: "
            f"{document_id}"
        )

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=str(persist_directory),
    )