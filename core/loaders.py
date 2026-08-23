from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
}


def generate_document_id() -> str:
    """Generate a unique ID for an uploaded document."""
    return uuid4().hex


def load_document(file_path: str | Path):
    """
    Load a supported document and return LangChain Documents.
    """

    file_path = Path(file_path)
    

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    if file_path.stat().st_size == 0:
        raise ValueError(
            f"Document is empty: {file_path.name}"
        )

    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".pdf":
        loader = PyPDFLoader(str(file_path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(file_path))

    elif extension == ".txt":
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return loader.load()

def add_document_metadata(
    documents,
    document_id: str,
    filename: str,
):
    """
    Add document-level metadata to every loaded page.
    """

    file_type = Path(filename).suffix.lower().replace(".", "")

    for index, document in enumerate(documents):

        existing_page = document.metadata.get(
            "page",
            index,
        )

        document.metadata.update(
            {
                "document_id": document_id,
                "filename": filename,
                "file_type": file_type,
                "page_index": existing_page,
                "source": filename,

            }
        )

    return documents