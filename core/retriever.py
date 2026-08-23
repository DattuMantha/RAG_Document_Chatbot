from core.vectorstore import load_vectorstore


def create_retriever(
    embedding_function,
    document_id: str,
    top_k: int = 5,
):
    """
    Create a document-specific retriever.

    The retriever can only search the Chroma
    collection associated with document_id.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    vectorstore = load_vectorstore(
        embedding_function=embedding_function,
        document_id=document_id,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": top_k,
            "fetch_k": max(top_k * 2, 10),
            "lambda_mult": 0.7,
        },
    )

    return retriever