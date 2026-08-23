from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def create_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
):
    """
    Create the embedding model used for document vectors.
    """

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )