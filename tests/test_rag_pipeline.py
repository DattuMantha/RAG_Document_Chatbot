from core.loaders import (
    load_document,
    generate_document_id,
    add_document_metadata,
)

from core.splitter import split_documents

from core.embeddings import create_embedding_model

from core.vectorstore import create_vectorstore

from core.retriever import create_retriever

from core.llm import create_llm
from core.rag import generate_answer

import os


DOCUMENT_PATH = "data/doc/test_document.txt"
DOCUMENT_ID = generate_document_id()


def main():

    print("\n1. Loading document...")

    documents = load_document(DOCUMENT_PATH)

    documents = add_document_metadata(
        documents,
        DOCUMENT_ID,
        "test_document.txt",
    )

    print(f"Loaded {len(documents)} document(s)")


    print("\n2. Splitting document...")

    chunks = split_documents(documents)

    print(f"Created {len(chunks)} chunks")


    print("\n3. Creating embeddings...")

    embeddings = create_embedding_model()

    print("Embedding model ready")


    print("\n4. Creating Chroma vector store...")

    vectorstore = create_vectorstore(
        documents=chunks,
        embedding_function=embeddings,
        document_id=DOCUMENT_ID,
    )

    print("Vector store created")


    print("\n5. Creating retriever...")

    retriever = create_retriever(
        embedding_function=embeddings,
        document_id=DOCUMENT_ID,
        top_k=3,
    )

    print("Retriever ready")


    print("\n6. Testing retrieval...")

    results = retriever.invoke(
        "What is RAG?"
    )

    print(
        f"Retrieved {len(results)} chunks"
    )

    for i, document in enumerate(results, 1):

        print(f"\n--- Result {i} ---")

        print(
            document.page_content
        )

        print(
            "Metadata:",
            document.metadata,
        )


    print("\n7. Creating Gemini LLM...")

    llm = create_llm(
        provider="Google Gemini",
        model="gemini-3.6-flash",
        api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.0,
    )

    print("Gemini LLM ready")


    print("\n8. Testing complete RAG pipeline...")

    question = "What is Retrieval Augmented Generation?"

    result = generate_answer(
        llm=llm,
        retriever=retriever,
        question=question,
    )

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    for document in result["source_documents"]:
        print(
            f"- {document.metadata.get('filename')} "
            f"| Page: "
            f"{document.metadata.get('page_index', 'Unknown')}"
        )


if __name__ == "__main__":
    main()