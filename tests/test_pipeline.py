from core.pipeline import process_document


DOCUMENT_PATH = "data/doc/test_document.txt"


def main():

    print("\nProcessing document...")

    result = process_document(
        DOCUMENT_PATH
    )

    print("\nDocument processed successfully!")

    print(
        f"Document ID: "
        f"{result['document_id']}"
    )

    print(
        f"Filename: "
        f"{result['filename']}"
    )

    print(
        f"Original documents: "
        f"{len(result['documents'])}"
    )

    print(
        f"Chunks created: "
        f"{len(result['chunks'])}"
    )

    print("\nTesting retriever...")

    results = result["retriever"].invoke(
        "What is Retrieval Augmented Generation?"
    )

    print(
        f"Retrieved {len(results)} chunk(s)"
    )

    for i, document in enumerate(
        results,
        1,
    ):
        print(
            f"\n--- Result {i} ---"
        )

        print(
            document.page_content[:300]
        )

        print(
            "Metadata:",
            document.metadata,
        )


if __name__ == "__main__":
    main()