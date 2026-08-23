from pathlib import Path

from core.loaders import (
    load_document,
    generate_document_id,
    add_document_metadata,
)


TEST_DIRECTORY = Path("data/doc")


def test_document(file_path: Path):
    print(f"\nTesting: {file_path.name}")

    documents = load_document(file_path)

    documents = add_document_metadata(
        documents=documents,
        document_id=generate_document_id(),
        filename=file_path.name,
    )

    print(f"Loaded: {len(documents)} document/page(s)")

    if documents:
        print("Metadata:")
        print(documents[0].metadata)

        print("Text preview:")
        print(documents[0].page_content[:200])

    print("Status: SUCCESS")


def main():

    supported_files = [
        file
        for file in TEST_DIRECTORY.iterdir()
        if file.suffix.lower() in {".pdf", ".docx", ".txt"}
    ]

    if not supported_files:
        print("No PDF, DOCX, or TXT files found in data/doc")
        return

    for file_path in supported_files:
        try:
            test_document(file_path)
        except Exception as e:
            print(f"Status: FAILED")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()