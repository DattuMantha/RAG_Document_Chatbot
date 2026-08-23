import tempfile
from pathlib import Path

import streamlit as st

from core.pipeline import process_document
from core.llm import create_llm
from core.rag import generate_answer


SUPPORTED_TYPES = [
    "pdf",
    "docx",
    "txt",
]


def initialize_chat_state():
    """Initialize Streamlit session state."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "document_id" not in st.session_state:
        st.session_state.document_id = None

    if "document_name" not in st.session_state:
        st.session_state.document_name = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if "document_ready" not in st.session_state:
        st.session_state.document_ready = False


def render_chat(settings):
    """
    Render the main document upload and chat interface.
    """

    initialize_chat_state()

    # ==================================================
    # PAGE HEADER
    # ==================================================

    st.title("📚 RAG Document Chatbot")

    st.write(
        "Upload a document and ask questions about its content."
    )

    # ==================================================
    # DOCUMENT UPLOAD
    # ==================================================

    st.subheader("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF, DOCX, or TXT file",
        type=SUPPORTED_TYPES,
        help=(
            "The document will be processed and stored "
            "in its own vector-store namespace."
        ),
    )

    if uploaded_file is not None:

        st.info(
            f"Selected document: "
            f"**{uploaded_file.name}**"
        )

        # ----------------------------------------------
        # Process button
        # ----------------------------------------------

        if st.button(
            "⚙️ Process Document",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Processing document..."
            ):

                temp_path = None

                try:

                    # Preserve the original filename.
                    # This will be used for source citations.
                    original_filename = uploaded_file.name

                    # Create temporary file for processing.
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=Path(
                            uploaded_file.name
                        ).suffix,
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getbuffer()
                        )

                        temp_path = Path(
                            temp_file.name
                        )

                    # Process document.
                    # Pass the original filename separately
                    # from the temporary file path.
                    result = process_document(
                        file_path=str(temp_path),
                        filename=original_filename,
                        top_k=settings["top_k"],
                    )

                    # Store important objects
                    st.session_state.document_id = (
                        result["document_id"]
                    )

                    st.session_state.document_name = (
                        original_filename
                    )

                    st.session_state.retriever = (
                        result["retriever"]
                    )

                    st.session_state.document_ready = (
                        True
                    )

                    # Reset chat for new document
                    st.session_state.messages = []

                    st.success(
                        "✅ Document processed successfully!"
                    )

                except Exception as e:

                    st.session_state.document_ready = (
                        False
                    )

                    st.error(
                        f"❌ Document processing failed: {e}"
                    )

                finally:

                    # Always remove the temporary file.
                    if temp_path is not None:
                        temp_path.unlink(
                            missing_ok=True
                        )

    # ==================================================
    # DOCUMENT STATUS
    # ==================================================

    if st.session_state.document_ready:

        st.success(
            f"📄 Ready: "
            f"{st.session_state.document_name}"
        )

        st.divider()

        # ==================================================
        # CHAT HISTORY
        # ==================================================

        st.subheader("💬 Chat")

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

                # Show sources for assistant messages
                if (
                    message["role"] == "assistant"
                    and message.get("sources")
                ):

                    render_sources(
                        message["sources"]
                    )

        # ==================================================
        # CHAT INPUT
        # ==================================================

        question = st.chat_input(
            "Ask a question about your document..."
        )

        if question:

            # ----------------------------------------------
            # Display user message
            # ----------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):
                st.markdown(question)

            # ----------------------------------------------
            # Validate API key
            # ----------------------------------------------

            if not settings["api_key"]:

                st.error(
                    "Please enter your API key "
                    "in the sidebar."
                )

                return

            # ----------------------------------------------
            # Generate answer
            # ----------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching document and generating answer..."
                ):

                    try:

                        llm = create_llm(
                            provider=settings["provider"],
                            model=settings["model"],
                            api_key=settings["api_key"],
                            temperature=settings[
                                "temperature"
                            ],
                        )

                        result = generate_answer(
                            llm=llm,
                            retriever=(
                                st.session_state.retriever
                            ),
                            question=question,
                            chat_history=st.session_state.messages,
                        )

                        answer = result["answer"]

                        sources = result[
                            "source_documents"
                        ]

                        st.markdown(answer)

                        render_sources(
                            sources
                        )

                        # Save assistant response
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )

                    except Exception as e:

                        error_message = (
                            f"Unable to generate answer: {e}"
                        )

                        st.error(
                            error_message
                        )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": error_message,
                            }
                        )

    else:

        st.divider()

        st.info(
            "👆 Upload and process a document "
            "to start asking questions."
        )


def render_sources(documents):
    """
    Display source documents used to generate
    the answer.
    """

    if not documents:
        return

    with st.expander(
        "📚 Sources",
        expanded=False,
    ):

        seen = set()

        for document in documents:

            metadata = document.metadata

            filename = metadata.get(
                "filename",
                "Unknown document",
            )

            page = metadata.get(
                "page",
                metadata.get(
                    "page_index",
                    "Unknown",
                ),
            )

            source_key = (
                filename,
                page,
            )

            if source_key in seen:
                continue

            seen.add(source_key)

            st.markdown(
                f"**📄 {filename}**  \n"
                f"Page: **{page}**"
            )