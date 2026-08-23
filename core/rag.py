from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


RAG_SYSTEM_PROMPT = """
You are a document question-answering assistant.

Answer the user's question using ONLY the information
provided in the document context.

You may use the conversation history to understand
what the user is referring to, but the actual answer
must come ONLY from the document context.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. Use conversation history only to understand references
   such as "it", "this", "that", or follow-up questions.
4. If the answer cannot be found in the provided context,
   clearly say that the information was not found in
   the uploaded document.
5. Give a clear and concise answer.
6. When possible, mention the relevant source information.

Conversation history:
{chat_history}

Document context:
{context}
"""


def create_rag_prompt():
    """
    Create the prompt used by the conversational RAG pipeline.
    """

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                RAG_SYSTEM_PROMPT,
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )


def format_documents(documents: list[Document]) -> str:
    """
    Convert retrieved documents into a formatted
    context string.
    """

    formatted_context = []

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

        formatted_context.append(
            f"Source: {filename} | Page: {page}\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(
        formatted_context
    )


def format_chat_history(chat_history) -> str:
    """
    Convert previous chat messages into a simple
    text format for the LLM.
    """

    if not chat_history:
        return "No previous conversation."

    formatted_history = []

    for message in chat_history:

        role = message.get("role", "")
        content = message.get("content", "")

        # Ignore error messages or empty messages
        if not content:
            continue

        if role == "user":
            formatted_history.append(
                f"User: {content}"
            )

        elif role == "assistant":
            formatted_history.append(
                f"Assistant: {content}"
            )

    if not formatted_history:
        return "No previous conversation."

    return "\n".join(formatted_history)


def generate_answer(
    llm,
    retriever,
    question: str,
    chat_history=None,
):
    """
    Retrieve relevant documents and generate
    a grounded answer using the selected LLM
    and previous conversation history.
    """

    # --------------------------------------------------
    # 1. Retrieve relevant document chunks
    # --------------------------------------------------

    documents = retriever.invoke(question)

    # --------------------------------------------------
    # 2. Format retrieved context
    # --------------------------------------------------

    context = format_documents(documents)

    # --------------------------------------------------
    # 3. Format conversation history
    # --------------------------------------------------

    history = format_chat_history(
        chat_history
    )

    # --------------------------------------------------
    # 4. Create RAG prompt
    # --------------------------------------------------

    prompt = create_rag_prompt()

    # --------------------------------------------------
    # 5. Build messages
    # --------------------------------------------------

    messages = prompt.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": history,
        }
    )

    # --------------------------------------------------
    # 6. Generate answer
    # --------------------------------------------------

    response = llm.invoke(messages)

    content = response.content

    # --------------------------------------------------
    # 7. Handle structured responses
    #    (Gemini and other models)
    # --------------------------------------------------

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):
                text_parts.append(
                    item.get("text", "")
                )

            elif isinstance(item, str):
                text_parts.append(item)

        content = "\n".join(text_parts)

    # --------------------------------------------------
    # 8. Return answer and sources
    # --------------------------------------------------

    return {
        "answer": content,
        "source_documents": documents,
    }