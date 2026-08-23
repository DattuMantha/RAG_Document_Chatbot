from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

DEFAULT_MODELS = {
    "OpenAI": "gpt-4.1-mini",
    "Google Gemini": "gemini-3.6-flash",
    "Hugging Face": "",
}

def create_openai_llm(
    api_key: str,
    model: str,
    temperature: float = 0.0,
):
    """Create an OpenAI chat model."""

    return ChatOpenAI(
        api_key=api_key,
        model=model,
        temperature=temperature,
    )


def create_gemini_llm(
    api_key: str,
    model: str,
    temperature: float = 0.0,
):
    """Create a Google Gemini chat model."""

    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model=model,
        temperature=temperature,
    )


def create_huggingface_llm(
    api_key: str,
    model: str,
    temperature: float = 0.0,
):
    """Create a Hugging Face chat model."""

    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        huggingfacehub_api_token=api_key,
        temperature=temperature,
        max_new_tokens=1024,
    )

    return ChatHuggingFace(llm=endpoint)


def create_llm(
    provider: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
):
    """
    Create an LLM based on the selected provider.
    """

    if not api_key or not api_key.strip():
        raise ValueError(
            f"API key is required for {provider}."
        )
    provider = provider.strip()

    if provider == "OpenAI":
        return create_openai_llm(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

    if provider == "Google Gemini":
        return create_gemini_llm(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

    if provider == "Hugging Face":
        return create_huggingface_llm(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}"
    )