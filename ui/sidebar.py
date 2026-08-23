import streamlit as st

from core.llm import DEFAULT_MODELS


# Available LLM providers
PROVIDERS = [
    "OpenAI",
    "Google Gemini",
    "Hugging Face",
]


# Provider-specific models
MODELS = {
    "OpenAI": [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4.1-nano",
    ],
    "Google Gemini": [
        "gemini-3.6-flash",
    ],
    "Hugging Face": [],
}


def render_sidebar():
    """Render the RAG configuration sidebar."""

    with st.sidebar:

        st.title("⚙️ RAG Settings")

        st.divider()

        # -----------------------------------------
        # LLM Provider
        # -----------------------------------------

        provider = st.selectbox(
            "LLM Provider",
            options=PROVIDERS,
        )

        # -----------------------------------------
        # API Key
        # -----------------------------------------

        api_key = st.text_input(
            f"{provider} API Key",
            type="password",
            placeholder="Enter your API key",
            help=(
                "Your API key is used only for the "
                "current application session."
            ),
        )

        # -----------------------------------------
        # Model
        # -----------------------------------------

        available_models = MODELS.get(provider, [])

        default_model = DEFAULT_MODELS.get(
            provider,
            "",
        )

        if available_models:

            default_index = (
                available_models.index(default_model)
                if default_model in available_models
                else 0
            )

            model = st.selectbox(
                "Model",
                options=available_models,
                index=default_index,
            )

        else:

            model = st.text_input(
                "Model",
                value=default_model,
                placeholder="Enter Hugging Face model ID",
                help=(
                    "Enter a Hugging Face model that is "
                    "available through your configured "
                    "Inference Provider."
                ),
            )

        # -----------------------------------------
        # Temperature
        # -----------------------------------------

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help=(
                "Controls how creative or deterministic "
                "the response is."
            ),
        )

        st.divider()

        # -----------------------------------------
        # Retrieval Settings
        # -----------------------------------------

        st.subheader("🔎 Retrieval")

        top_k = st.slider(
            "Documents to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help=(
                "Number of relevant document chunks "
                "retrieved for each question."
            ),
        )

        return {
            "provider": provider,
            "api_key": api_key,
            "model": model,
            "temperature": temperature,
            "top_k": top_k,
        }