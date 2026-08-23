import streamlit as st

from ui.sidebar import render_sidebar
from ui.chat import render_chat


st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚",
    layout="wide",
)


def main():
    settings = render_sidebar()
    render_chat(settings)


if __name__ == "__main__":
    main()