from pathlib import Path
import os

from dotenv import load_dotenv


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")


# Data directories
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "doc"
VECTORSTORE_DIR = DATA_DIR / "vector_stores"


# Supported LLM providers
LLM_PROVIDERS = [
    "OpenAI",
    "Google Gemini",
    "Hugging Face",
]


# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")