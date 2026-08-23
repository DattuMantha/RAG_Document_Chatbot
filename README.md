# 📚 RAG Document Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF, DOCX, and TXT documents and ask questions about their content.

The application retrieves relevant document chunks and provides them as context to a selected Large Language Model (LLM), helping generate answers grounded in the uploaded document.

## 🚀 Features

- Upload PDF, DOCX, and TXT documents
- Automatic document loading and text extraction
- Recursive text chunking
- Semantic embeddings using Sentence Transformers
- Chroma vector database for document storage
- MMR-based document retrieval
- Question answering using RAG
- Source citations showing the filename and page
- Multiple LLM provider support:
  - OpenAI
  - Google Gemini
  - Hugging Face
- Selectable LLM model
- Configurable temperature
- Configurable number of retrieved documents (`top_k`)
- Streamlit web interface
- Document-specific vector-store namespaces

## 🏗️ Architecture

```text
                ┌─────────────────────┐
                │   Streamlit UI      │
                │                     │
                │  Sidebar            │
                │  - LLM Provider     │
                │  - API Key          │
                │  - Model             │
                │  - Temperature      │
                │  - Retrieval Top-K  │
                │                     │
                │  Main Area          │
                │  - Upload Document  │
                │  - Chat             │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Document Loader     │
                │ PDF / DOCX / TXT    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Text Splitter       │
                │ Recursive Chunking  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Embedding Model     │
                │ SentenceTransformers│
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Chroma Vector Store │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ MMR Retriever       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ RAG Pipeline        │
                │                     │
                │ Retrieved Context   │
                │        +            │
                │ User Question       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Selected LLM        │
                │ OpenAI / Gemini /   │
                │ Hugging Face        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Grounded Answer     │
                │ + Source            │
                └─────────────────────┘
📁 Project Structure
RAG_Document_Chatbot/
│
├── app.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── core/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── loaders.py
│   ├── pipeline.py
│   ├── rag.py
│   ├── retriever.py
│   ├── splitter.py
│   └── vectorstore.py
│
├── data/
│   └── doc/
│       └── test_document.txt
│
├── tests/
│   ├── test_document_formats.py
│   ├── test_pipeline.py
│   └── test_rag_pipeline.py
│
└── ui/
    ├── __init__.py
    ├── chat.py
    └── sidebar.py
🔄 RAG Pipeline

The application follows these steps:

User uploads a document.
The document is loaded based on its file type.
Metadata such as filename, document ID, file type, and page index is added.
The document is divided into smaller chunks.
Each chunk is converted into an embedding.
Embeddings are stored in a document-specific Chroma collection.
The user's question is converted into a retrieval query.
Relevant chunks are retrieved using MMR.
Retrieved chunks are passed to the selected LLM as context.
The LLM generates an answer using the provided document context.
Source information is displayed with the answer.
🤖 Supported LLM Providers
OpenAI

Supports OpenAI chat models configured in the application.

Google Gemini

Supports Google Gemini models configured in the application.

Hugging Face

Supports Hugging Face models available through the configured inference provider.

API availability and model availability may vary depending on the provider and account.

⚙️ Configuration

The sidebar allows the user to configure:

LLM Provider

Choose between:

OpenAI
Google Gemini
Hugging Face
API Key

Enter the API key for the selected provider.

API keys are entered at runtime and should never be committed to GitHub.

Model

Select the model to use for generating responses.

Temperature

Controls the randomness of the generated response.

Lower temperature → more deterministic responses
Higher temperature → more varied/creative responses
Retrieval

The top_k value controls how many relevant document chunks are retrieved before generating the answer.

For example:

top_k = 5

means the retriever attempts to provide the five most relevant chunks to the RAG pipeline.

🛠️ Installation
1. Clone the repository
git clone https://github.com/DattuMantha/RAG_Document_Chatbot.git
cd RAG_Document_Chatbot
2. Create a virtual environment

Windows:

python -m venv .venv
3. Activate the environment
.venv\Scripts\activate
4. Install dependencies
pip install -r requirements.txt
🔑 API Keys

Create a .env file if required by your local configuration.

Use .env.example as a reference.

Never commit API keys to GitHub.

The project .gitignore excludes the .env file.

▶️ Run the Application

Start the Streamlit application with:

streamlit run app.py

The application will open in your browser.

🧪 Testing

Document format loading can be tested using:

python tests/test_document_formats.py

The complete document processing and retrieval pipeline can be tested using:

python tests/test_pipeline.py

The complete RAG pipeline can be tested using:

python tests/test_rag_pipeline.py
📄 Supported Documents
Format	Supported
PDF	✅
DOCX	✅
TXT	✅
🔒 Grounded Responses

The RAG prompt instructs the LLM to answer using only the retrieved document context.

If the requested information cannot be found in the retrieved context, the assistant is instructed to indicate that the information was not found in the uploaded document.

🧰 Technologies Used
Python
Streamlit
LangChain
Chroma
Sentence Transformers
Hugging Face
OpenAI
Google Gemini
PyPDF
python-docx
🎯 Project Objective

The objective of this project is to build a document-based question-answering system using Retrieval-Augmented Generation.

Instead of relying only on the knowledge contained within an LLM, the application retrieves relevant information from user-uploaded documents and provides that information to the LLM as context before generating an answer.

👨‍💻 Author

Dattu Mantha