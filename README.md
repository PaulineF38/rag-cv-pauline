# 💬 RAG CV — Interactive Résumé Chatbot

A Streamlit application that lets you chat with my CV using Retrieval-Augmented Generation. Ask anything — in French or in English — and the app retrieves the relevant parts of my résumé before generating a grounded, factual answer.

This project was built as a personal deep-dive into the RAG stack: from chunking and embedding a document to wiring up semantic retrieval with a real LLM.

---

## Quick Start

To launch the project, you only need **Docker** and an API key from one of the supported providers.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/rag-cv-pauline.git
   cd rag-cv-pauline
   ```

2. **Start the application:**
   ```bash
   docker compose up web_app
   ```

3. **Open your browser:**

   Go to: [http://localhost:8501](http://localhost:8501)

   **🔑 Requirement:** You'll need an API key to use the chatbot. Pick one of these:
   - [OpenAI](https://platform.openai.com/api-keys) — uses `gpt-4o-mini`
   - [Mistral AI](https://console.mistral.ai/) — uses `mistral-large-latest`
   - [Google AI Studio](https://aistudio.google.com/app/apikey) — uses `gemini-1.5-flash`

   Select your provider in the sidebar, paste your key, and start chatting.

> **Note on first launch:** Docker will download the embedding model (`BAAI/bge-m3`, ~570 MB) on first run, which takes a couple of minutes. The first query will also have a short delay while the model loads into memory — completely normal.

---

### 📝 Note : About this project

I built this project to go deeper into how RAG actually works under the hood. I wanted to understand each step: how a document gets chunked, how chunks become vectors, how semantic search retrieves the right context, and how the LLM uses that context to answer without making things up.

My own CV felt like the perfect document to experiment with — small enough to stay manageable, but personal enough to make the results meaningful. The source PDF is not included in the repo, but the vector database is pre-built so the app runs as-is. The ingestion script (`scripts/embedder.py`) is there if you want to see exactly how the embedding pipeline works.

---

## Project objectives

**RAG pipeline from scratch:** Understanding each layer — document loading, chunking strategy (1200 tokens, 250 overlap), vector embedding with `BAAI/bge-m3`, semantic retrieval (top-5 chunks), and LLM answer generation.

**Multi-provider LLM integration:** Abstracting three different providers (OpenAI, Mistral, Gemini) behind a single interface so the user can swap models without touching the code.

**Streaming UX:** Responses are streamed word by word for a more natural chat experience.

**Multilingual support:** The prompt is designed so the LLM answers in the same language as the question — French or English.

**Containerisation:** The app and the ingestion pipeline are two separate Docker Compose services, keeping concerns clean.

---

## Tech stack

**UI:** Streamlit

**RAG framework:** LangChain

**Embedding model:** `BAAI/bge-m3` (HuggingFace — multilingual, strong on French)

**Vector store:** Chroma (local, persisted on disk)

**LLM providers:** OpenAI · Mistral AI · Google Gemini

**Orchestration:** Docker & Docker Compose

---

## 🔒 Security note

API keys are never stored anywhere in the app. They are kept in the user's session only and discarded as soon as the tab is closed.
