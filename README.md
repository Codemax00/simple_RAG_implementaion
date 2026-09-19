# 🤖 Autonomous Agentic RAG System

An advanced, production-grade **Agentic Retrieval-Augmented Generation (RAG)** system built with **LangGraph**, **ChromaDB**, and **Sentence Transformers**, featuring seamless multi-model switching between **Ollama (Local)**, **Groq (Cloud)**, and **Google Gemini (Cloud)**.

---

## ✨ Features

- 🧠 **LangGraph ReAct Agent**: An autonomous reasoning agent that determines when to search documents, perform math, run Python code, or generate formatted reports.
- 🔀 **Multi-Model Provider Support**:
  - **Ollama (Local)**: 100% private, zero-cost, offline local inference with no rate limits (auto-discovers local models like `qwen3:4b`, `llama3.2`, `deepseek-r1`, etc.).
  - **Groq (Cloud)**: Ultra-fast sub-second inference (`qwen/qwen3.8-27b`, `openai/gpt-oss-20b`, `groq/compound`).
  - **Google Gemini (Cloud)**: Deep reasoning and massive context (`gemini-3.6-flash`, `gemini-2.5-pro`).
- 🛠️ **Standalone Model Switcher (`model_switcher.py`)**:
  - Interactive CLI to test latency, check provider health, list local Ollama models, and switch active models.
- ⚡ **Smart Vector Database & Deduplication**:
  - Persistent ChromaDB storage on disk.
  - Automatically verifies if an uploaded document is already indexed—skips redundant chunking and re-embedding to save time and compute.
- 📂 **Interactive Document Upload**:
  - Upload `.pdf`, `.txt`, `.md`, and `.csv` files via file path or native Windows File Explorer picker.
- 🧰 **Extensible Dynamic Tool Suite**:
  - `search_knowledge_base`: Semantic vector retrieval with source and page citations.
  - `list_available_documents`: Real-time inventory of all files and chunk counts in ChromaDB.
  - `calculate`: Safe mathematical evaluations.
  - `run_python_code`: Safe sandboxed Python execution for data analysis.
  - `generate_pdf_report`: Automatically designs and exports professional PDF reports.
  - `create_custom_tool`: Dynamically codes and registers new tools on the fly.
- 💾 **Multi-Turn Conversational Memory**: Preserves context across conversation turns with thread checkpointing.

---

## 🏗️ Architecture

```
                       ┌────────────────────────────┐
                       │       User Terminal        │
                       │    (uv run python app.py)  │
                       └─────────────┬──────────────┘
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
┌──────────────────────┐                           ┌──────────────────────┐
│ 1. Model Selector    │                           │ 2. Smart Doc Uploader│
│ (Ollama/Groq/Gemini) │                           │ (ChromaDB Duplication│
└──────────┬───────────┘                           │        Check)        │
           │                                       └──────────┬───────────┘
           │ initializes                                      │ adds new
           ▼                                                  ▼
┌──────────────────────┐       queries context     ┌──────────────────────┐
│ LangGraph ReAct Agent│ ◄─────────────────────────┤  ChromaDB Vector DB  │
│      (LLM Brain)     │                           │ (Indexed Book Chunks)│
└──────────┬───────────┘                           └──────────────────────┘
           │
           │ uses tools when needed
           ▼
┌────────────────────────────────────────────────────────┐
│ Tools: search_knowledge_base, calculate, run_python,   │
│        generate_pdf_report, list_available_documents   │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) (optional, for local offline models)

### 2. Installation

Clone the repository and install dependencies using `uv` (recommended) or `pip`:

```bash
# Using uv (fastest)
uv sync

# Or using standard pip
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the example environment file and configure your API keys (only required if using cloud models):

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Groq (optional)
GROQ_API_KEY="your_groq_api_key"

# Google Gemini (optional)
GOOGLE_API_KEY="your_gemini_api_key"

# Ollama local endpoint (default)
OLLAMA_BASE_URL="http://localhost:11434"
```

---

## 💻 Usage

### Run the Agentic Assistant

```bash
uv run python app.py
# or: python app.py
```

1. **Select / Confirm Model**: Choose Ollama, Groq, or Gemini (or press Enter for default).
2. **Upload Documents**: View active documents in the library or add new files via file path or Explorer.
3. **Chat**: Ask questions grounded in your uploaded documents!

#### In-Session Commands:
- `/upload` (or `/add`): Add new documents to the vector database on the fly.
- `/model <provider> <model_name>`: Switch models without restarting (e.g. `/model ollama qwen3:4b` or `/model gemini gemini-3.6-flash`).
- `/clear`: Reset conversational memory and token usage.
- `exit`: Quit the application.

---

### Standalone Model Switcher CLI

Run the dedicated model management tool:

```bash
uv run python model_switcher.py
```

Or via command-line flags:
```bash
# Check provider health and installed local models
python model_switcher.py --status

# Switch model directly
python model_switcher.py --set ollama qwen3:4b
python model_switcher.py --set groq qwen/qwen3.8-27b
python model_switcher.py --set gemini gemini-3.6-flash

# Benchmark connection and latency
python model_switcher.py --test
```

---

## 📁 Project Structure

```
├── app.py                     # Main interactive Agent application entrypoint
├── model_switcher.py          # Standalone Model Switcher CLI
├── pyproject.toml             # Project configuration and dependencies
├── requirements.txt           # Pip dependencies specification
├── .env.example               # Template for environment variables
├── .gitignore                 # Excludes credentials, database, and cache
└── src/
    └── rag/
        ├── agent.py           # LangGraph ReAct agent orchestration
        ├── models.py          # Unified LLM provider factory & config manager
        ├── vectorstore.py     # ChromaDB manager, sentence embeddings & deduplication
        ├── doc.py             # Multi-format document loading & file ingestion
        ├── tools.py           # Core RAG similarity search tool
        └── dynamic_tools.py   # Full agent tool suite (Calculator, Python sandbox, PDF writer)
```

---

## 📜 License
MIT License. Feel free to use and adapt for your own research and applications.
