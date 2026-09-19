import os
from typing import Optional
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from src.rag.models import get_llm
from src.rag.tools import make_rag_tool
from src.rag.vectorstore import VectorStoreManager
from src.rag.dynamic_tools import create_tool_suite, ToolRegistry

load_dotenv()

SYSTEM_PROMPT = """You are an expert autonomous research assistant with access to a library of books (including 'The 33 Strategies of War', 'The 48 Laws of Power', and 'The Molecule of More') and a powerful suite of tools.

Your Tools:
1. `search_knowledge_base`: Semantic search across all indexed book passages. Use focused queries.
2. `list_available_documents`: Check which books and how many chunks are loaded in the library.
3. `calculate`: Compute math expressions, ratios, or statistics accurately.
4. `run_python_code`: Run Python snippets to analyze text, filter data, or perform custom logic.
5. `create_custom_tool`: Dynamically create and register a new Python tool whenever you need a specialized capability.
6. `create_pdf` (or `generate_pdf_report` / `write_pdf`): Export formatted reports, summaries, or facts into a professional PDF file on disk.

CRITICAL EFFICIENCY RULES (To stay within API rate limits):
- NEVER issue more than 2 `search_knowledge_base` queries in a single turn. Be highly selective and concise with search queries.
- Ground your answers in retrieved passages and cite source books and page numbers.
- Deliver structured, thorough answers based on the retrieved facts without requesting redundant searches.
"""


def build_rag_agent(
    vectorstore_manager: Optional[VectorStoreManager] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """Builds and compiles a LangGraph ReAct agent with RAG, utility tools, and dynamic tool creation."""
    llm = get_llm(provider=provider, model_name=model_name, temperature=temperature)

    # Initialize Tool Suite (RAG, List Docs, Calculator, Python Exec, Tool Creator)
    registry = ToolRegistry(vectorstore_manager)
    tools = create_tool_suite(registry)

    # Memory checkpointer for multi-turn conversations
    checkpointer = MemorySaver()

    # Build LangGraph agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    return agent
