import os
import sys
import asyncio
from dotenv import load_dotenv

# Ensure Windows terminal prints Unicode / emojis / special symbols properly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.rag.doc import load_all_documents
from src.rag.vectorstore import VectorStoreManager
from src.rag.tools import make_rag_tool

load_dotenv()


async def asetup_knowledge_base() -> VectorStoreManager:
    """Initializes and verifies the persistent ChromaDB vector store."""
    vm = VectorStoreManager(persist_directory="./chroma_db")

    if vm.is_populated():
        print(f"[INFO] Vector store already loaded from './chroma_db' ({vm.count()} indexed chunks).")
    else:
        print("[INFO] No existing vector store found. Ingesting documents...")
        docs = load_all_documents("./documents")
        if not docs:
            print("[WARN] No documents found in './documents' folder!")
            return vm
        print(f"[INFO] Ingested {len(docs)} document pages. Starting pipelined async chunking & embedding...")
        total_chunks = await vm.aindex_document_stream(docs)
        print(f"[SUCCESS] Successfully indexed {total_chunks} chunks into './chroma_db'.")

    return vm


def run_direct_search_mode(vm: VectorStoreManager):
    """Direct search mode if no LLM API key is configured."""
    print("\n" + "=" * 60)
    print("DIRECT RAG SEARCH MODE (No API Key detected)")
    print("Add GROQ_API_KEY or GOOGLE_API_KEY to your .env file,")
    print("or start Ollama locally to enable the LangGraph Agent.")
    print("=" * 60 + "\n")

    tool = make_rag_tool(vm)

    while True:
        try:
            query = input("\nEnter search query (or 'exit' to quit): ").strip()
            if not query or query.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break

            print("\n[Tool: search_knowledge_base.invoke]")
            result = tool.invoke({"query": query})
            print(result)
        except KeyboardInterrupt:
            print("\nExiting.")
            break


def extract_text_from_chunk(content) -> str:
    """Extracts plain text from different provider chunk formats (Ollama string vs Gemini dict list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


async def run_langgraph_agent(vm: VectorStoreManager):
    """Runs the full interactive LangGraph Agent asynchronously with real-time token streaming."""
    from src.rag.agent import build_rag_agent
    from src.rag.models import load_model_config, save_model_config, prompt_select_model

    def create_agent_instance():
        while True:
            cfg = load_model_config()
            prov = cfg.get("provider", "ollama")
            mod = cfg.get("model_name", "unknown")
            try:
                agent = build_rag_agent(vm, provider=prov, model_name=mod)
                print(f"[INFO] Active Model: {mod} ({prov.upper()})")
                return agent, cfg
            except ValueError as ve:
                print(f"\n[MODEL INITIALIZATION ERROR] {ve}")
                print("\nPlease choose a different model or provider:")
                cfg = prompt_select_model()

    agent, model_cfg = create_agent_instance()
    session_counter = 1
    config = {"configurable": {"thread_id": f"interactive-session-{session_counter}"}, "recursion_limit": 5}

    # Document upload option comes AFTER model is selected and loaded
    from src.rag.doc import aprompt_upload_documents
    await aprompt_upload_documents(vm)

    print("============================================================")
    print("      ASYNC LANGGRAPH AGENTIC RAG SYSTEM READY")
    print(f"Provider: {model_cfg.get('provider', '').upper()} | Model: {model_cfg.get('model_name', '')}")
    print("Commands:")
    print("  • Type your question to query the knowledge base (Real-time Token Streaming)")
    print("  • Type '/upload' to add more documents into the library")
    print("  • Type '/model' to view or switch model (Ollama / Groq / Gemini)")
    print("  • Type '/clear' to reset chat memory")
    print("  • Type 'exit' to quit")
    print("=" * 60 + "\n")

    while True:
        try:
            # Asynchronously wait for user input without blocking the event loop
            user_input = await asyncio.to_thread(input, "\nYou: ")
            user_input = user_input.strip()

            if not user_input or user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            if user_input.lower() in ("/clear", "/reset", "clear"):
                session_counter += 1
                config = {"configurable": {"thread_id": f"interactive-session-{session_counter}"}, "recursion_limit": 5}
                print("[INFO] Conversation history reset! Starting with fresh token quota.")
                continue

            if user_input.lower() in ("/upload", "/add", "upload"):
                await aprompt_upload_documents(vm)
                continue

            if user_input.lower().startswith("/model"):
                parts = user_input.split()
                if len(parts) >= 3:
                    new_prov = parts[1].lower()
                    new_model = parts[2]
                    save_model_config(new_prov, new_model)
                    agent, model_cfg = create_agent_instance()
                    session_counter += 1
                    config = {"configurable": {"thread_id": f"interactive-session-{session_counter}"}, "recursion_limit": 5}
                    print(f"[SUCCESS] Switched model to {new_model} ({new_prov.upper()})")
                else:
                    curr = load_model_config()
                    print(f"\nCurrent Model: {curr.get('model_name')} ({curr.get('provider', '').upper()})")
                    print("Usage: /model <provider> <model_name>")
                    print("Examples:")
                    print("  /model ollama llama3.2")
                    print("  /model groq qwen/qwen3.8-27b")
                    print("  /model gemini gemini-3.6-flash")
                    print("Or run `python model_switcher.py` for full interactive model management.")
                continue

            # Stream the agent asynchronously with real-time token rendering
            inputs = {"messages": [("user", user_input)]}
            printed_ai_header = False
            called_tools = set()

            async for chunk, meta in agent.astream(inputs, config, stream_mode="messages"):
                node = meta.get("langgraph_node", "")

                if node == "agent":
                    # Tool call detection
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        for call in chunk.tool_calls:
                            cname = call.get("name")
                            cargs = call.get("args")
                            call_id = call.get("id") or f"{cname}_{cargs}"
                            if cname and call_id not in called_tools:
                                called_tools.add(call_id)
                                query_str = cargs.get("query", "") if isinstance(cargs, dict) else str(cargs)
                                print(f"\n[Agent Tool Call] -> {cname}({query_str})")

                    # Real-time token streaming to console
                    text = extract_text_from_chunk(chunk.content)
                    if text:
                        if not printed_ai_header:
                            print("\nAssistant:\n", end="", flush=True)
                            printed_ai_header = True
                        print(text, end="", flush=True)

            print("\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            error_str = str(e)
            if "rate_limit_exceeded" in error_str or "429" in error_str:
                print("\n[Rate Limit Notice] Provider rate limit reached.")
                print("Tip: Run `python model_switcher.py` to switch to Ollama (Local) for unlimited offline queries!")
            elif "Cannot connect to Ollama" in error_str or "Connection refused" in error_str:
                print("\n[Ollama Connection Error] Cannot reach Ollama at http://localhost:11434.")
                print("Please ensure Ollama is running (`ollama serve` or open the Ollama app).")
            else:
                print(f"\n[Error] {e}")


async def main():
    from src.rag.models import prompt_select_model, check_ollama_status

    # Initialize ChromaDB vector store
    vm = await asetup_knowledge_base()

    # Interactively ask the user to select or confirm model on startup
    config = await asyncio.to_thread(prompt_select_model)
    provider = config.get("provider", "ollama")

    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_gemini = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    ollama_up, _ = check_ollama_status(config.get("ollama_base_url", "http://localhost:11434"))

    can_run_agent = (
        (provider == "ollama")
        or (provider == "groq" and has_groq)
        or (provider in ("gemini", "google") and has_gemini)
        or has_groq
        or has_gemini
        or ollama_up
    )

    if can_run_agent:
        await run_langgraph_agent(vm)
    else:
        run_direct_search_mode(vm)


if __name__ == "__main__":
    asyncio.run(main())
