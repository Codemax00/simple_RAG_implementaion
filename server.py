import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.rag.vectorstore import VectorStoreManager
from src.rag.agent import build_rag_agent
from src.rag.models import (
    load_model_config,
    save_model_config,
    check_ollama_status,
    DEFAULT_PRESETS,
    get_llm,
)
from src.rag.doc import aingest_new_files

load_dotenv()

app = FastAPI(title="Agentic RAG Backend API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Backend State
STATE: Dict[str, Any] = {
    "vm": None,
    "agent": None,
    "session_counter": 1,
}


def get_agent(force_reload: bool = False):
    """Retrieves or instantiates the LangGraph RAG Agent directly from the existing backend."""
    if STATE["vm"] is None:
        STATE["vm"] = VectorStoreManager(persist_directory="./chroma_db")
    if STATE["agent"] is None or force_reload:
        cfg = load_model_config()
        STATE["agent"] = build_rag_agent(
            vectorstore_manager=STATE["vm"],
            provider=cfg.get("provider", "ollama"),
            model_name=cfg.get("model_name", "qwen3:4b"),
            temperature=cfg.get("temperature", 0.2),
        )
    return STATE["agent"]


@app.on_event("startup")
async def on_startup():
    Path("./documents").mkdir(parents=True, exist_ok=True)
    Path("./static").mkdir(parents=True, exist_ok=True)
    get_agent()


# Mount Static Assets
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_file = Path("static/index.html")
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Static UI loading...</h1>")


# -------------------------------------------------------------------
# Model APIs
# -------------------------------------------------------------------
class ModelSelectPayload(BaseModel):
    provider: str
    model_name: str
    temperature: Optional[float] = 0.2


@app.get("/api/models")
async def get_models():
    """Returns active model, available presets, and provider health."""
    cfg = load_model_config()
    base_url = cfg.get("ollama_base_url", "http://localhost:11434")
    ollama_ok, _ = check_ollama_status(base_url)
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_gemini = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

    return {
        "active": cfg,
        "providers": [
            {
                "id": "ollama",
                "name": "Ollama (Local)",
                "description": "Unlimited, private offline model inference.",
                "status": "ready" if ollama_ok else "offline",
                "status_message": "Online" if ollama_ok else "Offline (start `ollama serve`)",
                "models": DEFAULT_PRESETS.get("ollama", []),
            },
            {
                "id": "groq",
                "name": "Groq Cloud (LPU)",
                "description": "Ultra-fast inference on Groq hardware.",
                "status": "ready" if has_groq else "needs_key",
                "status_message": "Key Configured" if has_groq else "GROQ_API_KEY missing in .env",
                "models": DEFAULT_PRESETS.get("groq", []),
            },
            {
                "id": "gemini",
                "name": "Google Gemini",
                "description": "High-context intelligence from Google.",
                "status": "ready" if has_gemini else "needs_key",
                "status_message": "Key Configured" if has_gemini else "GOOGLE_API_KEY missing in .env",
                "models": DEFAULT_PRESETS.get("gemini", []),
            },
        ],
    }


@app.post("/api/models/select")
async def select_model(payload: ModelSelectPayload):
    """Switches model configuration and refreshes the backend agent."""
    try:
        # Pre-verify LLM parameters
        get_llm(provider=payload.provider, model_name=payload.model_name, temperature=payload.temperature)
        save_model_config(payload.provider, payload.model_name, payload.temperature)
        STATE["session_counter"] += 1
        get_agent(force_reload=True)
        return {"success": True, "active": load_model_config()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------------------------------------------------------------------
# Document Library & Pipelined Ingestion APIs
# -------------------------------------------------------------------
@app.get("/api/documents")
async def get_documents():
    """Returns library files and indexed chunk counts from ChromaDB."""
    vm = STATE["vm"] or VectorStoreManager(persist_directory="./chroma_db")
    STATE["vm"] = vm

    docs_dir = Path("./documents")
    indexed = vm.get_indexed_documents()
    files = []

    if docs_dir.exists():
        for f in docs_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                size = f.stat().st_size
                chunks = indexed.get(f.name, 0)
                files.append({
                    "name": f.name,
                    "size_formatted": f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB",
                    "chunk_count": chunks,
                    "is_indexed": chunks > 0,
                    "extension": f.suffix.replace(".", "").upper(),
                })

    return {
        "total_chunks": vm.count(),
        "total_files": len(files),
        "files": sorted(files, key=lambda x: x["name"]),
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accepts uploaded documents and uses our backend async streaming pipeline."""
    vm = STATE["vm"] or VectorStoreManager(persist_directory="./chroma_db")
    STATE["vm"] = vm

    dest_file = Path("./documents") / file.filename
    content = await file.read()
    with open(dest_file, "wb") as f:
        f.write(content)

    async def sse_pipeline():
        if vm.is_document_indexed(file.filename):
            c = vm.get_document_chunk_count(file.filename)
            yield f"data: {json.dumps({'stage': 'skipped', 'message': f'{file.filename} is already indexed ({c} chunks).' })}\n\n"
            yield f"data: {json.dumps({'stage': 'done', 'chunks_indexed': 0})}\n\n"
            return

        yield f"data: {json.dumps({'stage': 'start', 'message': f'Starting pipelined ingestion for {file.filename}...' })}\n\n"
        # Directly call backend aingest_new_files which runs our async producer-consumer pipeline!
        chunks = await aingest_new_files([dest_file], target_dir="./documents", vm=vm)
        yield f"data: {json.dumps({'stage': 'done', 'chunks_indexed': chunks, 'message': f'Successfully indexed {chunks} chunks for {file.filename}!' })}\n\n"

    return StreamingResponse(sse_pipeline(), media_type="text/event-stream")


from src.rag.optimizer import UserInputOptimizer

OPTIMIZER = UserInputOptimizer()

# -------------------------------------------------------------------
# Real-time Agent Streaming APIs
# -------------------------------------------------------------------
class ChatPayload(BaseModel):
    message: str
    thread_id: Optional[str] = None


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatPayload, request: Request):
    """Streams tokens and tool calls in real time using backend LangGraph agent with Dual-Memory Query Optimization."""
    agent = get_agent()
    thread_id = payload.thread_id or f"web-session-{STATE['session_counter']}"
    config = {"configurable": {"thread_id": thread_id}}

    # Fetch active LLM for query optimization
    cfg = load_model_config()
    try:
        opt_llm = get_llm(provider=cfg.get("provider", "ollama"), model_name=cfg.get("model_name", "qwen3:4b"), temperature=0.0)
    except Exception:
        opt_llm = None

    def parse_chunk_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join([p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"])
        return ""

    async def sse_stream():
        called = set()
        accumulated_answer = []

        try:
            # 1. Optimize user input using Dual-Memory (History + Keywords)
            opt_res = await OPTIMIZER.aoptimize_query(payload.message, thread_id, llm=opt_llm)
            yield f"data: {json.dumps({'type': 'optimized_query', 'original': opt_res['original'], 'optimized': opt_res['optimized'], 'keywords': opt_res['keywords'], 'was_rewritten': opt_res['was_rewritten']})}\n\n"

            # 2. Formulate agent input message
            if opt_res.get("was_rewritten") and opt_res["optimized"] != payload.message:
                agent_msg = f"{payload.message} (Context & Retrieval Focus: {opt_res['optimized']})"
            else:
                agent_msg = payload.message

            inputs = {"messages": [("user", agent_msg)]}

            # 3. Stream agent execution
            async for chunk, meta in agent.astream(inputs, config, stream_mode="messages"):
                if await request.is_disconnected():
                    print("[INFO] Client disconnected (Stop requested). Halting generation.")
                    break

                if meta.get("langgraph_node") == "agent":
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        for tc in chunk.tool_calls:
                            cid = tc.get("id") or f"{tc.get('name')}_{tc.get('args')}"
                            if cid not in called:
                                called.add(cid)
                                yield f"data: {json.dumps({'type': 'tool', 'name': tc.get('name'), 'args': tc.get('args')})}\n\n"

                    text = parse_chunk_text(chunk.content)
                    if text:
                        accumulated_answer.append(text)
                        yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"

            # 4. Record turn into Dual Memory
            full_text = "".join(accumulated_answer)
            OPTIMIZER.record_turn(thread_id, "user", payload.message)
            if full_text:
                OPTIMIZER.record_turn(thread_id, "assistant", full_text)

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as err:
            yield f"data: {json.dumps({'type': 'error', 'error': str(err)})}\n\n"

    return StreamingResponse(sse_stream(), media_type="text/event-stream")


@app.post("/api/chat/reset")
async def reset_chat():
    old_id = f"web-session-{STATE['session_counter']}"
    OPTIMIZER.clear_session(old_id)
    STATE["session_counter"] += 1
    new_id = f"web-session-{STATE['session_counter']}"
    OPTIMIZER.clear_session(new_id)
    return {"success": True, "thread_id": new_id}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
