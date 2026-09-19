import os
import json
import time
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv

load_dotenv()

CONFIG_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "model_config.json"))

DEFAULT_PRESETS = {
    "ollama": [
        "llama3.2",
        "llama3.1",
        "mistral",
        "qwen2.5",
        "deepseek-r1",
        "phi3",
        "gemma2",
    ],
    "groq": [
        "llama-3.3-70b-versatile",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ],
    "gemini": [
        "gemini-3.6-flash",
        "gemini-2.5-pro",
    ],
}


def load_model_config() -> Dict[str, Any]:
    """Loads saved model configuration, or derives a default based on available keys."""
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                if isinstance(config, dict) and "provider" in config:
                    return config
        except Exception as e:
            print(f"[WARN] Failed to read {CONFIG_FILE_PATH}: {e}")

    # Fallback heuristic based on environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if groq_api_key:
        provider = "groq"
        model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    elif google_api_key:
        provider = "gemini"
        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    else:
        provider = "ollama"
        model = "llama3.2"

    default_config = {
        "provider": provider,
        "model_name": model,
        "temperature": 0.2,
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    }
    return default_config


def save_model_config(
    provider: str,
    model_name: str,
    temperature: float = 0.2,
    ollama_base_url: str = "http://localhost:11434",
    **kwargs,
) -> Dict[str, Any]:
    """Saves the active model configuration to model_config.json."""
    config = {
        "provider": provider.lower().strip(),
        "model_name": model_name.strip(),
        "temperature": float(temperature),
        "ollama_base_url": ollama_base_url.strip(),
        **kwargs,
    }
    try:
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"[SUCCESS] Model configuration saved to {os.path.basename(CONFIG_FILE_PATH)}")
    except Exception as e:
        print(f"[ERROR] Could not save model config: {e}")
    return config


def check_ollama_status(base_url: str = "http://localhost:11434") -> Tuple[bool, str]:
    """Verifies if the local Ollama server is reachable."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                return True, "Ollama service is active and responding."
            return False, f"Ollama returned HTTP status {resp.status}."
    except Exception as e:
        return False, (
            f"Cannot connect to Ollama at {base_url}. "
            "Please ensure Ollama is installed and running (`ollama serve` or open the Ollama desktop app)."
        )


def get_ollama_local_models(base_url: str = "http://localhost:11434") -> List[str]:
    """Retrieves list of locally installed Ollama models via its API."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            return models
    except Exception:
        return []


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
):
    """Instantiates and returns the configured LangChain chat model."""
    load_dotenv()
    active_config = load_model_config()

    chosen_provider = (provider or active_config.get("provider", "groq")).lower().strip()
    chosen_model = (model_name or active_config.get("model_name", "")).strip()
    chosen_temp = temperature if temperature is not None else active_config.get("temperature", 0.2)
    ollama_url = active_config.get("ollama_base_url", "http://localhost:11434")

    if chosen_provider == "ollama":
        from langchain_ollama import ChatOllama

        if not chosen_model:
            chosen_model = "llama3.2"

        # Check server reachability
        is_up, msg = check_ollama_status(ollama_url)
        if not is_up:
            print(f"[WARNING] {msg}")

        print(f"[INFO] Initializing Ollama Local LLM: model='{chosen_model}' on '{ollama_url}'")
        return ChatOllama(
            model=chosen_model,
            base_url=ollama_url,
            temperature=chosen_temp,
        )

    elif chosen_provider == "groq":
        from langchain_groq import ChatGroq

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "Missing GROQ_API_KEY! Please set GROQ_API_KEY in your .env file,\n"
                "or switch to a local Ollama model using `python model_switcher.py`."
            )

        if not chosen_model:
            chosen_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

        print(f"[INFO] Initializing Groq LLM: model='{chosen_model}'")
        max_tokens = 800 if "qwen" in chosen_model.lower() else 2000
        return ChatGroq(
            model=chosen_model,
            temperature=chosen_temp,
            max_tokens=max_tokens,
            max_retries=5,
            api_key=groq_api_key,
        )

    elif chosen_provider in ("gemini", "google"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not google_api_key:
            raise ValueError(
                "Missing GOOGLE_API_KEY! Please set GOOGLE_API_KEY in your .env file,\n"
                "or switch to a local Ollama model using `python model_switcher.py`."
            )

        if not chosen_model:
            chosen_model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

        print(f"[INFO] Initializing Gemini LLM: model='{chosen_model}'")
        return ChatGoogleGenerativeAI(
            model=chosen_model,
            temperature=chosen_temp,
            google_api_key=google_api_key,
        )

    else:
        raise ValueError(
            f"Unknown model provider: '{chosen_provider}'. Supported: 'ollama', 'groq', 'gemini'."
        )


def test_model_connection(provider: str, model_name: str, base_url: str = "http://localhost:11434") -> Tuple[bool, str, float]:
    """Tests the connection to a specific model by issuing a quick prompt."""
    start_time = time.time()
    try:
        llm = get_llm(provider=provider, model_name=model_name)
        response = llm.invoke("Hi! Please confirm in under 10 words that you are online.")
        latency = round(time.time() - start_time, 2)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
        return True, content.strip(), latency
    except Exception as e:
        latency = round(time.time() - start_time, 2)
        return False, str(e), latency


async def atest_model_connection(provider: str, model_name: str, base_url: str = "http://localhost:11434") -> Tuple[bool, str, float]:
    """Asynchronously tests connection to a model by issuing a quick prompt without blocking."""
    start_time = time.time()
    try:
        llm = get_llm(provider=provider, model_name=model_name)
        response = await llm.ainvoke("Hi! Please confirm in under 10 words that you are online.")
        latency = round(time.time() - start_time, 2)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = " ".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
        return True, content.strip(), latency
    except Exception as e:
        latency = round(time.time() - start_time, 2)
        return False, str(e), latency


async def acheck_ollama_status(base_url: str = "http://localhost:11434") -> Tuple[bool, str]:
    """Asynchronously checks if the local Ollama server is reachable."""
    import asyncio
    return await asyncio.to_thread(check_ollama_status, base_url)


def prompt_select_model() -> Dict[str, Any]:
    """Interactively prompts the user on startup to select or confirm the LLM model."""
    config = load_model_config()
    current_prov = config.get("provider", "ollama").lower()
    current_model = config.get("model_name", "unknown")
    ollama_url = config.get("ollama_base_url", "http://localhost:11434")

    print("\n" + "=" * 60)
    print("               SELECT LLM MODEL FOR SESSION")
    print("=" * 60)
    print(f"Current Model: {current_model} ({current_prov.upper()})")
    print("\nSelect a provider:")
    print("  [1] Ollama (Local Model - Offline & Unlimited)")
    print("  [2] Groq (Cloud LLM - Ultra Fast)")
    print("  [3] Google Gemini (Cloud LLM)")
    print(f"  [4] Keep Current [{current_model}] (Press Enter)")

    try:
        choice = input("\nEnter choice (1-4) [default: 4]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nUsing current model configuration.")
        return config

    if choice == "1":
        is_up, _ = check_ollama_status(ollama_url)
        local_models = get_ollama_local_models(ollama_url) if is_up else []
        options = []
        if local_models:
            print("\nInstalled local models on your machine:")
            for idx, m in enumerate(local_models, 1):
                print(f"  [{idx}] {m}")
                options.append(m)
        else:
            print("\nRecommended Ollama models:")
            for idx, m in enumerate(DEFAULT_PRESETS["ollama"], 1):
                print(f"  [{idx}] {m}")
                options.append(m)
        custom_idx = len(options) + 1
        print(f"  [{custom_idx}] Custom model name...")

        sel = input(f"\nSelect model (1-{custom_idx}) [default: 1]: ").strip()
        if not sel:
            selected = options[0] if options else "llama3.2"
        elif sel.isdigit() and 1 <= int(sel) <= len(options):
            selected = options[int(sel) - 1]
        elif sel == str(custom_idx):
            selected = input("Enter model name: ").strip() or (options[0] if options else "llama3.2")
        else:
            selected = sel
        return save_model_config("ollama", selected)

    elif choice == "2":
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            print("\n[WARN] GROQ_API_KEY is not set in your .env file!")
            entered_key = input("Enter your Groq API Key (or press Enter to cancel): ").strip()
            if entered_key:
                os.environ["GROQ_API_KEY"] = entered_key
            else:
                print("Cancelled. Keeping current model.")
                return config

        presets = DEFAULT_PRESETS["groq"]
        print("\nAvailable Groq models:")
        for idx, m in enumerate(presets, 1):
            print(f"  [{idx}] {m}")
        custom_idx = len(presets) + 1
        print(f"  [{custom_idx}] Custom model name...")

        sel = input(f"\nSelect model (1-{custom_idx}) [default: 1]: ").strip()
        if not sel:
            selected = presets[0]
        elif sel.isdigit() and 1 <= int(sel) <= len(presets):
            selected = presets[int(sel) - 1]
        elif sel == str(custom_idx):
            selected = input("Enter model name: ").strip() or presets[0]
        else:
            selected = sel
        return save_model_config("groq", selected)

    elif choice == "3":
        gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            print("\n[WARN] GOOGLE_API_KEY is not set in your .env file!")
            entered_key = input("Enter your Google Gemini API Key (or press Enter to cancel): ").strip()
            if entered_key:
                os.environ["GOOGLE_API_KEY"] = entered_key
            else:
                print("Cancelled. Keeping current model.")
                return config

        presets = DEFAULT_PRESETS["gemini"]
        print("\nAvailable Gemini models:")
        for idx, m in enumerate(presets, 1):
            print(f"  [{idx}] {m}")
        custom_idx = len(presets) + 1
        print(f"  [{custom_idx}] Custom model name...")

        sel = input(f"\nSelect model (1-{custom_idx}) [default: 1]: ").strip()
        if not sel:
            selected = presets[0]
        elif sel.isdigit() and 1 <= int(sel) <= len(presets):
            selected = presets[int(sel) - 1]
        elif sel == str(custom_idx):
            selected = input("Enter model name: ").strip() or presets[0]
        else:
            selected = sel
        return save_model_config("gemini", selected)

    else:
        # Default or choice 4: Keep current
        print(f"[INFO] Continuing with {current_model} ({current_prov.upper()})")
        return config

