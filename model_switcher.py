"""
Standalone Model Switcher CLI for Agentic RAG System
Allows switching between Ollama (Local), Groq (Cloud), and Google Gemini models.
"""

import os
import sys
import argparse

# Ensure Windows terminal prints Unicode / emojis / special symbols properly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.rag.models import (
    load_model_config,
    save_model_config,
    get_ollama_local_models,
    check_ollama_status,
    test_model_connection,
    DEFAULT_PRESETS,
)


def print_banner():
    print("=" * 65)
    print("           AGENTIC RAG - MODEL SWITCHER & MANAGER")
    print("=" * 65)


def display_status():
    config = load_model_config()
    provider = config.get("provider", "unknown").upper()
    model = config.get("model_name", "unknown")
    temp = config.get("temperature", 0.2)
    ollama_url = config.get("ollama_base_url", "http://localhost:11434")

    print("\n[ACTIVE CONFIGURATION]")
    print(f"  • Provider:    {provider}")
    print(f"  • Model:       {model}")
    print(f"  • Temperature: {temp}")
    if provider == "OLLAMA":
        print(f"  • Base URL:    {ollama_url}")

    print("\n[PROVIDER HEALTH STATUS]")
    # Ollama status
    is_up, msg = check_ollama_status(ollama_url)
    if is_up:
        models = get_ollama_local_models(ollama_url)
        print(f"  [+] Ollama (Local):   ONLINE (Found {len(models)} installed local model(s))")
    else:
        print(f"  [-] Ollama (Local):   OFFLINE ({msg})")

    # Groq status
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"  [+] Groq (Cloud):     READY (API Key configured: ...{groq_key[-6:]})")
    else:
        print("  [-] Groq (Cloud):     NOT CONFIGURED (GROQ_API_KEY not found in .env)")

    # Gemini status
    gemini_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print(f"  [+] Gemini (Cloud):   READY (API Key configured: ...{gemini_key[-6:]})")
    else:
        print("  [-] Gemini (Cloud):   NOT CONFIGURED (GOOGLE_API_KEY not found in .env)")
    print("-" * 65)


def switch_to_ollama():
    config = load_model_config()
    ollama_url = config.get("ollama_base_url", "http://localhost:11434")

    print("\n--- SWITCH TO OLLAMA (LOCAL MODEL) ---")
    is_up, msg = check_ollama_status(ollama_url)

    detected = []
    if is_up:
        detected = get_ollama_local_models(ollama_url)
        print(f"[INFO] Ollama service detected at {ollama_url}")
    else:
        print(f"[WARN] {msg}")
        print("Tip: You can still choose an Ollama model, but ensure `ollama serve` is running before querying.")

    choices = []
    if detected:
        print("\nInstalled local models on your machine:")
        for idx, m in enumerate(detected, start=1):
            print(f"  [{idx}] {m} (installed)")
            choices.append(m)
    else:
        print("\nRecommended Ollama models:")
        for idx, m in enumerate(DEFAULT_PRESETS["ollama"], start=1):
            print(f"  [{idx}] {m}")
            choices.append(m)

    custom_idx = len(choices) + 1
    print(f"  [{custom_idx}] Custom model name...")
    print("  [0] Cancel")

    sel = input(f"\nSelect an option (0-{custom_idx}): ").strip()
    if not sel or sel == "0":
        print("Cancelled.")
        return

    try:
        val = int(sel)
        if 1 <= val <= len(choices):
            selected_model = choices[val - 1]
        elif val == custom_idx:
            selected_model = input("Enter custom Ollama model name (e.g. llama3.2, mistral:7b): ").strip()
            if not selected_model:
                print("Invalid model name.")
                return
        else:
            print("Invalid choice.")
            return
    except ValueError:
        selected_model = sel

    save_model_config(provider="ollama", model_name=selected_model, ollama_base_url=ollama_url)
    print(f"\n[ACTIVE MODEL UPDATED] Provider set to OLLAMA with model '{selected_model}'.")


def switch_to_groq():
    print("\n--- SWITCH TO GROQ (CLOUD MODEL) ---")
    if not os.getenv("GROQ_API_KEY"):
        print("[WARN] GROQ_API_KEY is not set in your .env file!")
        proceed = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if proceed != "y":
            return

    presets = DEFAULT_PRESETS["groq"]
    print("\nAvailable Groq Models:")
    for idx, m in enumerate(presets, start=1):
        print(f"  [{idx}] {m}")
    custom_idx = len(presets) + 1
    print(f"  [{custom_idx}] Custom model name...")
    print("  [0] Cancel")

    sel = input(f"\nSelect an option (0-{custom_idx}): ").strip()
    if not sel or sel == "0":
        print("Cancelled.")
        return

    try:
        val = int(sel)
        if 1 <= val <= len(presets):
            selected_model = presets[val - 1]
        elif val == custom_idx:
            selected_model = input("Enter custom Groq model name: ").strip()
            if not selected_model:
                print("Invalid model name.")
                return
        else:
            print("Invalid choice.")
            return
    except ValueError:
        selected_model = sel

    save_model_config(provider="groq", model_name=selected_model)
    print(f"\n[ACTIVE MODEL UPDATED] Provider set to GROQ with model '{selected_model}'.")


def switch_to_gemini():
    print("\n--- SWITCH TO GOOGLE GEMINI (CLOUD MODEL) ---")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("[WARN] GOOGLE_API_KEY is not set in your .env file!")
        proceed = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if proceed != "y":
            return

    presets = DEFAULT_PRESETS["gemini"]
    print("\nAvailable Gemini Models:")
    for idx, m in enumerate(presets, start=1):
        print(f"  [{idx}] {m}")
    custom_idx = len(presets) + 1
    print(f"  [{custom_idx}] Custom model name...")
    print("  [0] Cancel")

    sel = input(f"\nSelect an option (0-{custom_idx}): ").strip()
    if not sel or sel == "0":
        print("Cancelled.")
        return

    try:
        val = int(sel)
        if 1 <= val <= len(presets):
            selected_model = presets[val - 1]
        elif val == custom_idx:
            selected_model = input("Enter custom Gemini model name: ").strip()
            if not selected_model:
                print("Invalid model name.")
                return
        else:
            print("Invalid choice.")
            return
    except ValueError:
        selected_model = sel

    save_model_config(provider="gemini", model_name=selected_model)
    print(f"\n[ACTIVE MODEL UPDATED] Provider set to GEMINI with model '{selected_model}'.")


def test_active_connection():
    config = load_model_config()
    provider = config.get("provider", "groq")
    model = config.get("model_name", "")
    print(f"\n[TESTING] Testing connection to {provider.upper()} ({model})...")
    ok, res, latency = test_model_connection(provider, model)
    if ok:
        print(f"[SUCCESS] Model replied in {latency}s:")
        print(f"          \"{res}\"")
    else:
        print(f"[FAILED] Could not connect to model (after {latency}s):")
        print(f"         {res}")


def launch_rag_app():
    print("\n[LAUNCHING] Starting Agentic RAG System...")
    import subprocess
    cmd = [sys.executable, "app.py"]
    subprocess.run(cmd)


def interactive_menu():
    while True:
        print_banner()
        display_status()
        print("\nSelect an action:")
        print("  1. Switch to Ollama (Local Model)")
        print("  2. Switch to Groq (Cloud LLM)")
        print("  3. Switch to Google Gemini (Cloud LLM)")
        print("  4. Test Connection to Active Model")
        print("  5. Launch Agentic RAG System (app.py)")
        print("  6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()
        if choice == "1":
            switch_to_ollama()
        elif choice == "2":
            switch_to_groq()
        elif choice == "3":
            switch_to_gemini()
        elif choice == "4":
            test_active_connection()
        elif choice == "5":
            launch_rag_app()
            break
        elif choice in ("6", "exit", "quit", "q"):
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

        input("\nPress Enter to continue...")


def main():
    parser = argparse.ArgumentParser(description="Model Switcher for Agentic RAG")
    parser.add_argument("--status", action="store_true", help="Display active model and provider status")
    parser.add_argument("--set", nargs=2, metavar=("PROVIDER", "MODEL"), help="Set provider and model (e.g. --set ollama llama3.2)")
    parser.add_argument("--test", action="store_true", help="Test connection to currently active model")
    parser.add_argument("--run", action="store_true", help="Launch app.py directly")
    args = parser.parse_args()

    if args.set:
        prov, mod = args.set
        save_model_config(provider=prov, model_name=mod)
        display_status()
        return

    if args.status:
        print_banner()
        display_status()
        return

    if args.test:
        test_active_connection()
        return

    if args.run:
        launch_rag_app()
        return

    interactive_menu()


if __name__ == "__main__":
    main()
