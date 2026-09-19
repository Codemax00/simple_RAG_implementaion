import os
import shutil
from pathlib import Path
from typing import List, Any, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader


def load_single_document(file_path: Path | str) -> List[Any]:
    """Loads a single document based on its file extension."""
    path = Path(file_path).resolve()
    if not path.is_file():
        print(f"[WARN] File not found: {path}")
        return []

    ext = path.suffix.lower()
    docs = []
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()
        elif ext in (".txt", ".md", ".log"):
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
        elif ext == ".csv":
            loader = CSVLoader(str(path), encoding="utf-8")
            docs = loader.load()
        else:
            print(f"[WARN] Unsupported file format: {ext}. Skipping {path.name}.")
    except Exception as e:
        print(f"[ERROR] Failed to load {path.name}: {e}")

    return docs


def load_all_documents(data_dir: str) -> List[Any]:
    """Load all supported files (PDF, TXT, CSV) from the data directory."""
    data_path = Path(data_dir).resolve()
    documents = []

    if not data_path.exists():
        return documents

    supported_patterns = ["*.pdf", "*.txt", "*.md", "*.csv"]
    for pattern in supported_patterns:
        for file in data_path.glob(pattern):
            try:
                loaded = load_single_document(file)
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Error loading {file.name}: {e}")

    return documents


def ingest_new_files(
    file_paths: List[Path | str],
    target_dir: str = "./documents",
    vm: Optional[Any] = None,
) -> int:
    """Checks if each file already exists in the Vector Database:
    - If already indexed: notifies the user and LLM that it is known, skipping re-chunking/embedding.
    - If NOT in vector database: chunks the document, generates embeddings, and saves into ChromaDB.
    """
    dest_dir = Path(target_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    files_to_process = []
    for item in file_paths:
        src = Path(item).resolve()
        if not src.exists():
            print(f"[WARN] Path does not exist: {src}")
            continue

        if src.is_dir():
            files_to_process.extend(
                [f for f in src.iterdir() if f.is_file() and f.suffix.lower() in (".pdf", ".txt", ".md", ".csv")]
            )
        else:
            files_to_process.append(src)

    if not files_to_process:
        print("[INFO] No valid document files found to process.")
        return 0

    total_new_indexed_chunks = 0

    for file in files_to_process:
        fname = file.name
        dest_file = dest_dir / fname

        # Check if already indexed in Vector Database
        if vm is not None and hasattr(vm, "is_document_indexed") and vm.is_document_indexed(fname):
            chunk_count = vm.get_document_chunk_count(fname)
            print(f"\n[ALREADY IN VECTOR DB] '{fname}' is already indexed ({chunk_count} chunks).")
            print(f"  -> The LLM already has full knowledge of '{fname}'. Skipping re-chunking & re-embedding.")
            continue

        # If not already present in ./documents, copy it
        if file.resolve() != dest_file.resolve():
            shutil.copy2(file, dest_file)
            print(f"\n[COPIED] {fname} -> {target_dir}/{fname}")
        else:
            print(f"\n[NEW FILE DETECTED] {fname}")

        # Extract document pages
        docs = load_single_document(dest_file)
        if not docs:
            print(f"[WARN] Could not extract any text from '{fname}'.")
            continue

        print(f"[EXTRACTED] Loaded {len(docs)} page(s) from '{fname}'.")

        # Chunk, make embeddings, and store in vector database
        if vm is not None:
            print(f"[CHUNKING & EMBEDDING] Generating embeddings for '{fname}' into ChromaDB...")
            chunks_indexed = vm.index_documents(docs)
            total_new_indexed_chunks += chunks_indexed
            print(f"[SUCCESS] '{fname}' indexed into Vector Database ({chunks_indexed} new chunks)!")
            print(f"  -> The LLM can now answer questions about '{fname}'.")

    return total_new_indexed_chunks


def prompt_upload_documents(vm: Any, docs_dir: str = "./documents") -> None:
    """Prompts the user to upload or add new documents on startup or in-session."""
    docs_path = Path(docs_dir).resolve()
    docs_path.mkdir(parents=True, exist_ok=True)
    existing_files = [f.name for f in docs_path.iterdir() if f.is_file() and not f.name.startswith(".")]

    print("\n" + "=" * 60)
    print("              DOCUMENT KNOWLEDGE BASE & UPLOAD")
    print("=" * 60)
    chunk_count = vm.count() if hasattr(vm, "count") else 0
    indexed_map = vm.get_indexed_documents() if hasattr(vm, "get_indexed_documents") else {}

    print(f"Vector Database Status: {chunk_count} total indexed chunks across {len(indexed_map)} files")
    if existing_files:
        print("Documents in Library:")
        for f in existing_files:
            c = indexed_map.get(f, 0)
            status = f"Already in Vector DB ({c} chunks)" if c > 0 else "Not yet indexed"
            print(f"  • {f} -> [{status}]")
    else:
        print("  (No documents currently loaded)")

    print("\nWould you like to add / upload documents to the library?")
    print("  [1] Enter file or folder path")
    print("  [2] Browse via Windows File Explorer")
    print("  [3] Skip & Continue (Press Enter)")

    try:
        choice = input("\nEnter choice (1-3) [default: 3]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nSkipping document upload.")
        return

    if choice == "1":
        path_str = input("Enter path to file or directory: ").strip().strip('"').strip("'")
        if path_str:
            p = Path(path_str)
            if p.exists():
                ingest_new_files([p], target_dir=docs_dir, vm=vm)
            else:
                print(f"[ERROR] Path not found: {path_str}")
        else:
            print("No path provided.")

    elif choice == "2":
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            print("[INFO] Opening File Explorer... Please choose your document(s).")
            selected = filedialog.askopenfilenames(
                title="Select Documents to Upload into RAG Library",
                filetypes=[
                    ("Supported Documents", "*.pdf;*.txt;*.md;*.csv"),
                    ("PDF Files", "*.pdf"),
                    ("Text Files", "*.txt;*.md"),
                    ("All Files", "*.*"),
                ],
            )
            root.destroy()

            if selected:
                print(f"[INFO] Selected {len(selected)} file(s).")
                ingest_new_files(list(selected), target_dir=docs_dir, vm=vm)
            else:
                print("[INFO] No files were selected.")
        except Exception as e:
            print(f"[WARN] File picker dialog failed: {e}")
            path_str = input("Please enter file path manually: ").strip().strip('"').strip("'")
            if path_str and Path(path_str).exists():
                ingest_new_files([Path(path_str)], target_dir=docs_dir, vm=vm)

    else:
        print("[INFO] Keeping existing library documents.")