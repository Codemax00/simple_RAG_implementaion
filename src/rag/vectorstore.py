import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer


class LocalSentenceTransformerEmbeddings(Embeddings):
    """Wrapper around SentenceTransformer for LangChain compatibility."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class VectorStoreManager:
    """Manages the persistent Chroma vector store."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_knowledge_base",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_fn = LocalSentenceTransformerEmbeddings(model_name=model_name)
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_fn,
        )

    def is_populated(self) -> bool:
        """Checks if the vector store already contains indexed documents."""
        try:
            existing = self.vectorstore.get(limit=1)
            return len(existing["ids"]) > 0
        except Exception:
            return False

    def count(self) -> int:
        """Return the number of items stored in ChromaDB."""
        try:
            return len(self.vectorstore.get()["ids"])
        except Exception:
            return 0

    def get_indexed_documents(self) -> dict:
        """Returns a dict mapping filename -> number of chunks currently stored in ChromaDB."""
        try:
            data = self.vectorstore.get(include=["metadatas"])
            counts = {}
            for m in data.get("metadatas", []):
                if m and "source" in m:
                    fname = Path(m["source"]).name
                    counts[fname] = counts.get(fname, 0) + 1
            return counts
        except Exception:
            return {}

    def is_document_indexed(self, filename: str) -> bool:
        """Checks if a given file name is already present in ChromaDB."""
        target_name = Path(filename).name.lower()
        indexed = self.get_indexed_documents()
        return any(k.lower() == target_name for k in indexed.keys())

    def get_document_chunk_count(self, filename: str) -> int:
        """Returns the number of chunks indexed for a specific file name."""
        target_name = Path(filename).name.lower()
        indexed = self.get_indexed_documents()
        for k, v in indexed.items():
            if k.lower() == target_name:
                return v
        return 0

    def index_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        batch_size: int = 200,
    ) -> int:
        """Chunks and stores documents into ChromaDB in batches with persistence."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")

        # Batch indexing into Chroma
        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            self.vectorstore.add_documents(batch)
            print(f" [INFO] Indexed chunks {i + 1}-{min(i + batch_size, total_chunks)} of {total_chunks}")

        return total_chunks

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Performs semantic similarity search on the knowledge base."""
        return self.vectorstore.similarity_search(query, k=k)

    async def asimilarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Asynchronously performs semantic similarity search without blocking."""
        import asyncio
        return await asyncio.to_thread(self.similarity_search, query, k)

    async def aindex_document_stream(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        micro_batch_size: int = 10,
    ) -> int:
        """Asynchronously streams chunks directly to the embedding layer in real-time.

        The very first chunk of the document is immediately dispatched to embedding and
        persisted to ChromaDB without waiting for subsequent chunks or pages to be split.
        Chunking and embedding execute concurrently as an async Producer-Consumer pipeline.
        """
        import asyncio

        queue: asyncio.Queue[Optional[Document]] = asyncio.Queue(maxsize=100)
        indexed_count = 0
        total_chunks_produced = 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        async def chunker_producer():
            nonlocal total_chunks_produced
            chunk_index = 0
            for doc in documents:
                doc_chunks = splitter.split_documents([doc])
                for chunk in doc_chunks:
                    chunk_index += 1
                    total_chunks_produced = chunk_index
                    if chunk_index == 1:
                        print("[ASYNC PIPELINE] Chunk 1 created -> directly passing to embedding layer...")
                    await queue.put(chunk)
            # Signal completion
            await queue.put(None)

        async def embedder_consumer():
            nonlocal indexed_count
            first_chunk_processed = False
            batch: List[Document] = []

            while True:
                item = await queue.get()
                if item is None:
                    # Flush remaining batch
                    if batch:
                        await asyncio.to_thread(self.vectorstore.add_documents, batch)
                        indexed_count += len(batch)
                        print(f"[ASYNC PIPELINE] Final batch indexed ({indexed_count} total chunks in ChromaDB).")
                    queue.task_done()
                    break

                # The first chunk is directly and immediately embedded with 0 delay
                if not first_chunk_processed:
                    await asyncio.to_thread(self.vectorstore.add_documents, [item])
                    indexed_count += 1
                    first_chunk_processed = True
                    print("[ASYNC PIPELINE] Chunk 1 embedded and saved into ChromaDB! (Streaming remainder...)")
                    queue.task_done()
                    continue

                batch.append(item)
                queue.task_done()

                # Process in micro-batches or whenever queue is empty to maintain steady flow
                if len(batch) >= micro_batch_size or queue.empty():
                    await asyncio.to_thread(self.vectorstore.add_documents, batch)
                    indexed_count += len(batch)
                    print(f"[ASYNC PIPELINE] Concurrently embedded chunks up to {indexed_count}...")
                    batch = []

        await asyncio.gather(chunker_producer(), embedder_consumer())
        return indexed_count

    async def aindex_documents(
        self,
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        batch_size: int = 200,
    ) -> int:
        """Asynchronously chunks and indexes documents using the streaming pipeline."""
        return await self.aindex_document_stream(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
