from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from src.rag.vectorstore import VectorStoreManager


class SearchInput(BaseModel):
    query: str = Field(description="The semantic search query or topic to look up in the documents.")


def format_search_results(results: List[Document]) -> str:
    if not results:
        return "No relevant information found in the knowledge base."

    formatted = []
    for idx, doc in enumerate(results, 1):
        source_raw = doc.metadata.get("source", "Document")
        source_name = Path(source_raw).name
        page_num = doc.metadata.get("page", None)
        page_info = f", Page {page_num + 1}" if page_num is not None else ""

        content = doc.page_content.strip()
        if len(content) > 220:
            content = content[:220] + "..."

        formatted.append(f"[{source_name}{page_info}]: {content}")

    return "\n\n".join(formatted)


def make_rag_tool(vectorstore_manager: Optional[VectorStoreManager] = None) -> StructuredTool:
    """Creates a callable LangChain tool with both sync and async implementations."""
    vm = vectorstore_manager or VectorStoreManager()

    def search_knowledge_base(query: str) -> str:
        results = vm.similarity_search(query, k=2)
        return format_search_results(results)

    async def asearch_knowledge_base(query: str) -> str:
        results = await vm.asimilarity_search(query, k=2)
        return format_search_results(results)

    return StructuredTool.from_function(
        func=search_knowledge_base,
        coroutine=asearch_knowledge_base,
        name="search_knowledge_base",
        description=(
            "Search the library of documents for relevant passages, strategies, "
            "psychological concepts, and historical examples. Returns formatted context snippets with source and page citations."
        ),
        args_schema=SearchInput,
    )
