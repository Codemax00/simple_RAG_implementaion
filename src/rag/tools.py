from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from src.rag.vectorstore import VectorStoreManager


class SearchInput(BaseModel):
    query: str = Field(description="The semantic search query or topic to look up in the documents.")


def make_rag_tool(vectorstore_manager: Optional[VectorStoreManager] = None):
    """Creates a callable LangChain tool that searches the knowledge base."""
    vm = vectorstore_manager or VectorStoreManager()

    @tool(args_schema=SearchInput)
    def search_knowledge_base(query: str) -> str:
        """Search the library of documents (e.g., The 33 Strategies of War,
        The 48 Laws of Power, The Molecule of More) for relevant passages,
        strategies, psychological concepts, and historical examples.

        Args:
            query: The semantic search query or topic to look up.

        Returns:
            Formatted context snippets with source document names and page numbers.
        """
        results = vm.similarity_search(query, k=2)
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

            formatted.append(
                f"[{source_name}{page_info}]: {content}"
            )

        return "\n\n".join(formatted)

    return search_knowledge_base
