import ast
import io
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool
from src.rag.vectorstore import VectorStoreManager
from src.rag.tools import make_rag_tool


class ListDocsInput(BaseModel):
    filter_keyword: str = Field(default="", description="Optional keyword to filter documents by, or empty to list all.")


class CalculateInput(BaseModel):
    expression: str = Field(description="A valid Python mathematical expression (e.g. '48 * 12', 'math.sqrt(144)').")


class RunPythonInput(BaseModel):
    code: str = Field(description="Python source code snippet to execute.")


class CreateToolInput(BaseModel):
    tool_name: str = Field(description="The exact name of the Python function being defined.")
    description: str = Field(description="Clear explanation of what this tool does and its parameters.")
    python_code: str = Field(description="The Python function definition code.")


class PDFReportInput(BaseModel):
    title: str = Field(description="Title of the PDF report.")
    content: str = Field(description="The body text or bulleted facts to put in the PDF report.")
    filename: str = Field(default="generated_rag_report.pdf", description="Filename for the PDF, e.g. 'facts_report.pdf'.")


class ToolRegistry:
    """Registry that holds both static and dynamically created tools."""

    def __init__(self, vectorstore_manager: Optional[VectorStoreManager] = None):
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager()
        self.dynamic_tools: Dict[str, Callable] = {}

    def register_tool(self, name: str, fn: Callable):
        self.dynamic_tools[name] = fn


def create_tool_suite(registry: Optional[ToolRegistry] = None) -> List[Callable]:
    """Creates the full suite of tools including RAG, utilities, and dynamic tool creation."""
    reg = registry or ToolRegistry()

    # 1. RAG Search Tool
    rag_tool = make_rag_tool(reg.vectorstore_manager)

    # 2. List Available Documents Tool
    @tool(args_schema=ListDocsInput)
    def list_available_documents(filter_keyword: str = "") -> str:
        """List all indexed books and document sources currently loaded in the vector database (ChromaDB),
        along with the total number of indexed passages and chunk counts per document.
        """
        try:
            vm = reg.vectorstore_manager
            if vm and hasattr(vm, "get_indexed_documents"):
                indexed_map = vm.get_indexed_documents()
            else:
                docs_dir = Path("./documents")
                indexed_map = {f.name: 0 for f in docs_dir.glob("*") if f.is_file()}

            if filter_keyword:
                indexed_map = {k: v for k, v in indexed_map.items() if filter_keyword.lower() in k.lower()}

            total_chunks = vm.count() if vm else 0
            if indexed_map:
                file_lines = [f"- {fname} ({chunks} indexed chunks in Vector DB)" for fname, chunks in indexed_map.items()]
                file_list = "\n".join(file_lines)
            else:
                file_list = "No matching documents found in Vector Database."

            return (
                f"Vector Database (ChromaDB) Knowledge Status:\n"
                f"Total Indexed Passages: {total_chunks}\n\n"
                f"Currently Indexed Documents in Vector Store:\n{file_list}"
            )
        except Exception as e:
            return f"Error inspecting vector database: {e}"

    async def alist_available_documents(filter_keyword: str = "") -> str:
        import asyncio
        return await asyncio.to_thread(list_available_documents.invoke, {"filter_keyword": filter_keyword})

    list_available_documents.coroutine = alist_available_documents

    # 3. Safe Math / Calculation Tool
    @tool(args_schema=CalculateInput)
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression. Useful for quantitative comparisons,
        statistics, percentages, or counting calculations.
        """
        try:
            safe_dict = {
                "math": math,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
            }
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return f"Result: {result}"
        except Exception as e:
            return f"Error evaluating expression '{expression}': {e}"

    # 4. Python Code Execution Tool
    @tool(args_schema=RunPythonInput)
    def run_python_code(code: str) -> str:
        """Execute a snippet of Python code and return the standard output (print statements).
        Useful for advanced text manipulation, sorting, filtering, or analyzing retrieved data.
        """
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        try:
            local_vars = {}
            exec(code, {"math": math}, local_vars)
            output = redirected_output.getvalue().strip()
            return output if output else "Code executed successfully with no printed output."
        except Exception as e:
            return f"Code Execution Error: {type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout

    # 5. Dynamic Tool Creation Tool (Meta-Tool)
    @tool(args_schema=CreateToolInput)
    def create_custom_tool(tool_name: str, description: str, python_code: str) -> str:
        """Dynamically create and register a new Python tool for your own use.
        Define a Python function with the name matching `tool_name`.

        Args:
            tool_name: The exact name of the Python function being defined.
            description: Clear explanation of what this tool does and its parameters.
            python_code: The Python function definition code, including docstrings.

        Example python_code:
            def count_word_frequency(text: str, target: str) -> str:
                words = text.lower().split()
                count = words.count(target.lower())
                return f"The word '{target}' occurs {count} times."
        """
        try:
            # Parse syntax to ensure it's valid Python
            ast.parse(python_code)

            # Execute in a safe namespace
            namespace = {"math": math, "sys": sys}
            exec(python_code, namespace)

            if tool_name not in namespace or not callable(namespace[tool_name]):
                return f"Error: Function '{tool_name}' was not found in the provided code."

            fn = namespace[tool_name]
            fn.__doc__ = description
            reg.register_tool(tool_name, fn)

            return (
                f"Successfully created and registered tool '{tool_name}'!\n"
                f"Description: {description}\n"
                f"You can now call it using `run_python_code` or invoke it directly."
            )
        except SyntaxError as se:
            return f"Syntax Error while compiling tool '{tool_name}': {se}"
        except Exception as e:
            return f"Failed to create tool '{tool_name}': {e}"

    # 6. PDF Report Generator Tool
    @tool(args_schema=PDFReportInput)
    def generate_pdf_report(title: str, content: str, filename: str = "generated_rag_report.pdf") -> str:
        """Create and export a professional PDF report file on disk.
        Use this tool when the user asks to save, export, or generate a PDF file of facts, summaries, or reports.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

            clean_filename = filename.strip()
            if not clean_filename.endswith(".pdf"):
                clean_filename += ".pdf"

            pdf_path = os.path.abspath(clean_filename)
            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=letter,
                leftMargin=54,
                rightMargin=54,
                topMargin=54,
                bottomMargin=54,
            )
            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                "PDFTitle",
                parent=styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=18,
                leading=22,
                textColor=colors.HexColor("#0f172a"),
                spaceAfter=10,
            )
            body_style = ParagraphStyle(
                "PDFBody",
                parent=styles["Normal"],
                fontName="Helvetica",
                fontSize=10,
                leading=14,
                textColor=colors.HexColor("#334155"),
                spaceAfter=6,
            )

            story = [
                Paragraph(title, title_style),
                HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceBefore=4, spaceAfter=12),
            ]
            for para in content.split("\n\n"):
                clean_p = para.strip().replace("\n", "<br/>")
                if clean_p:
                    story.append(Paragraph(clean_p, body_style))
                    story.append(Spacer(1, 4))

            doc.build(story)
            return f"Successfully generated PDF report at: {pdf_path}"
        except Exception as e:
            return f"Error creating PDF: {e}"

    async def agenerate_pdf_report(title: str, content: str, filename: str = "generated_rag_report.pdf") -> str:
        import asyncio
        return await asyncio.to_thread(generate_pdf_report.invoke, {"title": title, "content": content, "filename": filename})

    generate_pdf_report.coroutine = agenerate_pdf_report

    # Alias tools for model calling flexibility
    @tool(args_schema=PDFReportInput)
    def create_pdf(title: str, content: str, filename: str = "generated_rag_report.pdf") -> str:
        """Create and export a PDF report file on disk."""
        return generate_pdf_report.invoke({"title": title, "content": content, "filename": filename})

    @tool(args_schema=PDFReportInput)
    def write_pdf(title: str, content: str, filename: str = "generated_rag_report.pdf") -> str:
        """Write and export a PDF report file on disk."""
        return generate_pdf_report.invoke({"title": title, "content": content, "filename": filename})

    return [
        rag_tool,
        list_available_documents,
        calculate,
        run_python_code,
        create_custom_tool,
        generate_pdf_report,
        create_pdf,
        write_pdf,
    ]
