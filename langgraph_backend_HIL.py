from __future__ import annotations

import ast
import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    TextLoader,
)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import interrupt, Command
import requests

load_dotenv()


llm = ChatMistralAI(model='mistral-small-2506')

embeddings = MistralAIEmbeddings(model='mistral-embed')


# -------------------
# 2. Document retriever store (per thread), with on-disk persistence
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_VECTOR_STORES: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}
_THREAD_USAGE: Dict[str, Dict[str, int]] = {}

VECTOR_STORE_DIR = Path("vector_stores")
VECTOR_STORE_DIR.mkdir(exist_ok=True)


def _store_path(thread_id: str) -> Path:
    return VECTOR_STORE_DIR / str(thread_id)


def _save_metadata(thread_id: str) -> None:
    path = _store_path(thread_id)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "metadata.json", "w") as f:
        json.dump(_THREAD_METADATA.get(str(thread_id), {}), f)


def _load_metadata(thread_id: str) -> dict:
    path = _store_path(thread_id) / "metadata.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread, loading a persisted index from disk if needed."""
    if not thread_id:
        return None
    tid = str(thread_id)

    if tid in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[tid]

    store_dir = _store_path(tid)
    if store_dir.exists():
        try:
            vector_store = FAISS.load_local(
                str(store_dir), embeddings, allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            _THREAD_RETRIEVERS[tid] = retriever
            _THREAD_VECTOR_STORES[tid] = vector_store
            if tid not in _THREAD_METADATA:
                loaded_meta = _load_metadata(tid)
                if loaded_meta:
                    _THREAD_METADATA[tid] = loaded_meta
            return retriever
        except Exception:
            return None

    return None


def _load_documents(temp_path: str, ext: str):
    """Dispatch to the right loader based on file extension."""
    ext = ext.lower()
    if ext == ".pdf":
        return PyPDFLoader(temp_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(temp_path).load()
    elif ext == ".csv":
        return CSVLoader(temp_path).load()
    elif ext in (".txt", ".md"):
        return TextLoader(temp_path, encoding="utf-8").load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_document(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build/extend a FAISS retriever for the uploaded document and persist it for the thread.
    Supports PDF, DOCX, CSV, and TXT. If a document is already indexed for this thread,
    the new document's chunks are merged into the existing index instead of overwriting it.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    tid = str(thread_id)
    ext = os.path.splitext(filename or "")[1] or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        docs = _load_documents(temp_path, ext)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source_file"] = filename

        # Reuse existing vector store for this thread if one exists (in-memory or on disk).
        existing_store = _THREAD_VECTOR_STORES.get(tid)
        if existing_store is None:
            _get_retriever(tid)
            existing_store = _THREAD_VECTOR_STORES.get(tid)

        if existing_store is not None:
            existing_store.add_documents(chunks)
            vector_store = existing_store
        else:
            vector_store = FAISS.from_documents(chunks, embeddings)

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        _THREAD_RETRIEVERS[tid] = retriever
        _THREAD_VECTOR_STORES[tid] = vector_store

        store_dir = _store_path(tid)
        store_dir.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(store_dir))

        meta = _THREAD_METADATA.get(tid) or _load_metadata(tid) or {"files": [], "documents": 0, "chunks": 0}
        meta.setdefault("files", [])
        if filename and filename not in meta["files"]:
            meta["files"].append(filename)
        meta["documents"] = meta.get("documents", 0) + len(docs)
        meta["chunks"] = meta.get("chunks", 0) + len(chunks)
        meta["filename"] = filename  # most recently added
        _THREAD_METADATA[tid] = meta
        _save_metadata(tid)

        return meta
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# Backward-compatible alias (frontend / older callers may still import ingest_pdf)
ingest_pdf = ingest_document


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


def request_approval(tool_name: str, question: str, details: Optional[dict] = None):
    """
    Shared helper for any tool that needs human-in-the-loop approval.
    Pauses the graph via interrupt() with a structured payload the UI can render.
    """
    payload = {"tool": tool_name, "question": question, "details": details or {}}
    return interrupt(payload)


def _is_approved(decision: Any) -> bool:
    text = decision if isinstance(decision, str) else str(decision)
    return text.strip().lower() in ("yes", "y", "approve", "approved")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    r = requests.get(url)
    return r.json()


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for human decision ("yes" / anything else).
    """
    decision = request_approval(
        "purchase_stock",
        f"Approve buying {quantity} share(s) of {symbol}?",
        {"symbol": symbol, "quantity": quantity},
    )

    if _is_approved(decision):
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
            "human_response": decision if isinstance(decision, str) else str(decision),
        }


@tool
def send_email(to: str, subject: str, body: str) -> dict:
    """
    Simulate sending an email to a recipient.

    HUMAN-IN-THE-LOOP:
    Email delivery is irreversible, so this tool interrupts and waits for
    human approval before "sending" ("yes" / anything else).
    """
    decision = request_approval(
        "send_email",
        f"Approve sending an email to {to} with subject '{subject}'?",
        {"to": to, "subject": subject, "body": body},
    )

    if _is_approved(decision):
        return {
            "status": "sent",
            "message": f"Email sent to {to}.",
            "to": to,
            "subject": subject,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Email to {to} was declined by human.",
            "to": to,
            "subject": subject,
            "human_response": decision if isinstance(decision, str) else str(decision),
        }


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the documents uploaded for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a document first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, get_stock_price, purchase_stock, send_email, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _record_usage(thread_id: Optional[str], response) -> None:
    """Track token usage per thread for a running usage/cost estimate."""
    if not thread_id:
        return
    tid = str(thread_id)

    usage = getattr(response, "usage_metadata", None)
    if not usage:
        response_metadata = getattr(response, "response_metadata", None) or {}
        usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    if not usage:
        return

    input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    output_tokens = usage.get("output_tokens") or usage.get("completion_tokens") or 0
    total_tokens = usage.get("total_tokens") or (input_tokens + output_tokens)

    bucket = _THREAD_USAGE.setdefault(tid, {"input": 0, "output": 0, "total": 0})
    bucket["input"] += input_tokens
    bucket["output"] += output_tokens
    bucket["total"] += total_tokens


def get_thread_usage(thread_id: str) -> Dict[str, int]:
    return _THREAD_USAGE.get(str(thread_id), {"input": 0, "output": 0, "total": 0})


def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about uploaded documents, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, purchase_stock, "
            "send_email, and calculator tools when helpful. purchase_stock and send_email "
            "require human approval before completing, so warn the user that a confirmation "
            "Only mention uploading a document (PDF, DOCX, CSV, or TXT) if the user asks a "
            "question that needs a document and `rag_tool` reports none is indexed for this "
            "thread. Do not mention document upload, supported file types, or document "
            "capabilities in greetings, small talk, or any other unrelated reply."

        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    _record_usage(thread_id, response)
    return {"messages": [response]}


tool_node = ToolNode(tools)

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS or _store_path(str(thread_id)).exists()


def thread_document_metadata(thread_id: str) -> dict:
    tid = str(thread_id)
    if tid not in _THREAD_METADATA:
        loaded = _load_metadata(tid)
        if loaded:
            _THREAD_METADATA[tid] = loaded
    return _THREAD_METADATA.get(tid, {})


def delete_thread(thread_id: str) -> None:
    """Remove a thread's checkpoints, vector store, and cached state."""
    tid = str(thread_id)

    try:
        cur = conn.cursor()
        for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs", "writes"):
            try:
                cur.execute(f"DELETE FROM {table} WHERE thread_id = ?", (tid,))
            except sqlite3.OperationalError:
                pass
        conn.commit()
    except Exception:
        pass

    _THREAD_RETRIEVERS.pop(tid, None)
    _THREAD_VECTOR_STORES.pop(tid, None)
    _THREAD_METADATA.pop(tid, None)
    _THREAD_USAGE.pop(tid, None)

    store_dir = _store_path(tid)
    if store_dir.exists():
        shutil.rmtree(store_dir, ignore_errors=True)


def extract_rag_sources(raw_content: Any):
    """
    Parse a rag_tool ToolMessage's content (str or dict) into a simple list of
    {"page": ..., "source": ...} dicts for citation display in the UI.
    """
    try:
        data = raw_content
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = ast.literal_eval(data)

        if not isinstance(data, dict):
            return None

        metadata = data.get("metadata", [])
        sources = []
        for m in metadata:
            if isinstance(m, dict):
                sources.append(
                    {
                        "page": m.get("page"),
                        "source": m.get("source_file") or m.get("source"),
                    }
                )
        return sources or None
    except Exception:
        return None
