import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph.types import Command
from langgraph_backend_HIL import (
    chatbot,
    delete_thread,
    extract_rag_sources,
    get_thread_usage,
    ingest_document,
    retrieve_all_threads,
    thread_document_metadata,
)




# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def make_title(text, max_words=6):
    words = text.strip().split()
    if not words:
        return "New chat"
    title = " ".join(words[:max_words])
    if len(words) > max_words:
        title += "…"
    return title


def get_interrupt_value(state):
    """Pull the structured payload passed to interrupt() out of the graph state."""
    if not state.tasks:
        return None
    for task in state.tasks:
        if getattr(task, "interrupts", None):
            return task.interrupts[0].value
    return None




# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

# HIL Initialisation
if "pending_hil" not in st.session_state:
    st.session_state["pending_hil"] = False

if "pending_config" not in st.session_state:
    st.session_state["pending_config"] = None

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None




# ============================ Sidebar ============================
st.sidebar.title("LangGraph Multi-Utility Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# Document status (supports multiple files per thread now)
doc_meta = thread_document_metadata(thread_key)
files = doc_meta.get("files") or ([doc_meta["filename"]] if doc_meta.get("filename") else [])
if files:
    st.sidebar.success(
        f"{len(files)} file(s) indexed ({doc_meta.get('chunks', 0)} chunks, "
        f"{doc_meta.get('documents', 0)} pages/rows total)\n\n"
        + "\n".join(f"- `{f}`" for f in files)
    )
else:
    st.sidebar.info("No document indexed yet.")

uploaded_files = st.sidebar.file_uploader(
    "Upload document(s) for this chat",
    type=["pdf", "docx", "csv", "txt"],
    accept_multiple_files=True,
)
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name in thread_docs:
            continue
        with st.sidebar.status(f"Indexing {uploaded_file.name}…", expanded=True) as status_box:
            summary = ingest_document(
                uploaded_file.getvalue(),
                thread_id=thread_key,
                filename=uploaded_file.name,
            )
            thread_docs[uploaded_file.name] = summary
            status_box.update(label=f"✅ {uploaded_file.name} indexed", state="complete", expanded=False)
    st.rerun()

# Token usage for this thread
usage = get_thread_usage(thread_key)
if usage.get("total"):
    st.sidebar.caption(
        f"🔢 Tokens — in: {usage['input']} · out: {usage['output']} · total: {usage['total']}"
    )

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for tid in threads:
        title = st.session_state["thread_titles"].get(str(tid), str(tid))
        col_a, col_b = st.sidebar.columns([5, 1])
        with col_a:
            if st.button(title, key=f"side-thread-{tid}", use_container_width=True):
                selected_thread = tid
        with col_b:
            if st.button("🗑️", key=f"del-thread-{tid}", help="Delete this conversation"):
                delete_thread(tid)
                st.session_state["chat_threads"] = [
                    t for t in st.session_state["chat_threads"] if str(t) != str(tid)
                ]
                st.session_state["thread_titles"].pop(str(tid), None)
                st.session_state["ingested_docs"].pop(str(tid), None)
                if str(tid) == thread_key:
                    reset_chat()
                st.rerun()




# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for s in message["sources"]:
                    page = s.get("page")
                    src = s.get("source") or "document"
                    st.caption(f"`{src}`" + (f" — page {page}" if page is not None else ""))

user_input = st.chat_input("Ask about your document or use tools")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Auto-title the thread from the first message
    if thread_key not in st.session_state["thread_titles"]:
        st.session_state["thread_titles"][thread_key] = make_title(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}
        sources_holder = {"sources": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    if tool_name == "rag_tool":
                        sources_holder["sources"] = extract_rag_sources(message_chunk.content)

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

        if sources_holder["sources"]:
            with st.expander("📄 Sources"):
                for s in sources_holder["sources"]:
                    page = s.get("page")
                    src = s.get("source") or "document"
                    st.caption(f"`{src}`" + (f" — page {page}" if page is not None else ""))

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message, "sources": sources_holder["sources"]}
    )

    # Check whether the graph is waiting for human approval
    state = chatbot.get_state(config=CONFIG)

    if state.next:
        st.session_state["pending_hil"] = True
        st.session_state["pending_config"] = CONFIG
    else:
        st.session_state["pending_hil"] = False

st.divider()



# ============================ HIL UI ==============================
if st.session_state.get("pending_hil", False):

    pending_config = st.session_state["pending_config"]
    state = chatbot.get_state(config=pending_config)
    interrupt_value = get_interrupt_value(state)

    st.warning("⚠️ Assistant is waiting for your approval.")

    if isinstance(interrupt_value, dict):
        st.markdown(f"**{interrupt_value.get('question', 'Approval required')}**")
        if interrupt_value.get("details"):
            st.json(interrupt_value["details"])
    elif interrupt_value:
        st.markdown(f"**{interrupt_value}**")

    def resume_stream(resume_value):
        for message_chunk, _ in chatbot.stream(
            Command(resume=resume_value),
            config=pending_config,
            stream_mode="messages",
        ):
            if isinstance(message_chunk, AIMessage):
                yield message_chunk.content

    def handle_hil_response(resume_value):
        with st.chat_message("assistant"):
            ai_message = st.write_stream(resume_stream(resume_value))

        if ai_message:
            st.session_state["message_history"].append(
                {"role": "assistant", "content": ai_message, "sources": None}
            )

        new_state = chatbot.get_state(config=pending_config)
        st.session_state["pending_hil"] = bool(new_state.next)
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve ✅", use_container_width=True):
            handle_hil_response("yes")

    with col2:
        if st.button("Reject ❌", use_container_width=True):
            handle_hil_response("no")

    with st.expander("Or provide a custom response"):
        custom_response = st.text_input("Custom response", key="custom_hil_response")
        if st.button("Submit custom response", key="submit-custom-hil") and custom_response:
            handle_hil_response(custom_response)

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content, "sources": None})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
