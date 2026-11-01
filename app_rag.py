import tempfile
import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agentic_chatbot_rag_backend import chatbot, get_all_threads, ingest_raw

def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    add_thread(st.session_state["thread_id"])

def load_conversation(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])

st.title("Agentic Chatbot with LangGraph")

# Initialize Session State
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

add_thread(st.session_state["thread_id"])

# Sidebar for Conversation History
st.sidebar.title("My Conversations")

if st.sidebar.button("New chats"):
    reset_chat()
    st.rerun()

st.sidebar.markdown("---")

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id), key=thread_id):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            temp_messages.append({
                "role": role,
                "content": message.content
            })

        st.session_state["message_history"] = temp_messages
        st.rerun()

# Display Chat History
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# --- INLINE UPLOAD SECTION (Near Type Bar) ---
uploaded_file = st.file_uploader("Attach a PDF document to query", type=["pdf"], label_visibility="collapsed")

if uploaded_file is not None:
    # Prevent re-processing the exact same file upload on every rerun
    if st.session_state.get("last_uploaded_file") != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        with st.spinner(f"Indexing `{uploaded_file.name}`..."):
            try:
                ingest_raw(tmp_file_path)
                st.success(f"Indexed `{uploaded_file.name}` successfully!")
                st.session_state["last_uploaded_file"] = uploaded_file.name
            except Exception as e:
                st.error(f"Error indexing PDF: {e}")
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

# User Chat Input
user_input = st.chat_input("Type here or ask about your PDF...")

if user_input:
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace"
    }

    with st.chat_message("assistant"):
        def response_generator():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                # Filter strictly for clean text from AIMessage chunks
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(response_generator())

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })