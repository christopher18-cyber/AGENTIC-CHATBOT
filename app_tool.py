from agentic_chatbot_tool_backend import chatbot, get_all_thread
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import streamlit as st
import uuid

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
        config={
            "configurable":{
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages",[])

st.title("Agentic Chatbot with LangGraph")

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_thread()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

add_thread(st.session_state["thread_id"])

st.sidebar.title("My Conversations")

if st.sidebar.button("New chats"):

    reset_chat()

    st.rerun()

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):
        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:
            if isinstance(message,HumanMessage):
                role = "user"

            elif isinstance(message,AIMessage):
                role = "assistant"
            else:
                continue

            temp_messages.append({
                "role" : role,
                "content": message.content
            })

        st.session_state["message_history"] = temp_messages
        st.rerun()


for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state["message_history"].append({
        "role":"user",
        "content":user_input
    })

    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable":{
            "thread_id": st.session_state["thread_id"]
        },
        "metadata":{
            "thread_id":st.session_state["thread_id"]
        },
        "run_name":"chat_trace"
    }

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages":[
                        HumanMessage(content=user_input)
                    ]
                }, config=CONFIG,
                stream_mode="messages"
            )
            if isinstance(message_chunk, AIMessage)
        )

    st.session_state["message_history"].append({
        "role":"assistant",
        "content":ai_message
    })