from langgraph.graph import START,END,StateGraph
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import sqlite3


load_dotenv()

llm = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=512)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages":[response]}

conn = sqlite3.connect(database="chatbot.db",check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

workflow = graph.compile(checkpointer=checkpoint)

for ckpt in checkpoint.list(None):
    print(ckpt.config)


# def get_all_threads():
#     all_threads = set()
#     for ckpt in checkpoint.list(None):
#         print(ckpt.config)
#     return list(all_threads)

def get_all_threads():
    all_threads = []
    # checkpoint.list(None) iterates through all saved checkpoints across all threads
    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config.get("configurable", {}).get("thread_id")
        if thread_id and thread_id not in all_threads:
            all_threads.append(thread_id)
    return all_threads