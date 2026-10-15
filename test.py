from agentic_chatbot_backend import workflow
from langchain_core.messages import HumanMessage, BaseMessage

CONFIG = {"configurable":{"thread_id":"thread-1"}}
for message_chunk, metadata in  workflow.stream(
    {"messages":[HumanMessage(content="Generate a blog on python for me")]},
    config=CONFIG,
    stream_mode="messages"):
    if message_chunk.content:
        print(message_chunk.content,end=" ",flush=True)
