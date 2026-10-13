from langgraph.graph import START,END,StateGraph
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=512)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages":[response]}

checkpoint = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

workflow = graph.compile(checkpointer=checkpoint)
initial_state = {
    "messages":[HumanMessage(content="What is object oriented programming?")]
}

# response = workflow.invoke(initial_state)
# print(response["messages"][-1].content)

if __name__ == "__main__":

    thread_id = "1"

    while True:
        user_message = input("Type here: ")
        print("User:",user_message)
        if user_message.strip().lower() in ["exit","quit","bye"]:
            break
        config = {"configurable":{"thread_id":thread_id}}
        response = workflow.invoke({"messages":[HumanMessage(content=user_message)]},config=config)
        print("AI: ", response["messages"][-1].content)