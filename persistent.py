from langgraph.graph import START,END,StateGraph
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=512)

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str


def generate_joke(state: JokeState):
    prompt = f"generate a joke on the topic {state["topic"]}"
    response = llm.invoke(prompt).content

    return {"joke":response}

def generate_explanation(state: JokeState):
    prompt = f"write an explanation for the joke - {state["joke"]}"
    response = llm.invoke(prompt).content

    return {"explanation":response}

graph = StateGraph(JokeState)

graph.add_node("generate_joke",generate_joke)
graph.add_node("generate_explanation", generate_explanation)

graph.add_edge(START,"generate_joke")
graph.add_edge("generate_joke","generate_explanation")
graph.add_edge("generate_explanation",END)

checkpointer = InMemorySaver()

workflow = graph.compile(checkpointer=checkpointer)

config1={"configurable":{"thread_id":"1"}}
workflow.invoke({"topic":"football"}, config=config1)
print(workflow.get_state(config1))

list(workflow.get_state_history(config1))

config2 = {"configurable":{"thread_id":"2"}}
workflow.invoke({"topic":"cricket"}, config=config2)