from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver
from typing import TypedDict
import time

class CrashState(TypedDict):
    input: str
    step1: str
    step2: str

def step_1(state:CrashState) -> CrashState:
    print("✅ Step 1 executed.")
    return {"step1": "done","input":state["input"]}

def step_2(state: CrashState) -> CrashState:
    print("⌛⌛ Step 2 hanging .. Now manually interrupt from the notebook.")
    time.sleep(30)
    return {"step2":"done"}

def step_3(state: CrashState) -> CrashState:
    print("✅ step 3 executed")
    return {"done":True}

graph = StateGraph(CrashState)
graph.add_node("step_1",step_1)
graph.add_node("step_2",step_2)
graph.add_node("step_3", step_3)

graph.set_entry_point("step_1")
graph.add_edge("step_1","step_2")
graph.add_edge("step_2","step_3")
graph.add_edge("step_3",END)


checkpoint = InMemorySaver()

workflow = graph.compile(checkpointer=checkpoint)


try:
    print("Running graph: Please manually interrupt during step 2...")
    workflow.invoke({"input":"start"}, config={"configurable":{"thread_id":"thread-1"}})
except KeyboardInterrupt:
    print("Kernel manually interrupted (Crash simulated).")

workflow.get_state({"configurable":{"thread_id":"thread-1"}})

final_state = workflow.invoke(None,config={"configurable":{"thread_id":"thread-1"}})
print("Final state",final_state)

workflow.get_state({"configurable":{"thread_id":"thread-1"}})