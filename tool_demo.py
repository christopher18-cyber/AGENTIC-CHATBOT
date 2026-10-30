from langgraph.graph import START,END,StateGraph
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import os
import requests
import math

load_dotenv()

llm = ChatGroq(model="qwen/qwen3.8-27b", max_tokens=512)

search_tool = TavilySearch(
    max_results = 2,
    topic = "general",
    search_depth = "advanced"
)

@tool
def calculator(expression: str)->str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__":{}},allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def get_stock_price(symbol:str)-> str:
    """
    Fetch latest stock price for a given symbol (e.g 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=DFAY4184HQ44GJOB"
    r = requests.get(url)
    return r.json()





def get_current_weather(location: str) -> dict:
    """
    Fetch real-time weather data for a given location using the Weatherstack API.

    Args:
        location (str): The city name, zip code, latitude/longitude, or IP address.

    Returns:
        dict: A dictionary containing key weather details or an error message.
    """
    # Retrieve the API key securely from environment variables
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return {"error": "WEATHERSTACK_API_KEY environment variable is not set."}

    # Weatherstack current endpoint
    # Note: Free tier accounts typically require HTTP; paid plans support HTTPS.
    url = "https://api.weatherstack.com/current"

    params = {
        "access_key": api_key,
        "query": location,
        "units": "m"  # 'm' for Metric (°C, km/h), 'f' for Fahrenheit (°F, mph), 's' for Scientific
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Handle Weatherstack API-level errors (e.g., invalid key or location)
        if "success" in data and data["success"] is False:
            error_info = data.get("error", {})
            return {"error": error_info.get("info", "Unknown Weatherstack API error")}

        current = data.get("current", {})
        loc = data.get("location", {})

        # Return a structured, clean dictionary for your agent to process
        return {
            "location": f"{loc.get('name')}, {loc.get('country')}",
            "temperature_c": current.get("temperature"),
            "condition": current.get("weather_descriptions", [""])[0],
            "humidity": current.get("humidity"),
            "wind_speed_kmh": current.get("wind_speed"),
            "feels_like_c": current.get("feelslike")
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"HTTP Request failed: {e}"}

tools = [search_tool,calculator,get_current_weather,get_stock_price]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

def chat_node(state:ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages":[response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"chat_node")
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge("tools",chat_node)

chatbot = graph.compile()