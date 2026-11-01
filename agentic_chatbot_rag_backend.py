import math
import os
import sqlite3
from typing import Annotated, TypedDict

import requests
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", max_tokens=512)

# Change this (or GoogleEmbeddings):
# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embeddings-001")

# To your working local HuggingFace embeddings setup:
@st.cache_resource
def get_cached_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_cached_embeddings()

def ingest_raw(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)

def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document

    use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents

    Args:
        query: The question or search query used to retrieve the PDF document
    """
    retriever = get_retriever()

    documents = retriever.invoke(query)
    if not documents:
        return "No relevant information found in this PDF"
    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)

@tool
def calculator(expression: str) -> str:
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

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> str:
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
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    if not api_key:
        return {"error": "WEATHERSTACK_API_KEY environment variable is not set."}

    url = "https://api.weatherstack.com/current"

    params = {
        "access_key": api_key,
        "query": location,
        "units": "m"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "success" in data and data["success"] is False:
            error_info = data.get("error", {})
            return {"error": error_info.get("info", "Unknown Weatherstack API error")}

        current = data.get("current", {})
        loc = data.get("location", {})

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

search_tool = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="advanced"
)

tools = [search_tool, calculator, get_stock_price, get_current_weather, rag_tool]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    """LLM node that can answer directly or call an appropriate tool."""
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document. "
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information from the uploaded document. "
            "If the user asks about a PDF but no document is available, ask them to upload a PDF. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

tool_node = ToolNode(tools)

conn = sqlite3.connect(database="rag.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    all_threads = []
    for ckpt in checkpoint.list(None):
        thread_id = ckpt.config.get("configurable", {}).get("thread_id")
        if thread_id and thread_id not in all_threads:
            all_threads.append(thread_id)
    return all_threads