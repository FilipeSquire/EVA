import os
import json
from datetime import datetime
from typing import Literal, Annotated
from typing_extensions import TypedDict

from openai import OpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import dotenv 
dotenv.load_dotenv()

# --- CONFIGURATION ---
MEMORY_DIR = "memory/trans"
os.makedirs(MEMORY_DIR, exist_ok=True)

# OpenAI client for Responses API

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- STATE DEFINITION ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# --- OPENAI RESPONSES API WITH WEB SEARCH ---
def call_openai_with_web_search(messages: list) -> str:
    """
    Call OpenAI using the Responses API with gpt-5.2, low reasoning, and web search.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        The assistant's response text
    """
    response = openai_client.responses.create(
        model="gpt-5-nano",
        reasoning={"effort": "medium"},
        tools=[{
            "type": "web_search",
            "search_context_size": "low"  # Options: "low", "medium", "high"
        }],
        input=messages,
    )

    # Extract text content from response
    response_text = ""
    for item in response.output:
        if item.type == "message":
            for content_block in item.content:
                if content_block.type == "output_text":
                    response_text += content_block.text

    return response_text


# --- MESSAGE CONVERSION ---
def convert_to_openai_messages(messages: list) -> list:
    """Convert internal messages to OpenAI format."""
    openai_messages = []
    for msg in messages:
        if hasattr(msg, 'type'):
            # LangGraph message object
            if msg.type == "human":
                openai_messages.append({"role": "user", "content": msg.content})
            elif msg.type == "ai":
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif msg.type == "system":
                openai_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, dict):
            # Already in dict format
            openai_messages.append(msg)
    return openai_messages


# --- GRAPH NODES ---
def call_model(state: AgentState):
    """Call OpenAI with web search capability."""
    openai_messages = convert_to_openai_messages(state["messages"])
    response_text = call_openai_with_web_search(openai_messages)

    return {"messages": [{"role": "assistant", "content": response_text}]}


# --- BUILD GRAPH ---
def build_agent_graph():
    """Build the LangGraph agent."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", call_model)

    # Add edges
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)

    return graph.compile()


# Create the agent
agent = build_agent_graph()


# --- MEMORY LOGGING ---
def save_conversation(prompt: str, response: str):
    """Save conversation to memory/trans as JSON."""
    timestamp = datetime.now().isoformat()
    conversation_data = {
        "timestamp": timestamp,
        "model": "gpt-5-nano",
        "reasoning": "low",
        "tools": ["web_search"],
        "conversation": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    }

    filename = f"conversation_{timestamp.replace(':', '-').replace('.', '-')}.txt"
    filepath = os.path.join(MEMORY_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(conversation_data, f, indent=4)

    return filepath


# --- MAIN FUNCTION ---
def ask_ai(prompt: str, system_prompt: str = None) -> str:
    """
    Send a prompt to the AI and get a response.

    Args:
        prompt: The user's question or command
        system_prompt: Optional system prompt to set agent behavior

    Returns:
        The AI's response as a string
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    # Run the agent
    result = agent.invoke({"messages": messages})

    # Extract the final response
    final_message = result["messages"][-1]
    response_text = final_message.content if hasattr(final_message, 'content') else final_message["content"]

    # Save to memory
    filepath = save_conversation(prompt, response_text)
    print(f"Conversation saved to: {filepath}")

    return response_text


# --- CLI FOR TESTING ---
if __name__ == "__main__":
    print("=== EVA Brain Test ===")
    print("Using OpenAI gpt-5.2 with web search")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quit":
            break

        try:
            response = ask_ai(user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"\nError: {e}")
