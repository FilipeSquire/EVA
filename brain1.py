import os
import json
import asyncio
from datetime import datetime
from typing import Literal, Optional, Annotated, AsyncIterator
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

import dotenv
dotenv.load_dotenv()

# Import provider functions
from apis.openai import openai_simple, openai_websearch, openai_simple_stream, openai_websearch_stream
from apis.gemini import gemini_simple, gemini_websearch, gemini_simple_stream, gemini_websearch_stream

# --- CONFIGURATION ---
MEMORY_DIR = "memory/trans"
os.makedirs(MEMORY_DIR, exist_ok=True)

WEBSEARCH_TRIGGER = "would you like me to search the web?"


# --- STATE DEFINITION ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    provider: str  # "openai" or "gemini"
    use_websearch: bool
    user_prompt: str
    system_prompt: Optional[str]
    context: Optional[str]
    response: str
    needs_websearch: bool  # Flag set when model asks to search


# --- PROVIDER ROUTER ---
async def call_simple(state: AgentState) -> dict:
    """Call the simple (no tools) version of the selected provider."""
    provider = state.get("provider", "openai")

    if provider == "openai":
        response = await openai_simple(
            sys_prompt=state.get("system_prompt"),
            user_prompt=state["user_prompt"],
            context=state.get("context")
        )
    elif provider == "gemini":
        response = await gemini_simple(
            sys_prompt=state.get("system_prompt"),
            user_prompt=state["user_prompt"],
            context=state.get("context")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Check if model wants to search the web
    needs_websearch = WEBSEARCH_TRIGGER in response.lower()

    return {
        "response": response,
        "needs_websearch": needs_websearch,
        "messages": [{"role": "assistant", "content": response}]
    }


async def call_websearch(state: AgentState) -> dict:
    """Call the websearch version of the selected provider."""
    provider = state.get("provider", "openai")

    if provider == "openai":
        response = await openai_websearch(
            sys_prompt=state.get("system_prompt"),
            user_prompt=state["user_prompt"],
            context=state.get("context")
        )
    elif provider == "gemini":
        response = await gemini_websearch(
            sys_prompt=state.get("system_prompt"),
            user_prompt=state["user_prompt"],
            context=state.get("context")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return {
        "response": response,
        "needs_websearch": False,
        "messages": [{"role": "assistant", "content": response}]
    }


# --- ROUTING LOGIC ---
def route_initial(state: AgentState) -> Literal["simple", "websearch"]:
    """Route based on whether websearch was explicitly requested."""
    if state.get("use_websearch", False):
        return "websearch"
    return "simple"


# --- BUILD GRAPH ---
def build_agent_graph():
    """
    Build the LangGraph agent with routing:

    START → route_initial → simple → END
                         ↘ websearch → END

    Note: Websearch confirmation is handled outside the graph.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("simple", call_simple)
    graph.add_node("websearch", call_websearch)

    # Add edges with routing
    graph.add_conditional_edges(START, route_initial, {
        "simple": "simple",
        "websearch": "websearch"
    })

    graph.add_edge("simple", END)
    graph.add_edge("websearch", END)

    return graph.compile()


# Create the agent
agent = build_agent_graph()


# --- MEMORY LOGGING ---
def save_conversation(prompt: str, response: str, provider: str, used_websearch: bool):
    """Save a single conversation to memory/trans as JSON."""
    timestamp = datetime.now().isoformat()
    conversation_data = {
        "timestamp": timestamp,
        "provider": provider,
        "used_websearch": used_websearch,
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


def save_session(conversations: list, session_start: str):
    """
    Save an entire session of conversations to memory/trans as JSON.

    Args:
        conversations: List of dicts with keys: prompt, response, provider, used_websearch, timestamp
        session_start: ISO timestamp of when the session started
    """
    if not conversations:
        return None

    session_end = datetime.now().isoformat()
    session_data = {
        "session_start": session_start,
        "session_end": session_end,
        "total_interactions": len(conversations),
        "conversations": conversations
    }

    filename = f"session_{session_start.replace(':', '-').replace('.', '-')}.json"
    filepath = os.path.join(MEMORY_DIR, filename)

    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=4)

    print(f"📁 Session saved to: {filepath}")
    return filepath


# --- RESPONSE CLASS ---
class AIResponse:
    """Response object that includes metadata about the response."""
    def __init__(self, text: str, needs_websearch: bool, provider: str, prompt: str):
        self.text = text
        self.needs_websearch = needs_websearch
        self.provider = provider
        self.prompt = prompt

    def __str__(self):
        return self.text


# --- MAIN FUNCTION ---
async def ask_ai(
    prompt: str,
    provider: str = "openai",
    system_prompt: Optional[str] = None,
    context: Optional[str] = None,
    use_websearch: bool = False
) -> AIResponse:
    """
    Send a prompt to the AI and get a response.

    Args:
        prompt: The user's question or command
        provider: Which AI provider to use ('openai' or 'gemini')
        system_prompt: Optional system prompt to set agent behavior
        context: Optional context to include with the prompt
        use_websearch: Force websearch (skip simple call)

    Returns:
        AIResponse object with text, needs_websearch flag, and metadata
    """
    # Build initial state
    initial_state = {
        "messages": [{"role": "user", "content": prompt}],
        "provider": provider,
        "use_websearch": use_websearch,
        "user_prompt": prompt,
        "system_prompt": system_prompt,
        "context": context,
        "response": "",
        "needs_websearch": False
    }

    # Run the agent
    result = await agent.ainvoke(initial_state)

    response_text = result["response"]
    needs_websearch = result.get("needs_websearch", False)

    # Save to memory
    filepath = save_conversation(prompt, response_text, provider, use_websearch)
    print(f"Conversation saved to: {filepath}")

    return AIResponse(
        text=response_text,
        needs_websearch=needs_websearch,
        provider=provider,
        prompt=prompt
    )


async def ask_ai_stream(
    prompt: str,
    provider: str = "openai",
    system_prompt: Optional[str] = None,
    context: Optional[str] = None,
    use_websearch: bool = False
) -> AsyncIterator[str]:
    """
    Send a prompt to the AI and stream the response.

    Args:
        prompt: The user's question or command
        provider: Which AI provider to use ('openai' or 'gemini')
        system_prompt: Optional system prompt to set agent behavior
        context: Optional context to include with the prompt
        use_websearch: Force websearch (skip simple call)

    Yields:
        Text chunks as they arrive from the model
    """
    if provider == "openai":
        if use_websearch:
            stream = openai_websearch_stream(
                sys_prompt=system_prompt,
                user_prompt=prompt,
                context=context
            )
        else:
            stream = openai_simple_stream(
                sys_prompt=system_prompt,
                user_prompt=prompt,
                context=context
            )
    elif provider == "gemini":
        if use_websearch:
            stream = gemini_websearch_stream(
                sys_prompt=system_prompt,
                user_prompt=prompt,
                context=context
            )
        else:
            stream = gemini_simple_stream(
                sys_prompt=system_prompt,
                user_prompt=prompt,
                context=context
            )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    async for chunk in stream:
        yield chunk


async def ask_ai_stream_with_metadata(
    prompt: str,
    provider: str = "openai",
    system_prompt: Optional[str] = None,
    context: Optional[str] = None,
    use_websearch: bool = False
):
    """
    Stream response and return metadata including needs_websearch flag.

    Args:
        prompt: The user's question or command
        provider: Which AI provider to use ('openai' or 'gemini')
        system_prompt: Optional system prompt to set agent behavior
        context: Optional context to include with the prompt
        use_websearch: Force websearch (skip simple call)

    Returns:
        A tuple of (async_generator, callback_to_get_metadata)
        Call the callback after consuming the generator to get AIResponse metadata.
    """
    full_response = []

    async def stream_generator():
        if provider == "openai":
            if use_websearch:
                stream = openai_websearch_stream(
                    sys_prompt=system_prompt,
                    user_prompt=prompt,
                    context=context
                )
            else:
                stream = openai_simple_stream(
                    sys_prompt=system_prompt,
                    user_prompt=prompt,
                    context=context
                )
        elif provider == "gemini":
            if use_websearch:
                stream = gemini_websearch_stream(
                    sys_prompt=system_prompt,
                    user_prompt=prompt,
                    context=context
                )
            else:
                stream = gemini_simple_stream(
                    sys_prompt=system_prompt,
                    user_prompt=prompt,
                    context=context
                )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        async for chunk in stream:
            full_response.append(chunk)
            yield chunk

    def get_response():
        response_text = "".join(full_response)
        needs_websearch = WEBSEARCH_TRIGGER in response_text.lower()

        # Note: Saving is now handled at session end, not per-interaction
        return AIResponse(
            text=response_text,
            needs_websearch=needs_websearch,
            provider=provider,
            prompt=prompt
        )

    return stream_generator(), get_response


async def confirm_websearch(response: AIResponse, system_prompt: Optional[str] = None, context: Optional[str] = None) -> AIResponse:
    """
    Confirm and execute web search for a response that requested it.

    Args:
        response: The AIResponse that requested web search
        system_prompt: Optional system prompt override
        context: Optional context override

    Returns:
        New AIResponse with web search results
    """
    return await ask_ai(
        prompt=response.prompt,
        provider=response.provider,
        system_prompt=system_prompt,
        context=context,
        use_websearch=True
    )


async def confirm_websearch_stream(response: AIResponse, system_prompt: Optional[str] = None, context: Optional[str] = None) -> AsyncIterator[str]:
    """
    Confirm and execute web search with streaming for a response that requested it.

    Args:
        response: The AIResponse that requested web search
        system_prompt: Optional system prompt override
        context: Optional context override

    Yields:
        Text chunks as they arrive
    """
    async for chunk in ask_ai_stream(
        prompt=response.prompt,
        provider=response.provider,
        system_prompt=system_prompt,
        context=context,
        use_websearch=True
    ):
        yield chunk


# --- CLI FOR TESTING ---
async def main():
    print("=== EVA Brain Test ===")
    print("Providers: openai, gemini")
    print("Commands: 'quit', 'provider <name>'")
    print()

    provider = "openai"

    while True:
        user_input = input(f"\n[{provider}] You: ").strip()

        if user_input.lower() == "quit":
            break
        elif user_input.lower().startswith("provider "):
            provider = user_input.split(" ", 1)[1].strip()
            print(f"Switched to: {provider}")
            continue

        try:
            response = await ask_ai(prompt=user_input, provider=provider)
            print(f"\nAssistant: {response}")

            # Check if web search was requested
            if response.needs_websearch:
                confirm = input("\nConfirm web search? (y/n): ").strip().lower()
                if confirm == "y" or confirm == "yes":
                    print("Searching the web...")
                    response = await confirm_websearch(response)
                    print(f"\nAssistant: {response}")

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
