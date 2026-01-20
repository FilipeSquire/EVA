import dotenv
import os
from typing import Optional, AsyncIterator

from google import genai
from google.genai import types

dotenv.load_dotenv()

# --- CONFIGURATION ---
MEMORY_DIR = "memory/trans"
os.makedirs(MEMORY_DIR, exist_ok=True)

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


async def gemini_websearch_stream(sys_prompt: Optional[str], user_prompt: str, context=None, model="gemini-2.0-flash") -> AsyncIterator[str]:
    """
    Call Gemini with Google Search grounding and streaming enabled.

    Args:
        sys_prompt: System instructions for the model
        user_prompt: The user's prompt
        context: Optional context to include
        model: Gemini model to use

    Yields:
        Text chunks as they arrive
    """
    user_content = f"<context>\n{context}\n</context>\n\n{user_prompt}" if context else user_prompt
    system_instruction = sys_prompt or "Return your answer in a concise and human manner, limited to 20 words maximum."

    async for chunk in gemini_client.aio.models.generate_content_stream(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    ):
        if chunk.text:
            yield chunk.text


async def gemini_simple_stream(sys_prompt: Optional[str], user_prompt: str, context=None, model="gemini-2.0-flash") -> AsyncIterator[str]:
    """
    Call Gemini without any tools, with streaming enabled.

    Args:
        sys_prompt: System instructions for the model
        user_prompt: The user's prompt
        context: Optional context to include
        model: Gemini model to use

    Yields:
        Text chunks as they arrive
    """
    user_content = f"<context>\n{context}\n</context>\n\n{user_prompt}" if context else user_prompt
    system_instruction = sys_prompt or "Return your answer in a concise and human manner, limited to 20 words maximum."

    async for chunk in gemini_client.aio.models.generate_content_stream(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    ):
        if chunk.text:
            yield chunk.text


async def gemini_websearch(sys_prompt: str, user_prompt: str, context=None, model="gemini-2.0-flash") -> str:
    """
    Call Gemini with Google Search grounding enabled.

    Args:
        sys_prompt: System instructions for the model
        user_prompt: The user's prompt
        context: Optional context to include
        model: Gemini model to use

    Returns:
        The assistant's response text
    """
    user_content = f"<context>\n{context}\n</context>\n\n{user_prompt}" if context else user_prompt
    system_instruction = sys_prompt or "Return your answer in a concise and human manner, limited to 20 words maximum."

    response = await gemini_client.aio.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )

    return response.text


async def gemini_simple(sys_prompt: str, user_prompt: str, context=None, model="gemini-2.0-flash") -> str:
    """
    Call Gemini without any tools.

    Args:
        sys_prompt: System instructions for the model
        user_prompt: The user's prompt
        context: Optional context to include
        model: Gemini model to use

    Returns:
        The assistant's response text
    """
    user_content = f"<context>\n{context}\n</context>\n\n{user_prompt}" if context else user_prompt
    system_instruction = sys_prompt or "Return your answer in a concise and human manner, limited to 20 words maximum."

    response = await gemini_client.aio.models.generate_content(
        model=model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    )

    return response.text
