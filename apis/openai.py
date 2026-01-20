import dotenv
import os
import re
from typing import Optional, AsyncIterator
from openai import AsyncOpenAI

dotenv.load_dotenv()

# --- CONFIGURATION ---
MEMORY_DIR = "memory/trans"
os.makedirs(MEMORY_DIR, exist_ok=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- PERSONA CONFIGURATION ---
# Centralized persona for consistent personality across all responses
PERSONA = """
You are Eva, a chill and friendly AI assistant. Talk like a real person would - casual, warm, and natural.

Style guide:
- Use contractions (I'm, you're, that's, won't, etc.)
- Throw in casual phrases like "honestly", "yeah", "so basically", "pretty much", "no worries"
- Be warm and supportive - use phrases like "oh nice!", "gotcha", "for sure", "totally"
- Keep responses short and punchy - around 1-2 sentences max
- Sound like you're chatting with a friend, not reading a manual
- Use "gonna", "wanna", "kinda", "gotta" when it feels natural
- React naturally - "oh that's cool!", "hmm let me think", "ah yeah"

Do NOT:
- Sound robotic or overly formal
- Use bullet points or lists
- Say "certainly", "indeed", "I'd be happy to", "absolutely"
- Over-explain or be verbose
"""

WEBSEARCH_PERSONA = PERSONA + """
Do not guess or make up information about current events, prices, weather, or recent news.
"""

SIMPLE_PERSONA = PERSONA + """
If you don't have up-to-date information or are unsure, just say:
'Want me to search the web for that?'

Do not guess or make up information about current events, prices, weather, or recent news.
"""


async def openai_websearch_stream(sys_prompt: Optional[str], user_prompt: str, context=None, model="gpt-5-nano") -> AsyncIterator[str]:
    """
    Call OpenAI using the Responses API with streaming and web search.

    Args:
        user_prompt: The user's prompt
        context: Optional context to include
        model: OpenAI model to use
        sys_prompt: System instructions for the model

    Yields:
        Text chunks as they arrive
    """
    messages = [
        {"role": "system", "content": sys_prompt if sys_prompt else WEBSEARCH_PERSONA},
        {"role": "user", "content": f"<context>\n{context}\n</context>\n\n{user_prompt}"}
    ]

    stream = await openai_client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        tools=[{
            "type": "web_search",
            "search_context_size": "low"
        }],
        input=messages,
        stream=True,
    )

    async for event in stream:
        if event.type == "response.output_text.delta":
            text = event.delta
            # Remove source citations like ([example.com](url))
            text = re.sub(r'\s*\(\[.*?\]\(.*?\)\)', '', text)
            if text:
                yield text


async def openai_simple_stream(sys_prompt: Optional[str], user_prompt: str, context=None, model="gpt-5-nano") -> AsyncIterator[str]:
    """
    Call OpenAI using the Responses API with streaming (no tools).

    Args:
        user_prompt: The user's prompt
        context: Optional context to include
        model: OpenAI model to use
        sys_prompt: System instructions for the model

    Yields:
        Text chunks as they arrive
    """
    messages = [
        {"role": "system", "content": sys_prompt if sys_prompt else SIMPLE_PERSONA},
        {"role": "user", "content": f"<context>\n{context}\n</context>\n\n{user_prompt}"}
    ]

    stream = await openai_client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        input=messages,
        stream=True,
    )

    async for event in stream:
        if event.type == "response.output_text.delta":
            yield event.delta


async def openai_websearch(sys_prompt: Optional[str], user_prompt: str, context=None, model="gpt-5-nano", api="responses") -> str:
        """
        Call OpenAI using the Responses API with web search.

        Args:
            user_prompt: The user's prompt
            context: Optional context to include
            model: OpenAI model to use
            sys_prompt: System instructions for the model
            api: API type to use ("responses" or other)

        Returns:
            The assistant's response text
        """
        messages = [
            {"role": "system", "content": sys_prompt if sys_prompt else WEBSEARCH_PERSONA},
            {"role": "user", "content": f"<context>\n{context}\n</context>\n\n{user_prompt}"}
        ]

        if api == "responses":
            response = await openai_client.responses.create(
                model=model,
                reasoning={"effort": "medium"},
                tools=[{
                    "type": "web_search",
                    "search_context_size": "low"  # Options: "low", "medium", "high"
                }],
                input=messages,
            )
        else:
             pass

        # Extract text content from response
        response_text = ""
        for item in response.output:
            if item.type == "message":
                for content_block in item.content:
                    if content_block.type == "output_text":
                        response_text += content_block.text

        # Remove source citations like ([example.com](url))
        response_text = re.sub(r'\s*\(\[.*?\]\(.*?\)\)', '', response_text)

        return response_text


async def openai_simple(sys_prompt: Optional[str], user_prompt: str, context= None, model="gpt-5-nano", api="responses") -> str:
        """
        Call OpenAI using the Responses API (no tools).

        Args:
            user_prompt: The user's prompt
            context: Optional context to include
            model: OpenAI model to use
            sys_prompt: System instructions for the model
            api: API type to use ("responses" or other)

        Returns:
            The assistant's response text
        """
        messages = [
            {"role": "system", "content": sys_prompt if sys_prompt else SIMPLE_PERSONA},
            {"role": "user", "content": f"<context>\n{context}\n</context>\n\n{user_prompt}"}
        ]


        if api == "responses":
            response = await openai_client.responses.create(
                model=model,
                reasoning={"effort": "medium"},
                input=messages,
            )
        else:
             pass

        # Extract text content from response
        response_text = ""
        for item in response.output:
            if item.type == "message":
                for content_block in item.content:
                    if content_block.type == "output_text":
                        response_text += content_block.text

        return response_text