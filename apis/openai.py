import dotenv
import os
from typing import Optional

from openai import AsyncOpenAI

dotenv.load_dotenv()

# --- CONFIGURATION ---
MEMORY_DIR = "memory/trans"
os.makedirs(MEMORY_DIR, exist_ok=True)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def openai_websearch(sys_prompt: Optional[str], user_prompt: str, context=None, model="gpt-5-nano", api="responses") -> str:
        """
        Call OpenAI using the Responses API with gpt-5.2, low reasoning, and web search.

        Args:
            prompt: The user's prompt

        Returns:
            The assistant's response text
        """

        system_default_prompt = """
            You're a personal assistant that mimics a human-like conversational style. 
            Answer the user's question in a concise, friendly and human manner.
            Keep it up to 20 words maximum.

            Do not guess or make up information about current events, prices, weather, or recent news.
        """


        messages = [
            {"role": "system", "content": sys_prompt if sys_prompt else system_default_prompt},
            {"role": "user", "content": f"<context>\n{context}\n</context>\n\n{user_prompt}"}
        ]


        response = await openai_client.responses.create(
            model=model,
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


async def openai_simple(sys_prompt: Optional[str], user_prompt: str, context= None, model="gpt-5-nano", api="responses") -> str:

        system_default_prompt = """
            Answer the user's question. 
            If you don't have up-to-date information or are unsure, respond ONLY with:
            'Would you like me to search the web?'

            Do not guess or make up information about current events, prices, weather, or recent news.
        """


        messages = [
            {"role": "system", "content": sys_prompt if sys_prompt else system_default_prompt},
            {"role": "user", "content": f"<context>\n{context}\n</context>\n\n{user_prompt}"}
        ]

        response = await openai_client.responses.create(
            model=model,
            reasoning={"effort": "medium"},
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