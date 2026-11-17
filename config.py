"""Model names and API clients. Keys and endpoints come from environment variables.

Embeddings always use DashScope (text-embedding-v4). The chat model defaults to
qwen-plus; set CHAT_PROVIDER=deepseek to switch. Both expose an OpenAI-compatible
API, so switching only changes base_url, key and model name.
"""
import os

from openai import OpenAI

EMBED_MODEL = "text-embedding-v4"

DASHSCOPE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
embed_client = OpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"), base_url=DASHSCOPE_URL)

CHAT_PROVIDER = os.getenv("CHAT_PROVIDER", "dashscope")
if CHAT_PROVIDER == "deepseek":
    chat_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-chat")
else:
    chat_client = embed_client
    CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-plus")
