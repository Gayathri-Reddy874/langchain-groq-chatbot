"""
Centralized configuration for GroqFlow Chat.

Reads defaults from environment variables (via .env when present) so the
app works both for local development and containerized deployment, while
still allowing the Streamlit sidebar to override values at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dotenv import load_dotenv
import os

load_dotenv()

# LangChain reads these directly from the environment to enable LangSmith
# tracing (run-by-run visibility into every prompt, chain step, and token
# usage). Loading .env above is enough to activate it — no code path in
# this app needs to reference these values directly.
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=ls__...
#   LANGCHAIN_PROJECT=langchain-groq-chatbot
LANGSMITH_TRACING_ENABLED: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

AVAILABLE_MODELS: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise, and friendly AI assistant."

MAX_HISTORY_MESSAGES = 20  # messages kept in context window sent to the model
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 2


@dataclass
class AppSettings:
    """Runtime-configurable settings for a single chat session."""

    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = AVAILABLE_MODELS[0]
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def is_valid(self) -> tuple[bool, str]:
        """Basic sanity checks before the app tries to call the API."""
        if not self.groq_api_key or not self.groq_api_key.strip():
            return False, "A Groq API key is required. Set GROQ_API_KEY or enter it in the sidebar."
        if self.model not in AVAILABLE_MODELS:
            return False, f"Unknown model '{self.model}'."
        if not (0.0 <= self.temperature <= 1.0):
            return False, "Temperature must be between 0.0 and 1.0."
        return True, ""

