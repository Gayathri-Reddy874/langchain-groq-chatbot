"""
Core conversational chain: builds a Groq-backed LangChain runnable wrapped
in LangChain's native RunnableWithMessageHistory for per-session memory,
streams tokens back to the caller, and retries transient failures instead
of surfacing a raw stack trace to the user.
"""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import AppSettings, MAX_HISTORY_MESSAGES, MAX_RETRIES, REQUEST_TIMEOUT_SECONDS
from .logging_utils import get_logger

logger = get_logger(__name__)


class ChatbotError(Exception):
    """Raised when the chatbot cannot produce a response after retries."""


class TrimmedChatMessageHistory(InMemoryChatMessageHistory):
    """
    In-memory LangChain chat history that keeps only the most recent
    MAX_HISTORY_MESSAGES turns, so long conversations don't grow the
    prompt (and the Groq bill) without bound.
    """

    def add_message(self, message) -> None:  # noqa: ANN001 - matches base signature
        super().add_message(message)
        if len(self.messages) > MAX_HISTORY_MESSAGES:
            self.messages = self.messages[-MAX_HISTORY_MESSAGES:]


# One history object per Streamlit session_id. Streamlit re-runs this module
# on every interaction but keeps the process alive, so a module-level dict
# is sufficient to persist memory for the lifetime of the app process.
_SESSION_HISTORY_STORE: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """LangChain-required factory: return (or create) history for a session_id."""
    if session_id not in _SESSION_HISTORY_STORE:
        _SESSION_HISTORY_STORE[session_id] = TrimmedChatMessageHistory()
    return _SESSION_HISTORY_STORE[session_id]


def clear_session_history(session_id: str) -> None:
    """Drop stored memory for a session, e.g. when the user clicks 'Clear Chat'."""
    _SESSION_HISTORY_STORE.pop(session_id, None)


def build_chain(settings: AppSettings) -> RunnableWithMessageHistory:
    """
    Construct prompt | llm | parser, wrapped in RunnableWithMessageHistory so
    LangChain itself manages reading and appending conversation turns rather
    than the caller manually converting and re-passing a message list.
    """
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{question}"),
        ]
    )

    base_chain = prompt | llm | StrOutputParser()

    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(MAX_RETRIES + 1),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type(Exception),
)
def _invoke_with_retry(chain, payload: dict, config: dict) -> str:
    return chain.invoke(payload, config=config)


def get_response(settings: AppSettings, question: str, session_id: str) -> str:
    """Return a single, fully-formed response (non-streaming)."""
    chain = build_chain(settings)
    payload = {"system_prompt": settings.system_prompt, "question": question}
    config = {"configurable": {"session_id": session_id}}
    try:
        return _invoke_with_retry(chain, payload, config)
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly ChatbotError
        logger.exception("Chat completion failed")
        raise ChatbotError(_friendly_error_message(exc)) from exc


def stream_response(settings: AppSettings, question: str, session_id: str) -> Iterator[str]:
    """Yield response chunks as they arrive, for a typing-style UI."""
    chain = build_chain(settings)
    payload = {"system_prompt": settings.system_prompt, "question": question}
    config = {"configurable": {"session_id": session_id}}
    try:
        yield from chain.stream(payload, config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Streaming chat completion failed")
        raise ChatbotError(_friendly_error_message(exc)) from exc


def _friendly_error_message(exc: Exception) -> str:
    text = str(exc).lower()
    if "401" in text or "invalid api key" in text or "authentication" in text:
        return "Your Groq API key was rejected. Double-check it in the sidebar."
    if "429" in text or "rate limit" in text:
        return "Groq is rate-limiting this key right now. Wait a moment and try again."
    if "timeout" in text:
        return "The request timed out. Please try again."
    return "Something went wrong while contacting Groq. Please try again in a moment."

