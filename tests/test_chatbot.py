from langchain_core.messages import AIMessage, HumanMessage

from src.chatbot import (
    TrimmedChatMessageHistory,
    _friendly_error_message,
    clear_session_history,
    get_session_history,
)
from src.config import MAX_HISTORY_MESSAGES


def test_get_session_history_creates_and_reuses_same_object():
    history_a = get_session_history("session-1")
    history_b = get_session_history("session-1")
    assert history_a is history_b


def test_get_session_history_is_isolated_per_session():
    history_a = get_session_history("session-a")
    history_b = get_session_history("session-b")
    history_a.add_message(HumanMessage(content="hello from a"))
    assert history_b.messages == []


def test_trimmed_history_caps_at_max_history_messages():
    history = TrimmedChatMessageHistory()
    for i in range(MAX_HISTORY_MESSAGES + 10):
        history.add_message(HumanMessage(content=str(i)) if i % 2 == 0 else AIMessage(content=str(i)))
    assert len(history.messages) == MAX_HISTORY_MESSAGES
    # oldest messages should have been dropped, newest retained
    assert history.messages[-1].content == str(MAX_HISTORY_MESSAGES + 9)


def test_clear_session_history_removes_stored_messages():
    history = get_session_history("session-clear")
    history.add_message(HumanMessage(content="hi"))
    clear_session_history("session-clear")
    fresh = get_session_history("session-clear")
    assert fresh.messages == []


def test_friendly_error_message_for_auth_failure():
    msg = _friendly_error_message(Exception("401 invalid api key"))
    assert "rejected" in msg.lower()


def test_friendly_error_message_for_rate_limit():
    msg = _friendly_error_message(Exception("429 rate limit exceeded"))
    assert "rate-limit" in msg.lower() or "rate limit" in msg.lower()


def test_friendly_error_message_fallback():
    msg = _friendly_error_message(Exception("something obscure broke"))
    assert "went wrong" in msg.lower()

