"""
GroqFlow Chat - a production-structured Streamlit chat UI backed by
Groq's LPU inference and orchestrated with LangChain.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import streamlit as st

from src.chatbot import ChatbotError, clear_session_history, stream_response
from src.config import AVAILABLE_MODELS, DEFAULT_SYSTEM_PROMPT, LANGSMITH_TRACING_ENABLED, AppSettings
from src.logging_utils import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="GroqFlow Chat", page_icon="💬", layout="centered")

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role": "user"|"assistant", "content": str}] — for display only

if "session_id" not in st.session_state:
    # LangChain's RunnableWithMessageHistory keys conversation memory off this id,
    # so each browser tab gets its own independent chat history.
    st.session_state.session_id = str(uuid.uuid4())

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.get("groq_api_key", ""),
        help="Get a free key at console.groq.com. You can also set GROQ_API_KEY as an env var.",
    )
    st.session_state["groq_api_key"] = groq_api_key

    model = st.selectbox("Model", AVAILABLE_MODELS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.05)
    max_tokens = st.slider("Max response tokens", 128, 4096, 1024, step=128)

    system_prompt = st.text_area(
        "System prompt",
        value=DEFAULT_SYSTEM_PROMPT,
        height=100,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            clear_session_history(st.session_state.session_id)
            st.rerun()
    with col2:
        chat_export = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            "⬇️ Export",
            data=chat_export,
            file_name=f"chat_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json",
            use_container_width=True,
            disabled=not st.session_state.messages,
        )

    st.caption("Built with Streamlit, LangChain, and Groq.")
    if LANGSMITH_TRACING_ENABLED:
        st.caption("🔍 LangSmith tracing is on for this session.")

# ---------------- HEADER ----------------
st.title("💬 GroqFlow Chat")
st.caption("A fast, memory-aware AI chatbot powered by Groq + LangChain")

# ---------------- CHAT HISTORY ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------- CHAT INPUT ----------------
user_prompt = st.chat_input("Type your message...")

if user_prompt:
    settings = AppSettings(
        groq_api_key=groq_api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )

    is_valid, error_message = settings.is_valid()
    if not is_valid:
        st.error(error_message)
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            for chunk in stream_response(settings, user_prompt, st.session_state.session_id):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except ChatbotError as exc:
            logger.warning("Chat error surfaced to user: %s", exc)
            placeholder.empty()
            st.error(str(exc))
            full_response = ""

    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})


