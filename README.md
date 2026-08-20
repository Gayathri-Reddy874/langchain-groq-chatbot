# 💬 LangChain Groq Chatbot

A production-structured, streaming AI chatbot built with **LangChain** and **Groq's LPU inference engine**, with **Streamlit** as the UI layer. Unlike a typical single-file demo, this project is organized like a real application: modular source layout, LangChain-native conversation memory, retry-safe API calls, structured logging, unit tests, and CI/CD - ready to extend or deploy.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.38-red)
![LangChain](https://img.shields.io/badge/langchain-0.3-1C3C3C)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Gayathri-Reddy874/langchain-groq-chatbot/actions/workflows/ci.yml/badge.svg)
[![Issues](https://img.shields.io/github/issues/Gayathri-Reddy874/langchain-groq-chatbot)](https://github.com/Gayathri-Reddy874/langchain-groq-chatbot/issues)
[![Last Commit](https://img.shields.io/github/last-commit/Gayathri-Reddy874/langchain-groq-chatbot)](https://github.com/Gayathri-Reddy874/langchain-groq-chatbot/commits/main)

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Development](#development)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## Features

- **True multi-turn memory, the LangChain-native way** — built on `RunnableWithMessageHistory` with a per-session `TrimmedChatMessageHistory`, so LangChain itself owns reading and appending turns instead of the app manually re-building a message list on every call.
- **Optional LangSmith tracing** — set three env vars and every chain run, prompt, and token count becomes inspectable at smith.langchain.com; the app shows a sidebar indicator when it's active.
- **Token streaming** — responses render word-by-word instead of waiting for the full completion.
- **Resilient API calls** — transient Groq errors are retried with exponential backoff (`tenacity`); permanent failures (bad key, rate limit) are shown as clear, human-readable messages instead of stack traces.
- **Configurable at runtime** — swap models, adjust temperature/max tokens, and edit the system prompt from the sidebar without touching code.
- **Chat export** — download the current conversation as JSON.
- **Modular architecture** — settings, the LangChain chain, and logging each live in their own module (`src/`) instead of one flat script, so the chain logic is testable independent of Streamlit.
- **Tested & linted** — `pytest` unit tests for config validation and message handling, `ruff` for style, both wired into GitHub Actions CI.
- **Containerized** — a slim Dockerfile for one-command deployment.

## Architecture

```
langchain-groq-chatbot/
├── app.py                  # Streamlit UI — session state, sidebar, chat loop
├── src/
│   ├── config.py            # AppSettings dataclass + validation
│   ├── chatbot.py            # LangChain chain: prompt | ChatGroq | parser, retry + streaming
│   └── logging_utils.py       # Structured logger factory
├── tests/
│   ├── test_config.py
│   └── test_chatbot.py
├── .github/workflows/ci.yml   # Lint + test on every push/PR
├── Dockerfile
├── requirements.txt
└── requirements-dev.txt
```

The chat chain is a standard LangChain LCEL pipeline, wrapped in LangChain's own history-management runnable:

```
RunnableWithMessageHistory(
    ChatPromptTemplate(system + history + question) | ChatGroq | StrOutputParser,
    get_session_history,   # -> TrimmedChatMessageHistory, keyed by Streamlit session_id
)
```

Each browser session gets its own `session_id` (a UUID stored in `st.session_state`), so `RunnableWithMessageHistory` automatically loads that session's prior turns before the call and appends the new exchange after — no manual message-list bookkeeping in the UI layer. `TrimmedChatMessageHistory` caps stored turns at `MAX_HISTORY_MESSAGES` so long conversations don't grow the prompt (and the Groq bill) without bound.

`src/chatbot.py` exposes `stream_response()` (used by the UI for a typing effect) and `get_response()` (a non-streaming variant, useful for tests or a future API layer), both wrapping the same chain so behavior stays consistent.

### LangSmith tracing (optional)

Add these to your `.env` to get full visibility into every chain run — prompts, latency, token usage — in the [LangSmith](https://smith.langchain.com) UI:

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your-key-here
LANGCHAIN_PROJECT=langchain-groq-chatbot
```

LangChain reads these directly from the environment, so no code changes are needed — the sidebar will show a "LangSmith tracing is on" indicator when active.

## Getting Started

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com)
- (Optional) Docker, if you'd rather run it containerized

### 1. Clone and install

```bash
git clone https://github.com/Gayathri-Reddy874/langchain-groq-chatbot.git
cd langchain-groq-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# edit .env and set GROQ_API_KEY=...
```

Never commit `.env` — it's already covered by `.gitignore`. `.env.example` should only ever contain placeholder values. Alternatively, skip `.env` entirely and paste the key into the sidebar when the app is running; it's never stored or logged.

### 3. Run

```bash
streamlit run app.py
```

Visit `http://localhost:8501`.

### Run with Docker

```bash
docker build -t langchain-groq-chatbot .
docker run -p 8501:8501 --env-file .env langchain-groq-chatbot
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check .        # lint
pytest -v           # unit tests
```

CI (`.github/workflows/ci.yml`) runs both on every push and pull request to `main`.

## Roadmap

- [ ] Optional Redis/SQLite-backed persistent chat history across sessions
- [ ] Pluggable model providers (OpenAI, Anthropic) behind the same interface
- [ ] Token/cost usage display per session

## Contributing

Contributions, issues, and feature requests are welcome.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes, and run `ruff check .` and `pytest -v` before committing
4. Commit (`git commit -m "Add your feature"`) and push
5. Open a pull request describing the change

Please keep PRs focused — one feature or fix per PR makes review much faster.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for the full text.

## Author

**Mallareddygari Gayathri**

- GitHub: [@Gayathri-Reddy874](https://github.com/Gayathri-Reddy874)

## Acknowledgments

- [LangChain](https://python.langchain.com/) — orchestration framework for the chain and conversation memory
- [Groq](https://groq.com/) — LPU inference engine powering fast completions
- [Streamlit](https://streamlit.io/) — the UI layer
- [LangSmith](https://smith.langchain.com/) — optional tracing and observability

