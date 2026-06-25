# AI POD Project Handoff Document

## Project Overview

AI POD is an internal HR and IT policy assistant for GIG EGYPT LIFE TAKAFUL. It uses a Retrieval-Augmented Generation workflow to search approved company policy documents, retrieve relevant policy chunks, and generate polished answers for employees in Arabic or English.

The app is built with Streamlit and includes authentication, chat history, answer caching, source metadata, and a ChatGPT-style question-and-answer interface.

## Current Run Command

```powershell
python -m streamlit run web_interface.py
```

## Main Files

| File | Purpose |
|---|---|
| `web_interface.py` | Streamlit UI, authentication flow, chat rendering, sidebar, cache usage, streaming display |
| `query_system.py` | RAG engine, language detection, semantic search, Groq response generation, streaming API |
| `ingest_documents.py` | Document ingestion, chunking, embeddings, FAISS index creation |
| `config.py` | Central configuration for paths, models, thresholds, company metadata |
| `auth.py` | Login/session authentication |
| `.streamlit/config.toml` | Streamlit production toolbar setting |
| `README.md` | Setup and run instructions |

## Core Features Implemented

### Authentication

- Users must log in before using AI POD.
- The current logged-in user is shown in the sidebar.
- A logout button is available in the sidebar.
- Chat history is stored per user.

### Sidebar

- Uses Streamlit's sidebar.
- Includes:
  - AI POD label
  - logged-in user name
  - user role
  - logout button
  - new chat button
  - search chats field
  - last 10 recent chats
- The previous Quick Questions and old Recent Questions sections were removed.

### Chat Interface

- The main input is a single question bar with an Ask button.
- Summary/Detailed buttons were removed.
- All answers are generated in detailed mode by default.
- The question bar clears after a question is submitted.
- Chat history is preserved and can be reopened from recent chats.

### Real-Time Typing Effect

- Assistant responses are displayed progressively.
- Cached answers, local fallback answers, and Groq answers all type out in small chunks.
- A typing indicator appears while the assistant is generating.
- The final saved message is stored only once.
- Auto-scroll follows the latest generated text.

### Auto-Scroll Behavior

- The chat follows the newest message by default.
- If the user scrolls up, auto-follow pauses.
- A "Scroll to latest" button appears when the user is not at the bottom.
- Clicking "Scroll to latest" resumes auto-follow.

### Answer Formatting

- Raw document text is not shown.
- Empty bullets and malformed markdown are cleaned before display.
- Duplicate blank lines are collapsed.
- Headings stay close to their related content.
- Answers are rendered in a compact HR-assistant style.
- Document filenames and OCR artifacts are removed from the answer body.

### Source and Time Metadata

- Each completed answer shows:
  - source document name or names
  - response time
- Cached answers also keep source information when available.
- If cached source information is missing, the app tries to backfill it using semantic search.

### Language Enforcement

- The app detects the dominant language of the user's question.
- Arabic questions receive Arabic-only answers.
- English questions receive English-only answers.
- Mixed-language questions use the dominant language.
- Bilingual answers are only allowed when explicitly requested.
- Arabic answers are displayed right-to-left and aligned to the right.
- English answers are displayed left-to-right and aligned to the left.

### Caching

- Answer cache file:

```text
cache/answer_cache.json
```

- Chat session files:

```text
cache/chat_sessions/{username}.json
```

- Legacy/simple chat history path:

```text
cache/chat_history/{username}.json
```

- Answer cache is app-wide because the answer cache key is based on the question, answer style, index stamp, app version, and answer format version.
- Chat history and chat sessions are per user.
- Cached answers are still typed out visually so the user experience remains consistent.

### Streamlit Toolbar Setting

The project includes:

```toml
[client]
toolbarMode = "viewer"
```

in:

```text
.streamlit/config.toml
```

This hides Streamlit developer options such as Clear caches for regular users while keeping the app itself functional.

## RAG Flow

1. User asks a question.
2. The app detects the question language.
3. `query_system.py` embeds the question.
4. FAISS retrieves relevant document chunks.
5. Results are reranked using semantic score, language match, and keyword overlap.
6. If the result passes threshold:
   - Groq generates a polished HR/IT answer from the retrieved chunks.
   - If Groq is unavailable, a local fallback answer is used.
7. If the result is too weak:
   - AI POD says it could not find enough information or uses a general fallback where appropriate.
8. The final answer is cleaned, rendered, cached, and saved to chat history.

## Streaming Flow

The streaming implementation is split between two files:

### `query_system.py`

- `_groq_stream(...)` streams token deltas from Groq.
- `_call_groq_policy(..., stream=True)` streams policy answers.
- `_call_groq_general(..., stream=True)` streams general answers.
- `ask_stream(...)` yields events:

```python
{"type": "delta", "text": "..."}
{"type": "done", "result": {...}}
```

### `web_interface.py`

- `render_chat_history(transient_chat=...)` renders a temporary in-progress assistant message.
- `TYPE_DELAY_SECONDS` controls the visible typing delay.
- `stream_text_chunks(...)` splits cached/local answers into smaller chunks.
- The final message is appended to `st.session_state.chat_history` only after streaming completes.

## Important Current Settings

| Setting | Current Value |
|---|---|
| Company name | `GIG EGYPT LIFE TAKAFUL` |
| App version | `1.3.0` |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` |
| Fast LLM | `llama-3.1-8b-instant` |
| Advanced LLM | `llama-3.3-70b-versatile` |
| Default answer mode | Detailed |
| Answer format version | `compact-answer-format-v1` |
| Streamlit toolbar mode | `viewer` |

## Environment Requirement

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

Without `GROQ_API_KEY`, the app can still run, but generated answers will fall back to local/basic behavior instead of full LLM synthesis.

## Document Ingestion

Policy documents should be placed in:

```text
HRandIT_documents/
```

Then run:

```powershell
python ingest_documents.py
```

This creates or updates:

```text
indices/faiss.index
indices/chunks.pkl
indices/metadata.pkl
indices/thresholds.pkl
```

Run ingestion again whenever policy documents are added, removed, or changed.

## Verification Commands Used

The following checks were used throughout development:

```powershell
python -m py_compile web_interface.py
python -m py_compile web_interface.py query_system.py
python -m streamlit config show
```

## Production Notes

- Use `.streamlit/config.toml` with `toolbarMode = "viewer"` for production users.
- Keep `.env` private and do not commit it.
- Re-run `ingest_documents.py` after any document update.
- Clear or rotate `cache/answer_cache.json` when policy content changes significantly.
- Chat history is stored locally under `cache/`, so deployment storage behavior matters.
- If deploying to a stateless host, cache and chat files may not persist across restarts unless persistent storage is configured.

## Known Behavior

- Cached answers are shared across users because the answer cache is app-wide.
- Chat sessions are per user.
- Cached answers are typed out visually, but the underlying answer is retrieved instantly.
- Streamlit reruns the script after interactions, so session-state changes must happen before widgets are instantiated or through rerun flags.

## Recent UI Decisions

- Quick Questions section removed.
- Old sidebar Recent Questions section removed.
- Admin/user profile and logout live in the sidebar.
- Reset Sidebar button removed.
- Summary/Detailed response controls removed.
- Clear button removed from the main question input.
- Main answer style is detailed by default.
- Source and response time are shown under each final answer.
- Arabic responses align right; English responses align left.
- Streamlit native sidebar was restored after trying a custom from-scratch sidebar.

## Suggested Next Improvements

- Add an admin-only page for cache management.
- Add a document update workflow that clears stale cached answers automatically.
- Add export/download for chat transcripts.
- Add role-based permissions for sensitive policy categories.
- Add automated tests for language enforcement and answer cleaning.
- Add deployment-specific persistent storage for cache and chat sessions.

