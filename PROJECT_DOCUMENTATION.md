# AI POD Project Documentation

## 1. Overview

AI POD is an internal AI assistant for GIG EGYPT LIFE TAKAFUL. It helps employees ask questions about HR and IT policies, while also supporting normal conversational chat for non-policy questions.

The system combines:

- Document ingestion and semantic search
- FAISS vector retrieval
- Groq LLM answer generation
- Arabic and English language handling
- Streamlit web UI
- Local authentication
- Per-user chat sessions
- App-wide answer cache
- Admin analytics dashboard

For policy questions, AI POD retrieves relevant company document chunks and sends them to Groq for answer generation. For casual/general questions, it skips document search and uses a dedicated general chat path.

## 2. Current Run Command

Run the Streamlit app from the project root:

```powershell
python -m streamlit run web_interface.py
```

Run the document ingestion pipeline:

```powershell
python ingest_documents.py
```

Run CLI testing:

```powershell
python query_system.py
```

## 3. Project Structure

```text
AI-POD-GIG/
├── auth.py                    # Login, local users, roles, session state
├── config.py                  # Model, path, document, and app configuration
├── ingest_documents.py        # Document extraction, chunking, embeddings, FAISS index
├── query_system.py            # RAG engine, classification, Groq generation, memory
├── web_interface.py           # Streamlit UI, chat, cache, dashboard, rendering
├── README.md                  # Original setup guide
├── PROJECT_DOCUMENTATION.md   # This documentation
├── AI_POD_PROJECT_HANDOFF.md  # Development handoff summary
├── requirements.txt           # Python dependencies
├── .env                       # Local secrets, not for commit
├── .gitignore
├── .streamlit/
│   └── config.toml            # Streamlit toolbar mode
├── HRandIT_documents/         # Source policy documents
├── indices/                   # Generated FAISS/vector artifacts
├── cache/                     # Answer cache and chat sessions
└── logs/
```

## 4. Main Components

### `config.py`

Central configuration for:

- Groq API key
- embedding model
- Groq model names
- company metadata
- document and index paths
- chunking settings
- search thresholds
- allowed file types

Important current values:

```python
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_FAST = "llama-3.1-8b-instant"
LLM_MODEL_ADVANCED = "llama-3.3-70b-versatile"
VERSION = "1.3.0"
```

### `ingest_documents.py`

Responsible for:

- reading files from `HRandIT_documents/`
- extracting text from PDF/TXT/DOCX/MD files
- detecting document language
- chunking documents
- generating sentence-transformer embeddings
- building FAISS index
- saving metadata and thresholds

Generated files:

```text
indices/faiss.index
indices/chunks.pkl
indices/metadata.pkl
indices/thresholds.pkl
```

Run ingestion again whenever policy documents change.

### `query_system.py`

Core AI logic:

- detects user language
- classifies intent
- performs semantic search
- rewrites follow-up questions for retrieval
- calls Groq for policy/general answers
- maintains short-term conversation memory
- streams generated responses
- enforces strict Arabic/English response language

### `web_interface.py`

Streamlit application:

- login gate
- sidebar
- chat interface
- typed streaming response
- answer formatting
- cache handling
- source/time metadata
- recent chats
- admin dashboard
- dark/light mode

### `auth.py`

Current local authentication:

- users stored in `users.json`
- roles include `admin`, `hr`, `employee`
- `check_permission(["admin"])` controls admin-only UI
- session timeout defaults to 8 hours

Future production recommendation: replace local login with Microsoft Entra ID authentication.

## 5. Environment Variables

Create `.env` in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
```

Important: The current production behavior requires Groq for generated answers. If the key is missing or invalid, policy generation returns a clean unavailable message.

Do not commit `.env`.

## 6. Streamlit Configuration

`.streamlit/config.toml`:

```toml
[client]
toolbarMode = "viewer"
```

This hides Streamlit developer options like Clear caches for normal users.

## 7. Answer Generation Behavior

AI POD has four major intent paths.

### Greeting

Examples:

- `hello`
- `good morning`
- `مرحبا`

Uses canned friendly responses.

### About Bot

Examples:

- `what can you do?`
- `who are you?`
- `كيف تساعد؟`

Uses canned descriptions of AI POD.

### General Chat

Examples:

- `what's the capital of France?`
- `can you help me write an email?`
- `lol thanks`
- `what's the weather like?`

This path skips document retrieval and calls Groq directly using `_call_groq_chat(...)`.

Returned result:

```python
mode = "general_chat"
match_type = "conversational"
sources = []
```

The UI omits source/time metadata for conversational answers.

### Policy Query

Examples:

- `how many annual leave days do I get?`
- `what is the password policy?`
- `كم يوم إجازة سنوية؟`

This path:

1. Optionally rewrites follow-up question for retrieval.
2. Runs semantic search.
3. Retrieves document chunks.
4. Sends chunks and original question to Groq.
5. Returns a polished HR/IT answer.

## 8. Follow-Up Query Rewriting

`query_system.py` includes:

```python
_resolve_followup_query(question)
```

Purpose:

If a user asks a follow-up like:

```text
How do I apply for it?
What about for managers?
```

the app rewrites the retrieval query using recent memory so FAISS can retrieve the right chunks.

Important behavior:

- The original question is still shown to the user.
- The original question is stored in memory.
- The original question is sent to the answer prompt.
- Only the semantic search query changes.

Debug output appears when rewriting happens:

```text
Resolved follow-up query: 'How do I apply for it?' -> 'How do I apply for annual leave?'
```

Fail-safe:

If rewriting fails or Groq is unavailable, the original question is used.

## 9. Groq Requirement

Recommended production behavior is implemented:

- Keep document retrieval.
- Require Groq for fresh answer generation.
- Keep cache enabled to reduce cost.
- If Groq fails, show a clean unavailable message.

English unavailable message:

```text
AI answer generation is currently unavailable. Please try again later.
```

Arabic unavailable message:

```text
خدمة توليد الإجابات غير متاحة حالياً. يرجى المحاولة مرة أخرى لاحقاً.
```

Cached answers can still be returned without a new Groq call.

## 10. Caching

Answer cache:

```text
cache/answer_cache.json
```

Cache key includes:

- normalized question
- answer style
- index stamp
- app version
- answer format version

Current answer format version:

```python
ANSWER_FORMAT_VERSION = "compact-answer-format-v1"
```

Cache scope:

- Answer cache is app-wide.
- Chat sessions are per user.

Cache behavior:

- Cached answers are typed out visually for consistent UX.
- Cached answers preserve sources where available.
- Missing cached sources are backfilled using semantic search where possible.

## 11. Chat Sessions

Per-user chat sessions:

```text
cache/chat_sessions/{username}.json
```

Legacy/simple chat history:

```text
cache/chat_history/{username}.json
```

The sidebar shows the last 10 recent chats for the current user.

## 12. UI Features

### Chat

- ChatGPT-style message layout
- typed response effect
- typing indicator
- auto-scroll to latest
- pause auto-scroll when user scrolls up
- "Scroll to latest" button
- Arabic answers right-aligned and RTL
- English answers left-aligned and LTR

### Composer

- single input bar
- Ask button
- clears after submit
- response mode is detailed by default

### Sidebar

- logged-in user
- role
- logout
- new chat
- search chats
- recent chats

### Icons

Inline Lucide-style SVG icons are used directly in `web_interface.py`.

Examples:

- Search: `search`
- Ask: `send`
- New Chat: `plus-circle`
- Logout: `log-out`
- Admin Dashboard: `layout-dashboard`
- Cache: `database`
- No Answer: `circle-alert`
- Source: `file-text`
- Time: `clock`

No icon package is required.

## 13. Admin Dashboard

Visible only to users with role:

```python
admin
```

Dashboard displays department-scoped analytics:

- total questions today
- total questions this week
- total questions this month
- questions with no answer
- cache hit rate
- top 10 most common questions
- recent unanswered questions

Department scope is based on the admin user's department where available.

The dashboard uses custom themed HTML tables for dark/light mode support.

## 14. Language Rules

AI POD enforces:

- Arabic question -> Arabic answer
- English question -> English answer
- Bilingual answer only when explicitly requested
- Mixed language uses dominant language
- Arabic layout is RTL and right-aligned
- English layout is LTR and left-aligned

Language enforcement is handled in:

```python
detect_language(...)
wants_bilingual_response(...)
_enforce_answer_language(...)
```

## 15. Answer Formatting Rules

The UI cleans generated markdown before rendering:

- removes empty bullets
- collapses duplicate blank lines
- keeps headings near content
- removes raw source/file references from answer body
- formats bullets and tables
- avoids raw OCR-style chunks

Final answer metadata can include:

- source document name
- response time

Conversational answers omit source metadata.

## 16. Production Notes

Recommended before production:

1. Rotate any exposed Groq API key.
2. Replace local login with Microsoft Entra ID.
3. Move JSON storage to SQL Server/PostgreSQL.
4. Add persistent storage if deploying to a stateless host.
5. Add audit logs.
6. Add feedback buttons.
7. Add admin review queue for unanswered/low-confidence questions.
8. Add document management and re-ingestion UI.
9. Add automated tests.
10. Add rate limiting.

## 17. Troubleshooting

### Invalid Groq API Key

Symptom:

```text
Error code: 401
Invalid API Key
```

Fix:

1. Create a new key in Groq.
2. Update `.env`.
3. Restart Streamlit.

### FAISS Index Missing

Symptom:

```text
Index not found. Please run: python ingest_documents.py
```

Fix:

```powershell
python ingest_documents.py
```

### Answers Say API Unavailable

Likely causes:

- missing `GROQ_API_KEY`
- invalid key
- Groq outage
- network issue

Check `.env` and restart the app.

### Dashboard Cannot Scroll

The app uses a scrollable page and chat-specific scroll container. If dashboard scrolling breaks, check `.block-container` CSS in `web_interface.py`.

### Arabic Text Looks Wrong

Check:

- document extraction quality
- `pymupdf` installation
- source PDF text layer
- language markers in files

## 18. Verification Commands

Compile checks:

```powershell
python -m py_compile web_interface.py query_system.py config.py auth.py ingest_documents.py
```

Streamlit config check:

```powershell
python -m streamlit config show
```

Classifier quick check:

```powershell
python -c "from query_system import classify_intent; print(classify_intent('how many sick days do I get'))"
```

## 19. Suggested Next Improvements

High value:

- Microsoft authentication
- rate limiting
- feedback buttons
- low-confidence review queue
- source preview
- document management page
- export dashboard data
- persistent database

Medium value:

- generated chat titles
- suggested follow-up questions
- copy answer button
- user activity dashboard
- admin cache controls

Longer-term:

- role-based document permissions
- group-based permissions from Microsoft Entra ID
- automated policy freshness checks
- usage/cost monitoring

