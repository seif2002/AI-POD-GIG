"""
AI POD Web Interface - Professional Version with Authentication
ENTER submits, Beautiful bullet points, ChatGPT-style UI
"""

import streamlit as st
import os
import sys
import time
import json
import html
from datetime import datetime
import hashlib
import re
import streamlit.components.v1 as components
# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from config and query system
try:
    from query_system import AIPodQuerySystem, detect_language, wants_bilingual_response
    from config import AIPodConfig
    from auth import AuthManager, init_session_state, login_required, logout, check_permission
    AI_POD_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import: {e}")
    AI_POD_AVAILABLE = False

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title=f"AI POD - {AIPodConfig.COMPANY_NAME}",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Initialize Authentication
# --------------------------------------------------
init_session_state()

# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------
if "question" not in st.session_state:
    st.session_state.question = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = hashlib.md5(
        str(datetime.now()).encode()
    ).hexdigest()[:8]

if "answer_style" not in st.session_state:
    st.session_state.answer_style = "detailed"

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "light"

if "chat_history_loaded" not in st.session_state:
    st.session_state.chat_history_loaded = False

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "chat_search" not in st.session_state:
    st.session_state.chat_search = ""

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
ANSWER_CACHE_PATH = os.path.join(CACHE_DIR, "answer_cache.json")
CHAT_HISTORY_DIR = os.path.join(CACHE_DIR, "chat_history")
CHAT_SESSIONS_DIR = os.path.join(CACHE_DIR, "chat_sessions")
ANSWER_CACHE_MAX_ITEMS = 500
ANSWER_FORMAT_VERSION = "compact-answer-format-v1"


def _ensure_cache_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)


def _load_json(path, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path, data):
    _ensure_cache_dirs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_user_key(username: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", username or "anonymous")


def _history_path() -> str:
    user = st.session_state.get("user") or {}
    username = _safe_user_key(user.get("username", "anonymous"))
    return os.path.join(CHAT_HISTORY_DIR, f"{username}.json")


def _sessions_path() -> str:
    user = st.session_state.get("user") or {}
    username = _safe_user_key(user.get("username", "anonymous"))
    return os.path.join(CHAT_SESSIONS_DIR, f"{username}.json")


def _new_chat_id() -> str:
    return hashlib.md5(f"{datetime.now().isoformat()}:{time.time()}".encode()).hexdigest()[:12]


def _empty_sessions_store() -> dict:
    return {"current_chat_id": None, "chats": {}}


def load_chat_sessions() -> dict:
    store = _load_json(_sessions_path(), _empty_sessions_store())
    if not isinstance(store, dict) or not isinstance(store.get("chats"), dict):
        store = _empty_sessions_store()

    old_history = _load_json(_history_path(), [])
    if not store["chats"] and isinstance(old_history, list) and old_history:
        chat_id = _new_chat_id()
        store["current_chat_id"] = chat_id
        store["chats"][chat_id] = {
            "title": old_history[0].get("question", "Previous chat")[:60],
            "created_at": old_history[0].get("timestamp", datetime.now().isoformat()),
            "updated_at": old_history[-1].get("timestamp", datetime.now().isoformat()),
            "messages": old_history[-100:],
        }
        _save_json(_sessions_path(), store)

    empty_chat_ids = [
        chat_id
        for chat_id, chat in store.get("chats", {}).items()
        if not chat.get("messages")
    ]
    for chat_id in empty_chat_ids:
        store["chats"].pop(chat_id, None)
    if empty_chat_ids:
        if store.get("current_chat_id") in empty_chat_ids:
            store["current_chat_id"] = None
        _save_json(_sessions_path(), store)

    return store


def save_chat_sessions(store: dict):
    _save_json(_sessions_path(), store)


def _chat_title(messages: list) -> str:
    if messages:
        return messages[0].get("question", "New chat")[:60]
    return "New chat"


def load_persisted_chat_history():
    if st.session_state.chat_history_loaded:
        return
    store = load_chat_sessions()
    chats = store.get("chats", {})
    chat_id = store.get("current_chat_id")
    if not chat_id or chat_id not in chats:
        sorted_chats = sorted(chats.items(), key=lambda item: item[1].get("updated_at", ""), reverse=True)
        chat_id = sorted_chats[0][0] if sorted_chats else _new_chat_id()
    if chat_id not in chats:
        chats[chat_id] = {
            "title": "New chat",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
        }
        store["current_chat_id"] = chat_id
        save_chat_sessions(store)
    st.session_state.current_chat_id = chat_id
    st.session_state.chat_history = chats[chat_id].get("messages", [])[-100:]
    st.session_state.chat_history_loaded = True


def save_persisted_chat_history():
    if not st.session_state.chat_history:
        return
    store = load_chat_sessions()
    chats = store.setdefault("chats", {})
    chat_id = st.session_state.current_chat_id or store.get("current_chat_id") or _new_chat_id()
    now = datetime.now().isoformat()
    existing = chats.get(chat_id, {})
    chats[chat_id] = {
        "title": _chat_title(st.session_state.chat_history),
        "created_at": existing.get("created_at", now),
        "updated_at": now,
        "messages": st.session_state.chat_history[-100:],
    }
    store["current_chat_id"] = chat_id
    st.session_state.current_chat_id = chat_id
    save_chat_sessions(store)


def clear_persisted_chat_history():
    st.session_state.chat_history = []
    st.session_state.question = ""
    save_persisted_chat_history()


def start_new_chat(ai_pod=None):
    """Save the current conversation and start a fresh chat session."""
    if st.session_state.current_chat_id and st.session_state.chat_history:
        save_persisted_chat_history()
    store = load_chat_sessions()
    chat_id = _new_chat_id()
    now = datetime.now().isoformat()
    store["current_chat_id"] = chat_id
    save_chat_sessions(store)
    st.session_state.current_chat_id = chat_id
    st.session_state.chat_history = []
    st.session_state.question = ""
    st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    st.session_state.ai_memory_rehydrated = False
    if ai_pod and hasattr(ai_pod, "clear_memory"):
        ai_pod.clear_memory()


def switch_chat(chat_id: str, ai_pod=None):
    """Save current chat, then load another saved chat."""
    if st.session_state.chat_history:
        save_persisted_chat_history()
    store = load_chat_sessions()
    chat = store.get("chats", {}).get(chat_id)
    if not chat:
        return
    store["current_chat_id"] = chat_id
    save_chat_sessions(store)
    st.session_state.current_chat_id = chat_id
    st.session_state.chat_history = chat.get("messages", [])[-100:]
    st.session_state.question = ""
    st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
    if ai_pod and hasattr(ai_pod, "clear_memory"):
        ai_pod.clear_memory()
        for message in st.session_state.chat_history[-6:]:
            ai_pod.memory.add(message.get("question", ""), message.get("answer", ""))
        st.session_state.ai_memory_rehydrated = True
    else:
        st.session_state.ai_memory_rehydrated = False


def _cache_key(question: str, answer_style: str, ai_pod) -> str:
    index_stamp = ""
    if ai_pod and getattr(ai_pod, "metadata", None):
        index_stamp = ai_pod.metadata.get("built_at", "") or ai_pod.metadata.get("version", "")
    payload = {
        "question": " ".join(question.lower().split()),
        "answer_style": answer_style,
        "index_stamp": index_stamp,
        "app_version": getattr(AIPodConfig, "VERSION", ""),
        "answer_format_version": ANSWER_FORMAT_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def get_cached_answer(question: str, answer_style: str, ai_pod):
    cache = _load_json(ANSWER_CACHE_PATH, {})
    key = _cache_key(question, answer_style, ai_pod)
    item = cache.get(key)
    if not item:
        return None
    result = item.get("result")
    if not isinstance(result, dict):
        return None
    answer = result.get("answer", "")
    question_lang = detect_language(question)
    bilingual = wants_bilingual_response(question)
    if not bilingual and question_lang == "ar" and re.search(r"[A-Za-z]", answer):
        return None
    if not bilingual and question_lang == "en" and re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", answer):
        return None

    if not result.get("sources") and ai_pod and hasattr(ai_pod, "search_semantic"):
        try:
            sources = ai_pod.search_semantic(question)[:AIPodConfig.TOP_K_RESULTS]
            result["sources"] = sources
            item["result"] = result
            cache[key] = item
            _save_json(ANSWER_CACHE_PATH, cache)
        except Exception:
            result["sources"] = []

    result["cached"] = True
    result["response_time"] = 0.0
    return result


def save_cached_answer(question: str, answer_style: str, ai_pod, result: dict):
    cache = _load_json(ANSWER_CACHE_PATH, {})
    cache[_cache_key(question, answer_style, ai_pod)] = {
        "created_at": datetime.now().isoformat(),
        "question": question,
        "answer_style": answer_style,
        "result": {
            "answer": result.get("answer", ""),
            "confidence": float(result.get("confidence", 0)),
            "mode": result.get("mode", ""),
            "match_type": result.get("match_type", "none"),
            "language": result.get("language", "en"),
            "response_time": float(result.get("response_time", 0)),
            "sources": result.get("sources", []),
        },
    }
    if len(cache) > ANSWER_CACHE_MAX_ITEMS:
        ordered = sorted(cache.items(), key=lambda kv: kv[1].get("created_at", ""))
        cache = dict(ordered[-ANSWER_CACHE_MAX_ITEMS:])
    _save_json(ANSWER_CACHE_PATH, cache)

# --------------------------------------------------
# Professional Answer Formatter - REAL HTML BULLETS
# --------------------------------------------------
def format_inline(text: str) -> str:
    """Escape text and apply minimal markdown-style inline formatting."""
    escaped = html.escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)


def clean_display_text(text: str) -> str:
    """Remove characters that can confuse Markdown/HTML rendering."""
    if text is None:
        return ""
    text = str(text)
    return "".join(
        ch
        for ch in text
        if ch in "\n\t" or ord(ch) >= 32
    )


def clean_response_markdown(answer_text: str) -> str:
    """Normalize assistant markdown before rendering it as chat HTML."""
    text = clean_display_text(answer_text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    section_names = [
        "Direct Answer",
        "Details",
        "Additional Notes",
        "Important Notes",
        "Annual Leave",
        "Sick Leave",
        "Weekly Offs",
        "Maternity Leave",
        "Paternity Leave",
        "Related Leave",
        "Accrual and Carryover",
        "الإجابة المباشرة",
        "التفاصيل",
        "ملاحظات إضافية",
        "ملاحظات مهمة",
        "الإجازة السنوية",
        "الإجازة المرضية",
        "أيام الراحة الأسبوعية",
        "إجازة الأمومة",
        "إجازات مرتبطة",
    ]
    for name in section_names:
        text = re.sub(rf"(?<!\n)({re.escape(name)}:)", r"\n\1", text)

    cleaned_lines = []
    seen_bullets = set()
    last_blank = False
    last_heading = None

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        line = re.sub(r"^#{1,6}\s*", "", line).strip()
        line = re.sub(r"^\s*[-*•]\s*[-*•\s]*$", "", line).strip()
        line = re.sub(r"^\s*\d+[\.\)]\s*$", "", line).strip()

        bullet_match = re.match(r"^([-*•]|\d+[\.\)])\s*(.*)$", line)
        if bullet_match:
            marker = "-" if bullet_match.group(1) in {"-", "*", "•"} else bullet_match.group(1)
            body = bullet_match.group(2).strip()
            body = re.sub(r"^[-*•\s]+", "", body).strip()
            if not body:
                continue
            bullet_key = re.sub(r"\s+", " ", body.lower())
            if bullet_key in seen_bullets:
                continue
            seen_bullets.add(bullet_key)
            line = f"{marker} {body}"

        heading_candidate = line.rstrip(":").strip("*").strip()
        is_heading = bool(line.endswith(":") and len(heading_candidate) <= 60 and not bullet_match)
        if is_heading:
            heading_key = heading_candidate.lower()
            if heading_key == last_heading:
                continue
            last_heading = heading_key
            line = f"{heading_candidate}:"
        elif line:
            last_heading = None

        if not line:
            if cleaned_lines and not last_blank:
                cleaned_lines.append("")
                last_blank = True
            continue

        cleaned_lines.append(line)
        last_blank = False

    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^([^:\n]{3,60}):\n\n+", r"\1:\n", text)
    return text


def format_bullet_text(text: str) -> str:
    """Format bullet text and bold a short leading label before a colon."""
    cleaned = text.strip()
    label_match = re.match(r"^(\*\*)?([A-Za-z][A-Za-z /&()'-]{1,42}):(\*\*)?\s*(.*)$", cleaned)
    if label_match:
        label = html.escape(label_match.group(2).strip())
        rest = format_inline(label_match.group(4).strip())
        return f"<strong>{label}:</strong> {rest}".strip()
    return format_inline(cleaned)


def normalize_answer_text(answer_text: str) -> str:
    """Repair common LLM markdown spacing issues before HTML formatting."""
    text = clean_response_markdown(answer_text)
    text = re.sub(r"(?<!\n)\s+(\*?\*\*[A-Z][A-Za-z /&()'-]{1,50}:\*\*)", r"\n\n\1", text)
    text = re.sub(r"(?<!\n)\s+(\*[A-Z][A-Za-z /&()'-]{1,50}:\*\*)", r"\n\n\1", text)
    return text


def render_markdown_table(lines: list) -> str:
    rows = []
    for raw in lines:
        cells = [format_inline(cell.strip()) for cell in raw.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    html_table = "<table class='answer-table'><thead><tr>"
    html_table += "".join(f"<th>{cell}</th>" for cell in header)
    html_table += "</tr></thead><tbody>"
    for row in body:
        html_table += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    html_table += "</tbody></table>"
    return html_table


def format_answer(answer_text: str) -> str:
    """Convert answer into clean HTML bullet format - Professional version"""
    
    lines = normalize_answer_text(answer_text).split("\n")
    html_output = ""
    bullet_mode = False
    table_lines = []

    def flush_table():
        nonlocal html_output, table_lines
        if table_lines:
            html_output += render_markdown_table(table_lines)
            table_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            flush_table()
            continue
        line = re.sub(r"^#{1,4}\s*", "", line)

        if "|" in line and line.count("|") >= 2:
            if bullet_mode:
                html_output += "</ul>"
                bullet_mode = False
            table_lines.append(line)
            continue

        flush_table()
        
        # Handle source citation
        if line.startswith('[From:') or line.startswith('From:'):
            if bullet_mode:
                html_output += "</ul>"
                bullet_mode = False
            html_output += f'<div class="source-citation">{format_inline(line)}</div>'
            continue

        # Handle markdown bold section headers, including malformed *Annual Leave:**
        header_match = re.match(r"^\*?\*\*?(.+?):\*\*\s*(.*)$", line)
        if header_match and len(header_match.group(1)) < 60:
            if bullet_mode:
                html_output += "</ul>"
                bullet_mode = False
            title = header_match.group(1).strip()
            rest = header_match.group(2).strip()
            html_output += f"<h4 class='section-header'>{format_inline(title)}</h4>"
            if rest:
                html_output += f"<p>{format_inline(rest)}</p>"
            continue
        
        # Handle numbered list (1., 2., etc)
        if re.match(r"^\d+[\.\)]", line):
            clean_line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if not clean_line:
                continue
            if not bullet_mode:
                html_output += "<ul class='bullet-list'>"
                bullet_mode = True
            html_output += f"<li>{format_bullet_text(clean_line)}</li>"
        
        # Handle dash/star list
        elif line.startswith("-") or line.startswith("*") or line.startswith("•"):
            clean_line = line[1:].strip()
            clean_line = re.sub(r"^[-*•\s]+", "", clean_line).strip()
            if not clean_line:
                continue
            if not bullet_mode:
                html_output += "<ul class='bullet-list'>"
                bullet_mode = True
            html_output += f"<li>{format_bullet_text(clean_line)}</li>"
        
        # Handle bullet points already in text
        elif "•" in line:
            if not bullet_mode:
                html_output += "<ul class='bullet-list'>"
                bullet_mode = True
            parts = line.split("•")
            for part in parts:
                if part.strip():
                    html_output += f"<li>{format_bullet_text(part.strip())}</li>"
        
        # Handle section headers
        elif (line.endswith(':') or line in {"Direct Answer", "Details", "Additional Notes", "Important Notes"}) and len(line) < 60:
            if bullet_mode:
                html_output += "</ul>"
                bullet_mode = False
            line = line.strip("*").rstrip(":")
            html_output += f"<h4 class='section-header'>{format_inline(line)}</h4>"

        # Handle "Heading: text" on one line
        elif re.match(r"^[^:]{3,45}:\s+\S+", line):
            heading, rest = line.split(":", 1)
            if bullet_mode:
                html_output += "</ul>"
                bullet_mode = False
            html_output += f"<h4 class='section-header'>{format_inline(heading.strip())}</h4>"
            html_output += f"<p>{format_inline(rest.strip())}</p>"
        
        # Regular paragraph
        else:
            if bullet_mode:
                html_output += "</ul>"
                bullet_mode = False
            html_output += f"<p>{format_inline(line)}</p>"
    
    # Close any open bullet list
    if bullet_mode:
        html_output += "</ul>"
    flush_table()
    
    return html_output


def format_source_names(sources) -> str:
    """Return a compact, de-duplicated source label for chat metadata."""
    names = []
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            metadata = source.get("metadata") or {}
            name = metadata.get("file_name") or metadata.get("document")
            if name and name not in names:
                names.append(str(name))
    return ", ".join(names[:3]) if names else "Source unavailable"


def format_response_time(seconds) -> str:
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds < 0.01:
        return "cached"
    return f"{seconds:.2f}s"


def render_chat_history():
    """Render the full conversation thread."""
    force_follow = st.session_state.pop("force_chat_follow", False)
    force_attr = "true" if force_follow else "false"
    if not st.session_state.chat_history:
        st.markdown(
            f"""
            <div class="chat-scroll" id="aiPodChatScroll" data-chat-follow="true" data-force-follow="{force_attr}">
                <div class="empty-chat">
                    <div class="empty-chat-title">How can I help?</div>
                    <div class="empty-chat-subtitle">Ask about HR, IT, or company policy documents.</div>
                </div>
                <div id="aiPodChatBottom" class="chat-bottom-anchor"></div>
            </div>
            <button id="aiPodScrollLatest" class="scroll-latest-button" type="button">Scroll to latest</button>
            """,
            unsafe_allow_html=True
        )
        render_chat_autoscroll_script()
        return

    chat_markup = [f'<div class="chat-scroll" id="aiPodChatScroll" data-chat-follow="true" data-force-follow="{force_attr}">']
    for idx, chat in enumerate(st.session_state.chat_history):
        formatted_question = format_inline(clean_display_text(chat.get("question", "")))
        formatted_answer = format_answer(chat.get("answer", ""))
        chat_lang = chat.get("language") or detect_language(chat.get("question", ""))
        direction = "rtl" if chat_lang == "ar" else "ltr"
        lang_class = "arabic-answer" if chat_lang == "ar" else "english-answer"
        source_label = format_inline(format_source_names(chat.get("sources", [])))
        time_label = format_inline(format_response_time(chat.get("response_time", 0)))
        chat_markup.append(f'<div class="chat-row user-row"><div class="user-bubble {lang_class}" dir="{direction}">{formatted_question}</div></div>')
        chat_markup.append(
            '<div class="chat-row assistant-row"><div class="assistant-bubble">'
            f'<div class="answer-box {lang_class}" dir="{direction}">{formatted_answer}</div>'
            f'<div class="chat-meta {lang_class}" dir="{direction}">Source: {source_label} | Time: {time_label}</div>'
            '</div></div>'
        )
    chat_markup.append('<div id="aiPodChatBottom" class="chat-bottom-anchor"></div></div>')
    chat_markup.append('<button id="aiPodScrollLatest" class="scroll-latest-button" type="button">Scroll to latest</button>')
    st.markdown("".join(chat_markup), unsafe_allow_html=True)
    render_chat_autoscroll_script()


def render_chat_autoscroll_script():
    """Keep the chat pinned to the newest message unless the user scrolls up."""
    components.html(
        """
        <script>
        (() => {
            const FOLLOW_KEY = "aiPodChatAutoFollow";
            const LAST_HEIGHT_KEY = "aiPodChatLastScrollHeight";
            const NEAR_BOTTOM_PX = 48;
            const root = window.parent.document;

            const install = (attempt = 0) => {
                const scroll = root.getElementById("aiPodChatScroll");
                const button = root.getElementById("aiPodScrollLatest");
                if (!scroll || !button) {
                    if (attempt < 40) window.parent.setTimeout(() => install(attempt + 1), 75);
                    return;
                }

                if (window.parent.__aiPodChatAutoScrollCleanup) {
                    window.parent.__aiPodChatAutoScrollCleanup();
                }

                let storedFollow = window.parent.sessionStorage.getItem(FOLLOW_KEY);
                let autoFollow = storedFollow === null ? true : storedFollow === "true";
                const lastHeight = Number(window.parent.sessionStorage.getItem(LAST_HEIGHT_KEY) || 0);
                const contentChanged = scroll.scrollHeight !== lastHeight;
                const forceFollow = scroll.dataset.forceFollow === "true";
                if (forceFollow) {
                    autoFollow = true;
                    window.parent.sessionStorage.setItem(FOLLOW_KEY, "true");
                }
                let programmaticScroll = false;

                const distanceFromBottom = () => scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight;
                const isNearBottom = () => distanceFromBottom() <= NEAR_BOTTOM_PX;
                const setButtonVisible = (visible) => {
                    button.classList.toggle("is-visible", visible);
                };
                const setAutoFollow = (value) => {
                    autoFollow = value;
                    window.parent.sessionStorage.setItem(FOLLOW_KEY, String(value));
                    scroll.dataset.chatFollow = String(value);
                    setButtonVisible(!value && !isNearBottom());
                };
                const scrollToBottom = (behavior = "smooth") => {
                    programmaticScroll = true;
                    scroll.scrollTo({ top: scroll.scrollHeight, behavior });
                    window.parent.setTimeout(() => {
                        programmaticScroll = false;
                        window.parent.sessionStorage.setItem(LAST_HEIGHT_KEY, String(scroll.scrollHeight));
                        if (isNearBottom()) setAutoFollow(true);
                    }, behavior === "smooth" ? 500 : 75);
                };
                const handleScroll = () => {
                    if (programmaticScroll) return;
                    if (isNearBottom()) {
                        setAutoFollow(true);
                    } else {
                        setAutoFollow(false);
                    }
                };
                const handleLatestClick = () => {
                    setAutoFollow(true);
                    scrollToBottom("smooth");
                };

                scroll.addEventListener("scroll", handleScroll, { passive: true });
                button.addEventListener("click", handleLatestClick);

                const observer = new MutationObserver(() => {
                    window.parent.sessionStorage.setItem(LAST_HEIGHT_KEY, String(scroll.scrollHeight));
                    if (autoFollow || isNearBottom()) {
                        setAutoFollow(true);
                        scrollToBottom("smooth");
                    } else {
                        setButtonVisible(true);
                    }
                });
                observer.observe(scroll, { childList: true, subtree: true, characterData: true });

                window.parent.requestAnimationFrame(() => {
                    if (forceFollow || autoFollow || isNearBottom() || (contentChanged && storedFollow === null)) {
                        scrollToBottom(contentChanged ? "smooth" : "auto");
                    } else {
                        setButtonVisible(true);
                    }
                    window.parent.sessionStorage.setItem(LAST_HEIGHT_KEY, String(scroll.scrollHeight));
                });

                window.parent.__aiPodChatAutoScrollCleanup = () => {
                    scroll.removeEventListener("scroll", handleScroll);
                    button.removeEventListener("click", handleLatestClick);
                    observer.disconnect();
                };
            };
            install();
        })();
        </script>
        """,
        height=0,
    )

# --------------------------------------------------
# Custom CSS - ChatGPT Style Premium UI
# --------------------------------------------------
is_dark_mode = st.session_state.theme_mode == "dark"
theme_vars = {
    "bg": "#0F172A" if is_dark_mode else "#FFFFFF",
    "surface": "#111827" if is_dark_mode else "#FFFFFF",
    "composer": "#0F172A" if is_dark_mode else "#FFFFFF",
    "surface_soft": "#1E293B" if is_dark_mode else "#F9FAFB",
    "text": "#E5E7EB" if is_dark_mode else "#111827",
    "muted": "#94A3B8" if is_dark_mode else "#6B7280",
    "border": "#334155" if is_dark_mode else "#E5E7EB",
    "user_bubble": "#2563EB" if is_dark_mode else "#F3F4F6",
    "user_text": "#FFFFFF" if is_dark_mode else "#111827",
    "input_bg": "#0B1220" if is_dark_mode else "#FFFFFF",
    "button_bg": "#1E293B" if is_dark_mode else "#FFFFFF",
    "button_hover": "#334155" if is_dark_mode else "#3B82F6",
    "disabled_bg": "#0F172A" if is_dark_mode else "#F3F4F6",
    "disabled_text": "#475569" if is_dark_mode else "#CBD5E1",
    "shadow": "0 10px 28px rgba(0,0,0,0.35)" if is_dark_mode else "0 8px 24px rgba(17, 24, 39, 0.08)",
    "accent": "#60A5FA" if is_dark_mode else "#2563EB",
    "section": "#93C5FD" if is_dark_mode else "#1E3A8A",
}

st.markdown(f"""
<style>
    :root {{
        --app-bg: {theme_vars["bg"]};
        --surface: {theme_vars["surface"]};
        --composer: {theme_vars["composer"]};
        --surface-soft: {theme_vars["surface_soft"]};
        --text-main: {theme_vars["text"]};
        --text-muted: {theme_vars["muted"]};
        --border: {theme_vars["border"]};
        --user-bubble: {theme_vars["user_bubble"]};
        --user-text: {theme_vars["user_text"]};
        --input-bg: {theme_vars["input_bg"]};
        --button-bg: {theme_vars["button_bg"]};
        --button-hover: {theme_vars["button_hover"]};
        --disabled-bg: {theme_vars["disabled_bg"]};
        --disabled-text: {theme_vars["disabled_text"]};
        --shadow: {theme_vars["shadow"]};
        --accent: {theme_vars["accent"]};
        --section: {theme_vars["section"]};
    }}

    .stApp {{
        background: var(--app-bg);
        color: var(--text-main);
    }}

    .stMarkdown, .stText, .stMetric, label, [data-testid="stSidebar"] {{
        color: var(--text-main);
    }}

    [data-testid="stSidebar"] {{
        background: var(--surface-soft);
        display: block !important;
        visibility: visible !important;
        transition: transform 0.2s ease, width 0.2s ease, min-width 0.2s ease;
    }}

    [data-testid="stSidebarContent"] {{
        overflow-y: auto !important;
        padding: 4.5rem 1.25rem 1.25rem 1.25rem;
    }}

    button[data-testid="collapsedControl"],
    button[data-testid="baseButton-headerNoPadding"] {{
        opacity: 0 !important;
        pointer-events: none !important;
    }}

    button[data-testid="collapsedControl"]:hover,
    button[data-testid="baseButton-headerNoPadding"]:hover {{
        opacity: 0 !important;
    }}

    [data-testid="stTabs"] button {{
        color: var(--text-main);
    }}

    #aiPodSidebarToggle {{
        position: fixed;
        top: 18px;
        left: 246px;
        z-index: 9999;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: transparent;
        color: var(--text-muted);
        border: none;
        border-radius: 999px;
        cursor: pointer;
        font-size: 22px;
        line-height: 1;
        font-weight: 700;
        transition: background 0.15s ease, color 0.15s ease, left 0.2s ease;
    }}

    #aiPodSidebarToggle:hover {{
        background: var(--button-bg);
        color: var(--text-main);
    }}

    body.ai-pod-sidebar-collapsed [data-testid="stSidebar"] {{
        transform: translateX(-100%) !important;
        min-width: 0 !important;
        width: 0 !important;
        margin-left: 0 !important;
        overflow: hidden !important;
    }}

    body.ai-pod-sidebar-collapsed [data-testid="stSidebar"] * {{
        pointer-events: none !important;
    }}

    body.ai-pod-sidebar-collapsed #aiPodSidebarToggle {{
        left: 14px !important;
    }}

</style>
""", unsafe_allow_html=True)

components.html("""
<script>
(() => {
    const root = window.parent.document;
    const STORAGE_KEY = "aiPodSidebarCollapsed";
    let toggleBtn = root.getElementById("aiPodSidebarToggle");

    if (!toggleBtn) {
        toggleBtn = root.createElement("button");
        toggleBtn.id = "aiPodSidebarToggle";
        toggleBtn.type = "button";
        toggleBtn.setAttribute("aria-label", "Toggle sidebar");
        root.body.appendChild(toggleBtn);
    }

    const sidebarIsOpen = () => {
        return !root.body.classList.contains("ai-pod-sidebar-collapsed");
    };

    const positionToggle = () => {
        const sidebar = root.querySelector('[data-testid="stSidebar"]');
        const open = sidebarIsOpen();
        if (open && sidebar) {
            const rect = sidebar.getBoundingClientRect();
            toggleBtn.style.left = Math.max(14, rect.right - 34) + "px";
            toggleBtn.textContent = "«";
        } else {
            toggleBtn.style.left = "14px";
            toggleBtn.textContent = "»";
        }
    };

    const setCollapsed = (collapsed) => {
        root.body.classList.toggle("ai-pod-sidebar-collapsed", collapsed);
        window.parent.localStorage.setItem(STORAGE_KEY, collapsed ? "true" : "false");
        positionToggle();
    };

    toggleBtn.onclick = () => {
        setCollapsed(sidebarIsOpen());
    };

    setCollapsed(window.parent.localStorage.getItem(STORAGE_KEY) === "true");
    positionToggle();
    window.parent.addEventListener("resize", positionToggle);
    window.parent.setTimeout(positionToggle, 300);
    window.parent.setTimeout(positionToggle, 900);
})();
</script>
""", height=0)

st.markdown("""
<style>
    .block-container {
        max-width: 980px;
        height: 100vh;
        overflow: hidden;
        padding-top: 1.5rem;
        padding-bottom: 0.8rem;
        color: var(--text-main);
    }

    .chat-scroll {
        position: relative;
        height: calc(100vh - 34rem);
        min-height: 140px;
        overflow-y: auto;
        overflow-x: hidden;
        padding: 0.25rem 0.35rem 1rem 0;
        margin-bottom: 0.9rem;
        scrollbar-color: var(--border) transparent;
        scrollbar-width: thin;
        scroll-behavior: smooth;
    }

    .chat-bottom-anchor {
        width: 100%;
        height: 1px;
    }

    .scroll-latest-button {
        display: none;
        position: fixed;
        left: 50%;
        bottom: 9.2rem;
        transform: translateX(-50%);
        z-index: 50;
        background: var(--button-bg);
        color: var(--text-main);
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 0.5rem 0.9rem;
        font-size: 0.86rem;
        font-weight: 600;
        cursor: pointer;
        box-shadow: var(--shadow);
        transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease, opacity 0.2s ease;
    }

    .scroll-latest-button.is-visible {
        display: block;
    }

    .scroll-latest-button:hover {
        background: var(--button-hover);
        color: #FFFFFF;
        transform: translateX(-50%) translateY(-1px);
    }

    .chat-scroll::-webkit-scrollbar {
        width: 8px;
    }

    .chat-scroll::-webkit-scrollbar-track {
        background: transparent;
    }

    .chat-scroll::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 999px;
    }

    .chat-row {
        display: flex;
        width: 100%;
        margin: 0.75rem 0;
    }

    .user-row {
        justify-content: flex-end;
    }

    .assistant-row {
        justify-content: flex-start;
    }

    .user-bubble {
        max-width: 68%;
        background: var(--user-bubble);
        color: var(--user-text);
        padding: 0.85rem 1rem;
        border-radius: 18px 18px 4px 18px;
        line-height: 1.55;
        font-size: 1rem;
        overflow-wrap: anywhere;
    }

    .assistant-bubble {
        max-width: 82%;
        color: var(--text-main);
        line-height: 1.75;
        overflow-wrap: anywhere;
    }

    .answer-box.arabic-answer,
    .user-bubble.arabic-answer {
        direction: rtl;
        text-align: right;
        unicode-bidi: plaintext;
    }

    .answer-box.english-answer,
    .user-bubble.english-answer {
        direction: ltr;
        text-align: left;
    }

    .chat-meta {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 0.65rem;
        border-top: 1px solid var(--border);
        padding-top: 0.45rem;
        line-height: 1.4;
    }

    .chat-meta.arabic-answer {
        text-align: right;
        direction: rtl;
    }

    .chat-meta.english-answer {
        text-align: left;
        direction: ltr;
    }

    @media (max-width: 700px) {
        .user-bubble,
        .assistant-bubble {
            max-width: 94%;
        }
    }

    .empty-chat {
        min-height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: var(--text-main);
    }

    .empty-chat-title {
        font-size: 2rem;
        font-weight: 650;
        margin-bottom: 0.5rem;
    }

    .empty-chat-subtitle {
        color: var(--text-muted);
        font-size: 1rem;
    }

    .composer-shell {
        flex: 0 0 auto;
        margin: 0 auto;
        padding: 0;
        border: none;
        background: transparent;
        box-shadow: none;
    }

    .composer-mode {
        color: var(--text-muted);
        font-size: 0.82rem;
        margin: 0.25rem 0 0.7rem 0.2rem;
    }

    div[data-testid="stForm"] {
        margin-bottom: 0 !important;
    }

    /* Main header - Premium gradient */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: var(--text-muted);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Answer box - ChatGPT style */
    .answer-box {
        background: transparent;
        padding: 0;
        border-radius: 0;
        border: none;
        box-shadow: none;
        margin: 0;
        font-size: 1.05rem;
        line-height: 1.6;
        color: var(--text-main);
    }
    
    .answer-box:hover {
        box-shadow: none;
    }
    
    /* Paragraphs */
    .answer-box p {
        margin: 0.35rem 0 0.75rem 0;
        color: var(--text-main);
    }

    .answer-box strong {
        color: var(--section);
        font-weight: 700;
    }
    
    /* Bullet lists - Real HTML bullets */
    .answer-box ul.bullet-list {
        margin: 0.35rem 0 0.85rem 0;
        padding-left: 1.8rem;
        list-style-type: disc;
    }

    .answer-box:dir(rtl) ul.bullet-list {
        padding-left: 0;
        padding-right: 1.8rem;
        text-align: right;
        direction: rtl;
    }
    
    .answer-box ul.bullet-list li {
        margin-bottom: 0.35rem;
        color: var(--text-main);
        line-height: 1.55;
        padding-left: 0.5rem;
    }

    .answer-box:dir(rtl) ul.bullet-list li {
        padding-left: 0;
        padding-right: 0.5rem;
        text-align: right;
    }
    
    /* Section headers */
    .answer-box h4.section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--section);
        margin: 1rem 0 0.35rem 0;
        border-bottom: none;
        padding-bottom: 0;
    }

    .answer-box h4.section-header:first-child {
        margin-top: 0;
    }

    .answer-box.arabic-answer h4.section-header,
    .answer-box.arabic-answer p,
    .answer-box.arabic-answer li {
        text-align: right;
    }

    .answer-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.9rem 0 1.3rem 0;
        font-size: 0.95rem;
    }

    .answer-table th {
        color: var(--text-main);
        font-weight: 700;
        text-align: left;
        border-bottom: 1px solid var(--border);
        padding: 0.65rem 0.7rem;
    }

    .answer-table td {
        border-bottom: 1px solid var(--border);
        padding: 0.65rem 0.7rem;
        color: var(--text-main);
    }

    .answer-table tr:last-child td {
        border-bottom: none;
    }
    
    /* Source citation */
    .source-citation {
        background: var(--surface-soft);
        padding: 0.8rem 1.5rem;
        border-radius: 30px;
        font-size: 0.9rem;
        color: var(--text-muted);
        margin: 1.2rem 0 0.5rem 0;
        border-left: 4px solid var(--border);
        display: inline-block;
        font-family: 'SF Mono', monospace;
    }
    
    /* Confidence indicators */
    .confidence-high { 
        color: #10B981; 
        font-weight: 700;
        background: rgba(16, 185, 129, 0.1);
        padding: 6px 18px;
        border-radius: 30px;
        display: inline-block;
        font-size: 1.1rem;
    }
    
    .confidence-medium { 
        color: #F59E0B; 
        font-weight: 700;
        background: rgba(245, 158, 11, 0.1);
        padding: 6px 18px;
        border-radius: 30px;
        display: inline-block;
        font-size: 1.1rem;
    }
    
    .confidence-low { 
        color: #EF4444; 
        font-weight: 700;
        background: rgba(239, 68, 68, 0.1);
        padding: 6px 18px;
        border-radius: 30px;
        display: inline-block;
        font-size: 1.1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: var(--surface);
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.08);
    }
    
    /* Quick question buttons - Pill style */
    .stButton > button {
        background-color: var(--button-bg);
        color: var(--text-main);
        border: 1.5px solid var(--border);
        padding: 8px 20px;
        border-radius: 40px;
        font-size: 0.9rem;
        transition: all 0.2s ease;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: var(--button-hover);
        color: white;
        border-color: var(--accent);
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(59, 130, 246, 0.2);
    }

    .stButton > button:disabled,
    .stButton > button:disabled:hover {
        background-color: var(--disabled-bg);
        color: var(--disabled-text);
        border-color: var(--border);
        opacity: 1;
        transform: none;
        box-shadow: none;
    }

    /* Strong overrides for Streamlit controls inside the composer */
    div[data-testid="stForm"] button,
    div[data-testid="stForm"] .stButton > button,
    div[data-testid="stForm"] button[kind="secondary"] {
        background: var(--button-bg) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--border) !important;
        box-shadow: none !important;
        min-height: 2.35rem !important;
        padding: 0.35rem 0.8rem !important;
        border-radius: 999px !important;
        white-space: nowrap !important;
        margin-top: 0 !important;
        pointer-events: auto !important;
        position: relative !important;
        z-index: 3 !important;
    }

    div[data-testid="stForm"] button:hover,
    div[data-testid="stForm"] .stButton > button:hover {
        background: var(--button-hover) !important;
        color: #FFFFFF !important;
        border-color: var(--accent) !important;
    }

    div[data-testid="stForm"] button:disabled,
    div[data-testid="stForm"] .stButton > button:disabled,
    div[data-testid="stForm"] button:disabled:hover {
        background: var(--disabled-bg) !important;
        color: var(--disabled-text) !important;
        border-color: var(--border) !important;
        opacity: 1 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    [data-testid="stButton"] button[kind="secondary"] {
        min-height: 2.35rem;
    }
    
    /* Primary button - Ask */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%);
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
        transform: translateY(-2px);
    }
    
    /* Text input - Clean and modern */
    .stTextInput > div > input {
        font-size: 0.98rem;
        padding: 0.55rem 0.35rem;
        border-radius: 0;
        border: none;
        transition: all 0.2s ease;
        background: var(--input-bg);
        color: var(--text-main);
        box-shadow: none;
    }

    div[data-testid="stForm"] input,
    div[data-testid="stTextInput"] input,
    .stTextInput input {
        background: transparent !important;
        color: var(--text-main) !important;
        border: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stForm"] input:focus,
    div[data-testid="stTextInput"] input:focus,
    .stTextInput input:focus {
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stForm"] input::placeholder,
    div[data-testid="stTextInput"] input::placeholder,
    .stTextInput input::placeholder {
        color: var(--text-muted) !important;
        opacity: 1 !important;
    }
    
    .stTextInput > div > input:focus {
        border-color: transparent;
        box-shadow: none;
    }
    
    .stTextInput > div > input::placeholder {
        color: var(--text-muted);
        font-size: 1rem;
    }
    
    /* Form styling */
    .stForm {
        background-color: var(--composer);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 0.55rem 0.7rem;
        box-shadow: 0 10px 28px rgba(17, 24, 39, 0.08);
    }

    div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {
        gap: 0.55rem;
        margin-top: 0;
        align-items: center;
    }

    div[data-testid="stForm"] [data-testid="stTextInput"] {
        margin-bottom: 0;
    }

    div[data-testid="stForm"] [data-testid="stTextInput"] > div {
        border: none !important;
        box-shadow: none !important;
    }

    div[data-testid="stForm"] [data-testid="stTextInput"] div,
    div[data-testid="stForm"] [data-testid="stTextInput"] div:focus,
    div[data-testid="stForm"] [data-testid="stTextInput"] div:focus-within {
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stForm"] [data-testid="column"] {
        display: flex;
        align-items: center;
    }

    div[data-testid="stForm"] [data-testid="column"] > div {
        width: 100%;
    }

    div[data-testid="stForm"] [data-baseweb="input"],
    div[data-testid="stForm"] [data-baseweb="input"]:focus,
    div[data-testid="stForm"] [data-baseweb="input"]:focus-within {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stForm"] [data-baseweb="input"] > div,
    div[data-testid="stForm"] [data-baseweb="input"] > div:focus-within,
    div[data-testid="stForm"] [data-baseweb="input"] div[role="presentation"] {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }

    div[data-testid="stForm"] input[aria-invalid],
    div[data-testid="stForm"] input[aria-invalid="true"],
    div[data-testid="stForm"] input[aria-invalid="false"],
    div[data-testid="stForm"] input:focus-visible {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .main-header {
            font-size: 2.1rem;
        }

        .sub-header {
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }

        .chat-scroll {
            height: calc(100vh - 31rem);
            min-height: 120px;
        }

        .scroll-latest-button {
            bottom: 8.7rem;
            font-size: 0.8rem;
            padding: 0.45rem 0.75rem;
        }
    }

    /* Divider */
    .stDivider {
        margin: 2rem 0;
    }
    
    /* Keep Streamlit's top taskbar available */
    #MainMenu {visibility: visible;}
    footer {visibility: hidden;}
    header {visibility: visible;}
    
    /* Sidebar */
    .css-1d391kg {
        background: var(--surface-soft);
    }

    .sidebar-brand {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-main);
        margin: 0.25rem 0 1rem 0;
    }

    .rail-label {
        color: var(--text-muted);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin: 1.1rem 0 0.35rem 0;
    }

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Check Authentication
# --------------------------------------------------
if not st.session_state.get('authenticated', False):
    from auth import show_login_page
    show_login_page()
    st.stop()

load_persisted_chat_history()
st.session_state.answer_style = "detailed"

# --------------------------------------------------
# Load AI POD (only for authenticated users)
# --------------------------------------------------
AI_POD_CACHE_VERSION = "local-embedding-loader-v16-compact-answer-format"

@st.cache_resource
def load_ai_pod(cache_version: str):
    try:
        ai_pod = AIPodQuerySystem()
        ai_pod.cache_version = cache_version
        return ai_pod
    except FileNotFoundError:
        st.error("Index not found. Please run: python ingest_documents.py")
        raise
    except Exception as e:
        st.error(f"Failed to load AI POD: {e}")
        raise

ai_pod = None

# --------------------------------------------------
# Header (for authenticated users)
# --------------------------------------------------
header_left, new_chat_col, theme_col = st.columns([5, 1, 1])
with new_chat_col:
    if st.button("New Chat", use_container_width=True, key="new_chat_top"):
        start_new_chat(ai_pod)
        st.rerun()

with theme_col:
    next_theme = "light" if st.session_state.theme_mode == "dark" else "dark"
    toggle_label = "Light" if st.session_state.theme_mode == "dark" else "Dark"
    if st.button(toggle_label, use_container_width=True, key="theme_toggle"):
        st.session_state.theme_mode = next_theme
        st.rerun()

st.markdown('<h1 class="main-header">AI POD</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-header">{AIPodConfig.COMPANY_NAME} – Internal AI Assistant</p>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">AI POD</div>', unsafe_allow_html=True)
    user = st.session_state.get("user") or {}
    st.markdown(
        f"""
        <div style="line-height: 1.35; margin: 0.25rem 0 0.75rem 0;">
            <strong>{user.get('name', 'User')}</strong><br>
            <span style="color: var(--text-muted); font-size: 0.82rem;">{user.get('role', 'employee').title()}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Logout", use_container_width=True, key="logout_sidebar"):
        logout()

    if st.button("New Chat", use_container_width=True, type="primary", key="new_chat_sidebar"):
        start_new_chat(ai_pod)
        st.rerun()

    st.session_state.chat_search = st.text_input(
        "Search chats",
        value=st.session_state.chat_search,
        placeholder="Search chats",
        key="chat_search_input",
        label_visibility="collapsed"
    )

    chat_store = load_chat_sessions()
    saved_chats = sorted(
        [
            (chat_id, chat)
            for chat_id, chat in chat_store.get("chats", {}).items()
            if chat.get("messages")
        ],
        key=lambda item: item[1].get("updated_at", ""),
        reverse=True,
    )
    search_term = st.session_state.chat_search.strip().lower()
    if search_term:
        saved_chats = [
            (chat_id, chat)
            for chat_id, chat in saved_chats
            if search_term in (chat.get("title") or "New chat").lower()
        ]
    st.markdown('<div class="rail-label">Recents</div>', unsafe_allow_html=True)
    if saved_chats:
        for chat_id, chat in saved_chats[:10]:
            title = chat.get("title") or "New chat"
            is_current = chat_id == st.session_state.current_chat_id
            label = f"• {title}" if is_current else title
            if st.button(label, use_container_width=True, key=f"chat_session_{chat_id}"):
                switch_chat(chat_id, ai_pod)
                st.rerun()
    else:
        st.caption("No recent chats yet.")

# --------------------------------------------------
# Main Area - Only for authenticated users
# --------------------------------------------------
tab1 = st.container()

with tab1:
    render_chat_history()

    st.markdown('<div class="composer-shell">', unsafe_allow_html=True)

    with st.form(key="ask_form", clear_on_submit=False):
        input_col, ask_col = st.columns([8, 1])

        with input_col:
            question = st.text_input(
                "Ask your question:",
                value=st.session_state.question,
                placeholder="Ask anything about HR, IT, or company policies",
                key="question_input",
                label_visibility="collapsed"
            )

        with ask_col:
            submit_button = st.form_submit_button(
                "Ask",
                type="primary",
                use_container_width=True
            )

    st.markdown("</div>", unsafe_allow_html=True)

    if submit_button and not question.strip():
        st.session_state.question = ""
        st.rerun()

    if submit_button and question.strip() and not AI_POD_AVAILABLE:
        st.error("AI POD is offline. Check the setup and run ingestion if needed.")

    if submit_button and question.strip() and AI_POD_AVAILABLE:
        with st.spinner("Searching policies..."):
            try:
                try:
                    ai_pod = load_ai_pod(AI_POD_CACHE_VERSION)
                except Exception:
                    st.error("AI POD could not load. Check the terminal for details.")
                    st.stop()
                if not st.session_state.get("ai_memory_rehydrated", False):
                    for chat in st.session_state.chat_history[-6:]:
                        if hasattr(ai_pod, "memory"):
                            ai_pod.memory.add(chat.get("question", ""), chat.get("answer", ""))
                    st.session_state.ai_memory_rehydrated = True
                result = get_cached_answer(question, "detailed", ai_pod)
                if result and hasattr(ai_pod, "memory"):
                    ai_pod.memory.add(question, result.get("answer", ""))
                if not result:
                    result = ai_pod.ask(question, answer_style="detailed")
                    save_cached_answer(question, "detailed", ai_pod, result)
                lang = detect_language(question)

                clean_answer = re.sub(r"\[From:\s*.*?\]", "", result["answer"]).strip()
                confidence = result["confidence"]

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": clean_answer,
                    "timestamp": datetime.now().isoformat(),
                    "language": lang,
                    "confidence": confidence,
                    "answer_style": "detailed",
                    "match_type": result.get("match_type", "none"),
                    "response_time": result.get("response_time", 0),
                    "sources": result.get("sources", []),
                    "cached": result.get("cached", False)
                })
                save_persisted_chat_history()
                st.session_state.question = ""
                st.session_state.force_chat_follow = True
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)[:200]}")

# --------------------------------------------------
# Footer - Clean and minimal
# --------------------------------------------------
st.divider()
footer_user = st.session_state.get("user") or {}
st.markdown(f"""
<div style="text-align: center; color: #6B7280; padding: 1rem; font-size: 0.85rem;">
    <strong>AI POD v{AIPodConfig.VERSION}</strong> • {AIPodConfig.COMPANY_NAME}<br>
    <span style="font-family: monospace;">User: {footer_user.get('username', 'guest')} | Session: {st.session_state.session_id}</span>
</div>
""", unsafe_allow_html=True)
