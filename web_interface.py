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
from datetime import datetime, timedelta
import hashlib
import re
import streamlit.components.v1 as components
from collections import Counter
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

if st.session_state.pop("clear_question_input", False):
    st.session_state.question = ""
    st.session_state.question_input = ""

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


def _parse_timestamp(value):
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except (TypeError, ValueError):
        return None


def _normalize_question_for_stats(question: str) -> str:
    return re.sub(r"\s+", " ", (question or "").strip().lower())


def _load_user_directory() -> dict:
    users = _load_json("users.json", {})
    return users if isinstance(users, dict) else {}


def _collect_chat_messages_for_dashboard(admin_user: dict) -> list:
    """Collect saved chat messages, scoped to the admin department when possible."""
    _ensure_cache_dirs()
    users = _load_user_directory()
    admin_department = (admin_user or {}).get("department")
    messages = []

    for filename in os.listdir(CHAT_SESSIONS_DIR):
        if not filename.endswith(".json"):
            continue
        username = os.path.splitext(filename)[0]
        user_record = users.get(username, {})
        user_department = user_record.get("department")
        if admin_department and user_department and user_department != admin_department:
            continue

        store = _load_json(os.path.join(CHAT_SESSIONS_DIR, filename), _empty_sessions_store())
        chats = store.get("chats", {}) if isinstance(store, dict) else {}
        for chat in chats.values():
            for message in chat.get("messages", []):
                if not isinstance(message, dict):
                    continue
                item = message.copy()
                item["_username"] = username
                item["_department"] = user_department or "Unknown"
                messages.append(item)

    return messages


def _is_no_answer_message(message: dict) -> bool:
    match_type = str(message.get("match_type", "")).lower()
    answer = str(message.get("answer", "")).lower()
    return (
        match_type in {"none", "error"}
        or "could not find" in answer
        or "not enough information" in answer
        or "لم أتمكن من العثور" in answer
    )


LUCIDE_PATHS = {
    "search": '<circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path>',
    "send": '<path d="m22 2-7 20-4-9-9-4Z"></path><path d="M22 2 11 13"></path>',
    "plus-circle": '<circle cx="12" cy="12" r="10"></circle><path d="M8 12h8"></path><path d="M12 8v8"></path>',
    "log-out": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" x2="9" y1="12" y2="12"></line>',
    "layout-dashboard": '<rect width="7" height="9" x="3" y="3" rx="1"></rect><rect width="7" height="5" x="14" y="3" rx="1"></rect><rect width="7" height="9" x="14" y="12" rx="1"></rect><rect width="7" height="5" x="3" y="16" rx="1"></rect>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M3 5v14c0 1.7 4 3 9 3s9-1.3 9-3V5"></path><path d="M3 12c0 1.7 4 3 9 3s9-1.3 9-3"></path>',
    "circle-alert": '<circle cx="12" cy="12" r="10"></circle><line x1="12" x2="12" y1="8" y2="12"></line><line x1="12" x2="12.01" y1="16" y2="16"></line>',
    "file-text": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"></path><path d="M14 2v4a2 2 0 0 0 2 2h4"></path><path d="M10 9H8"></path><path d="M16 13H8"></path><path d="M16 17H8"></path>',
    "clock": '<circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline>',
}


def lucide_icon(name: str, class_name: str = "lucide-inline") -> str:
    path = LUCIDE_PATHS.get(name, "")
    return (
        f'<svg class="{class_name}" xmlns="http://www.w3.org/2000/svg" width="18" height="18" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{path}</svg>'
    )


def dashboard_metric_card(label: str, value, icon_name: str, detail: str = "", tone: str = "default") -> str:
    return (
        f'<div class="dashboard-metric-card dashboard-tone-{html.escape(tone)}">'
        f'<div class="dashboard-metric-icon">{lucide_icon(icon_name, "dashboard-card-icon")}</div>'
        f'<div class="dashboard-metric-label">{html.escape(label)}</div>'
        f'<div class="dashboard-metric-value">{html.escape(str(value))}</div>'
        f'<div class="dashboard-metric-detail">{html.escape(detail)}</div>'
        '</div>'
    )


def _dashboard_status(message: dict) -> str:
    match_type = str(message.get("match_type", "none") or "none").lower()
    try:
        confidence = float(message.get("confidence", 0) or 0)
    except (TypeError, ValueError):
        confidence = 0
    if match_type == "error":
        return "error"
    if _is_no_answer_message(message):
        return "none"
    if confidence and confidence < 0.45:
        return "low confidence"
    return "matched"


def _dashboard_status_badge(status: str) -> str:
    normalized = str(status or "none").strip().lower()
    label = {
        "none": "No answer",
        "error": "Error",
        "matched": "Matched",
        "low confidence": "Low confidence",
    }.get(normalized, normalized.title() or "None")
    badge_class = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-") or "none"
    return f'<span class="dashboard-status-badge badge-{badge_class}">{html.escape(label)}</span>'


def _dashboard_truncate(value, limit: int = 96) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return html.escape(text)
    return f'<span title="{html.escape(text)}">{html.escape(text[:limit - 1])}...</span>'


def _format_dashboard_time(value) -> str:
    ts = _parse_timestamp(value)
    return ts.strftime("%Y-%m-%d %H:%M") if ts else str(value or "")


def _format_dashboard_confidence(value) -> str:
    try:
        return f"{float(value or 0) * 100:.0f}%"
    except (TypeError, ValueError):
        return "0%"


def _dashboard_panel_title(title: str, icon_name: str, subtitle: str = ""):
    st.markdown(
        (
            '<div class="dashboard-panel-heading">'
            f'<div class="dashboard-panel-title">{lucide_icon(icon_name, "dashboard-section-icon")}{html.escape(title)}</div>'
            f'<div class="dashboard-panel-subtitle">{html.escape(subtitle)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def _render_dashboard_table(rows: list, columns: list, empty_message: str = "No data available."):
    if not rows:
        st.markdown(f'<div class="dashboard-empty">{html.escape(empty_message)}</div>', unsafe_allow_html=True)
        return
    header = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if column in {"Status", "Match"}:
                rendered = _dashboard_status_badge(str(value).lower())
            elif column in {"Question", "Latest Question"}:
                rendered = _dashboard_truncate(value)
            elif column == "Action":
                rendered = f'<span class="dashboard-action-pill">{html.escape(str(value or "Review"))}</span>'
            else:
                rendered = html.escape(str(value))
            cells.append(f"<td>{rendered}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        f"""
        <div class="dashboard-table-wrap">
            <table class="dashboard-table">
                <thead><tr>{header}</tr></thead>
                <tbody>{''.join(body_rows)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dashboard_bar_chart(items: list, empty_message: str = "No trend data available."):
    if not items:
        st.markdown(f'<div class="dashboard-empty">{html.escape(empty_message)}</div>', unsafe_allow_html=True)
        return
    max_value = max(value for _, value in items) or 1
    bars = []
    for label, value in items:
        height = max(8, int((value / max_value) * 130)) if value else 8
        bars.append(
            '<div class="dashboard-bar-item">'
            f'<div class="dashboard-bar-value">{html.escape(str(value))}</div>'
            f'<div class="dashboard-bar" style="height:{height}px"></div>'
            f'<div class="dashboard-bar-label">{html.escape(str(label))}</div>'
            '</div>'
        )
    st.markdown(f'<div class="dashboard-bar-chart">{"".join(bars)}</div>', unsafe_allow_html=True)


def _render_dashboard_donut(segments: list):
    total = sum(max(0, value) for _, value, _ in segments)
    if total <= 0:
        st.markdown('<div class="dashboard-empty">No status data available.</div>', unsafe_allow_html=True)
        return
    current = 0
    gradient_parts = []
    legend = []
    for label, value, color in segments:
        percentage = max(0, value) / total * 100
        gradient_parts.append(f"{color} {current:.2f}% {current + percentage:.2f}%")
        current += percentage
        legend.append(
            '<div class="dashboard-donut-legend-item">'
            f'<span style="background:{color}"></span>'
            f'<strong>{html.escape(label)}</strong>'
            f'<em>{value}</em>'
            '</div>'
        )
    st.markdown(
        f"""
        <div class="dashboard-donut-wrap">
            <div class="dashboard-donut" style="background: conic-gradient({', '.join(gradient_parts)});">
                <div><strong>{total}</strong><span>Total</span></div>
            </div>
            <div class="dashboard-donut-legend">{''.join(legend)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_admin_dashboard():
    components.html(
        """
        <script>
        (() => {
            const root = window.parent.document;
            const updateDashboardWidth = () => {
                const activeTab = root.querySelector('[data-testid="stTabs"] button[aria-selected="true"]');
                const activeLabel = (activeTab?.textContent || "").trim();
                root.body.classList.toggle("ai-pod-dashboard-active", activeLabel.includes("Admin Dashboard"));
            };
            updateDashboardWidth();
            window.parent.setTimeout(updateDashboardWidth, 150);
            window.parent.setTimeout(updateDashboardWidth, 600);
            if (!window.parent.aiPodDashboardTabObserver) {
                window.parent.aiPodDashboardTabObserver = new MutationObserver(updateDashboardWidth);
                window.parent.aiPodDashboardTabObserver.observe(root.body, {
                    attributes: true,
                    childList: true,
                    subtree: true,
                    attributeFilter: ["aria-selected", "class"]
                });
            }
        })();
        </script>
        """,
        height=0,
    )

    admin_user = st.session_state.get("user") or {}
    dated_messages_all = []
    for message in _collect_chat_messages_for_dashboard(admin_user):
        timestamp = _parse_timestamp(message.get("timestamp"))
        if timestamp:
            dated_messages_all.append((timestamp, message))
    dated_messages_all.sort(key=lambda item: item[0], reverse=True)

    now = datetime.now()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=today_start.weekday())
    month_start = today_start.replace(day=1)

    users = sorted({message.get("_username", "Unknown") for _, message in dated_messages_all if message.get("_username")})
    departments = sorted({message.get("_department", "Unknown") for _, message in dated_messages_all if message.get("_department")})
    statuses = sorted({_dashboard_status(message) for _, message in dated_messages_all})
    languages = sorted({str(message.get("language", "") or detect_language(message.get("question", ""))).upper() for _, message in dated_messages_all})

    st.markdown(
        """
        <div class="dashboard-hero">
            <div>
                <div class="dashboard-eyebrow">Admin Analytics</div>
                <h2>Admin Dashboard</h2>
                <p>Monitor question volume, answer quality, cache performance, and knowledge-base gaps.</p>
            </div>
            <div class="dashboard-scope-card">
                <span>Department Scope</span>
                <strong>{scope}</strong>
            </div>
        </div>
        """.format(scope=html.escape(admin_user.get("department", "All available departments"))),
        unsafe_allow_html=True,
    )

    filter_cols = st.columns([1.15, 1, 1, 1, 1])
    date_range = filter_cols[0].selectbox(
        "Date range",
        ["Last 7 days", "Last 30 days", "Today", "This month", "All time", "Custom"],
        key="dashboard_date_range",
    )
    selected_department = filter_cols[1].selectbox(
        "Department",
        ["All"] + departments,
        key="dashboard_department_filter",
    )
    selected_user = filter_cols[2].selectbox(
        "User",
        ["All"] + users,
        key="dashboard_user_filter",
    )
    selected_status = filter_cols[3].selectbox(
        "Match status",
        ["All"] + [status.title() for status in statuses],
        key="dashboard_status_filter",
    )
    selected_language = filter_cols[4].selectbox(
        "Language",
        ["All"] + languages,
        key="dashboard_language_filter",
    )

    custom_start = custom_end = None
    if date_range == "Custom":
        custom_cols = st.columns([1, 1, 3])
        custom_start = custom_cols[0].date_input("Start date", value=(today_start - timedelta(days=30)).date(), key="dashboard_custom_start")
        custom_end = custom_cols[1].date_input("End date", value=now.date(), key="dashboard_custom_end")

    if date_range == "Today":
        start_at = today_start
    elif date_range == "Last 7 days":
        start_at = today_start - timedelta(days=6)
    elif date_range == "Last 30 days":
        start_at = today_start - timedelta(days=29)
    elif date_range == "This month":
        start_at = month_start
    elif date_range == "Custom" and custom_start and custom_end:
        start_at = datetime.combine(custom_start, datetime.min.time())
        end_at = datetime.combine(custom_end, datetime.max.time())
    else:
        start_at = None
        end_at = None
    if date_range != "Custom":
        end_at = now

    dated_messages = []
    for ts, message in dated_messages_all:
        if start_at and ts < start_at:
            continue
        if end_at and ts > end_at:
            continue
        if selected_department != "All" and message.get("_department", "Unknown") != selected_department:
            continue
        if selected_user != "All" and message.get("_username", "Unknown") != selected_user:
            continue
        if selected_status != "All" and _dashboard_status(message) != selected_status.lower():
            continue
        language = str(message.get("language", "") or detect_language(message.get("question", ""))).upper()
        if selected_language != "All" and language != selected_language:
            continue
        dated_messages.append((ts, message))

    today_count = sum(1 for ts, _ in dated_messages if ts >= today_start)
    week_count = sum(1 for ts, _ in dated_messages if ts >= week_start)
    month_count = sum(1 for ts, _ in dated_messages if ts >= month_start)
    no_answer_messages = [message for _, message in dated_messages if _is_no_answer_message(message)]
    cache_hits = sum(1 for _, message in dated_messages if bool(message.get("cached")))
    cache_hit_rate = (cache_hits / len(dated_messages) * 100) if dated_messages else 0
    unique_users = len({message.get("_username") for _, message in dated_messages if message.get("_username")})
    confidences = []
    for _, message in dated_messages:
        try:
            confidences.append(float(message.get("confidence", 0) or 0))
        except (TypeError, ValueError):
            pass
    average_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0
    previous_week_start = week_start - timedelta(days=7)
    previous_week_count = sum(1 for ts, _ in dated_messages_all if previous_week_start <= ts < week_start)
    if previous_week_count:
        trend_value = ((week_count - previous_week_count) / previous_week_count) * 100
        trend_text = f"{trend_value:+.0f}% vs last week"
    elif week_count:
        trend_text = "New activity this week"
    else:
        trend_text = "No activity this week"

    metric_cols = st.columns(7)
    metric_cols[0].markdown(dashboard_metric_card("Today's questions", today_count, "clock", "Since midnight"), unsafe_allow_html=True)
    metric_cols[1].markdown(dashboard_metric_card("This week", week_count, "clock", trend_text), unsafe_allow_html=True)
    metric_cols[2].markdown(dashboard_metric_card("This month", month_count, "clock", "Current month"), unsafe_allow_html=True)
    metric_cols[3].markdown(dashboard_metric_card("No-answer count", len(no_answer_messages), "circle-alert", "Needs review", "warning"), unsafe_allow_html=True)
    metric_cols[4].markdown(dashboard_metric_card("Cache hit rate", f"{cache_hit_rate:.1f}%", "database", f"{cache_hits} cached replies"), unsafe_allow_html=True)
    metric_cols[5].markdown(dashboard_metric_card("Unique users", unique_users, "search", "Filtered scope"), unsafe_allow_html=True)
    metric_cols[6].markdown(dashboard_metric_card("Avg. confidence", f"{average_confidence:.0f}%", "file-text", "Mean match score"), unsafe_allow_html=True)

    daily_counts = Counter(ts.date() for ts, _ in dated_messages)
    if start_at:
        chart_start = start_at.date()
    elif dated_messages:
        chart_start = min(ts.date() for ts, _ in dated_messages)
    else:
        chart_start = today_start.date()
    chart_end = (end_at or now).date()
    total_days = max(1, min(45, (chart_end - chart_start).days + 1))
    chart_start = chart_end - timedelta(days=total_days - 1)
    trend_items = [
        ((chart_start + timedelta(days=offset)).strftime("%m/%d"), daily_counts.get(chart_start + timedelta(days=offset), 0))
        for offset in range(total_days)
    ]

    status_counts = Counter(_dashboard_status(message) for _, message in dated_messages)
    language_counts = Counter(str(message.get("language", "") or detect_language(message.get("question", ""))).upper() for _, message in dated_messages)
    department_counts = Counter(message.get("_department", "Unknown") for _, message in dated_messages)
    user_counts = Counter(message.get("_username", "Unknown") for _, message in dated_messages)

    top_row_left, top_row_right = st.columns([1.75, 1])
    with top_row_left:
        with st.container(border=True):
            _dashboard_panel_title("Questions Over Time", "clock", "Filtered daily volume")
            _render_dashboard_bar_chart(trend_items)

    with top_row_right:
        with st.container(border=True):
            _dashboard_panel_title("Answer Health", "circle-alert", "Answered vs knowledge gaps")
            _render_dashboard_donut(
                [
                    ("Answered", max(0, len(dated_messages) - len(no_answer_messages)), "#22C55E"),
                    ("No answer", len(no_answer_messages), "#EF4444"),
                    ("Cached", cache_hits, "#3B82F6"),
                ]
            )

    breakdown_left, breakdown_mid, breakdown_right = st.columns(3)
    with breakdown_left:
        with st.container(border=True):
            _dashboard_panel_title("Match Status", "layout-dashboard")
            _render_dashboard_table(
                [{"Status": status, "Count": count} for status, count in status_counts.most_common()],
                ["Status", "Count"],
                "No status data available.",
            )
    with breakdown_mid:
        with st.container(border=True):
            _dashboard_panel_title("Language Split", "file-text")
            _render_dashboard_bar_chart(language_counts.most_common(), "No language data available.")
    with breakdown_right:
        with st.container(border=True):
            _dashboard_panel_title("Top Users", "search")
            _render_dashboard_table(
                [{"User": user, "Questions": count} for user, count in user_counts.most_common(8)],
                ["User", "Questions"],
                "No user activity available.",
            )

    question_counts = {}
    question_display = {}
    for _, message in dated_messages:
        normalized = _normalize_question_for_stats(message.get("question", ""))
        if not normalized:
            continue
        question_counts[normalized] = question_counts.get(normalized, 0) + 1
        question_display.setdefault(normalized, message.get("question", "").strip())

    table_left, table_right = st.columns(2)
    with table_left:
        with st.container(border=True):
            _dashboard_panel_title("Top Questions", "search", "Most common repeated topics")
            top_search = st.text_input("Search top questions", placeholder="Filter questions", key="dashboard_top_question_search")
            top_limit = st.selectbox("Rows", ["Top 10", "Top 25", "All"], key="dashboard_top_question_limit")
            top_limit_count = {"Top 10": 10, "Top 25": 25, "All": None}[top_limit]
            top_questions = sorted(question_counts.items(), key=lambda item: item[1], reverse=True)
            if top_search:
                top_questions = [
                    item for item in top_questions
                    if top_search.lower() in question_display.get(item[0], "").lower()
                ]
            if top_limit_count:
                top_questions = top_questions[:top_limit_count]
            _render_dashboard_table(
                [{"Question": question_display[key], "Count": count} for key, count in top_questions],
                ["Question", "Count"],
                "No question data is available for this scope.",
            )

    with table_right:
        with st.container(border=True):
            _dashboard_panel_title("Questions With No Answer", "circle-alert", "Action queue for knowledge-base gaps")
            no_answer_search = st.text_input("Search unanswered questions", placeholder="Filter unanswered questions", key="dashboard_no_answer_search")
            no_answer_limit = st.selectbox("Queue rows", ["Top 10", "Top 25", "All"], key="dashboard_no_answer_limit")
            no_answer_limit_count = {"Top 10": 10, "Top 25": 25, "All": None}[no_answer_limit]
            recent_no_answers = sorted(no_answer_messages, key=lambda item: item.get("timestamp", ""), reverse=True)
            if no_answer_search:
                recent_no_answers = [
                    item for item in recent_no_answers
                    if no_answer_search.lower() in str(item.get("question", "")).lower()
                ]
            if no_answer_limit_count:
                recent_no_answers = recent_no_answers[:no_answer_limit_count]
            _render_dashboard_table(
                [
                    {
                        "Time": _format_dashboard_time(item.get("timestamp", "")),
                        "User": item.get("_username", ""),
                        "Question": item.get("question", ""),
                        "Match": _dashboard_status(item),
                        "Department": item.get("_department", "Unknown"),
                        "Action": "Review",
                    }
                    for item in recent_no_answers
                ],
                ["Time", "User", "Question", "Match", "Department", "Action"],
                "No unanswered questions found.",
            )

    bottom_left, bottom_right = st.columns([1.5, 1])
    with bottom_left:
        with st.container(border=True):
            _dashboard_panel_title("Recent Activity", "clock", "Latest filtered questions")
            recent_search = st.text_input("Search recent activity", placeholder="Filter latest questions", key="dashboard_recent_search")
            recent_rows = dated_messages[:50]
            if recent_search:
                recent_rows = [
                    (ts, item) for ts, item in recent_rows
                    if recent_search.lower() in str(item.get("question", "")).lower()
                ]
            _render_dashboard_table(
                [
                    {
                        "Time": ts.strftime("%Y-%m-%d %H:%M"),
                        "User": item.get("_username", "Unknown"),
                        "Latest Question": item.get("question", ""),
                        "Status": _dashboard_status(item),
                        "Confidence": _format_dashboard_confidence(item.get("confidence", 0)),
                    }
                    for ts, item in recent_rows[:20]
                ],
                ["Time", "User", "Latest Question", "Status", "Confidence"],
                "No recent activity in this scope.",
            )

    with bottom_right:
        with st.container(border=True):
            _dashboard_panel_title("Department Breakdown", "layout-dashboard", "Questions by department")
            _render_dashboard_bar_chart(department_counts.most_common(8), "No department data available.")

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


TYPE_DELAY_SECONDS = 0.025


def stream_text_chunks(text: str, chunk_size: int = 10):
    """Yield small text chunks for cached/local responses."""
    text = text or ""
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]


def render_chat_history(transient_chat=None):
    """Render the full conversation thread."""
    force_follow = st.session_state.pop("force_chat_follow", False)
    force_attr = "true" if force_follow else "false"
    chat_items = list(st.session_state.chat_history)
    if transient_chat:
        chat_items.append(transient_chat)

    if not chat_items:
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
    for idx, chat in enumerate(chat_items):
        formatted_question = format_inline(clean_display_text(chat.get("question", "")))
        raw_answer = chat.get("answer", "")
        chat_lang = chat.get("language") or detect_language(chat.get("question", ""))
        direction = "rtl" if chat_lang == "ar" else "ltr"
        lang_class = "arabic-answer" if chat_lang == "ar" else "english-answer"
        source_label = format_inline(format_source_names(chat.get("sources", [])))
        time_label = format_inline(format_response_time(chat.get("response_time", 0)))
        source_icon = lucide_icon("file-text", "chat-meta-icon")
        time_icon = lucide_icon("clock", "chat-meta-icon")
        is_streaming = bool(chat.get("streaming"))
        is_conversational = chat.get("match_type") == "conversational" or chat.get("mode") in {"general_chat", "conversational"}
        if raw_answer:
            formatted_answer = format_answer(raw_answer)
            if is_streaming:
                formatted_answer += '<div class="typing-cursor"></div>'
        else:
            formatted_answer = '<div class="typing-indicator"><span></span><span></span><span></span></div>'
        chat_markup.append(f'<div class="chat-row user-row"><div class="user-bubble {lang_class}" dir="{direction}">{formatted_question}</div></div>')
        chat_markup.append(
            '<div class="chat-row assistant-row"><div class="assistant-bubble">'
            f'<div class="answer-box {lang_class}" dir="{direction}">{formatted_answer}</div>'
            + ("" if is_streaming or is_conversational else f'<div class="chat-meta {lang_class}" dir="{direction}"><span>{source_icon} Source: {source_label}</span><span>{time_icon} Time: {time_label}</span></div>')
            + '</div></div>'
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

    [data-testid="stTabs"] button {{
        color: var(--text-main);
    }}

</style>
""", unsafe_allow_html=True)

components.html("""
<script>
(() => {
    const root = window.parent.document;
    root.getElementById("aiPodSidebarToggle")?.remove();
    root.getElementById("aiPodSidebarToggleStyle")?.remove();
    root.body.classList.remove("ai-pod-sidebar-collapsed");
    window.parent.localStorage.removeItem("aiPodSidebarCollapsed");
})();
</script>
""", height=0)

components.html("""
<script>
(() => {
    const root = window.parent.document;
    const icons = {
        "New Chat": '<svg class="ai-pod-button-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M8 12h8"></path><path d="M12 8v8"></path></svg>',
        "Logout": '<svg class="ai-pod-button-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" x2="9" y1="12" y2="12"></line></svg>',
        "Ask": '<svg class="ai-pod-button-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m22 2-7 20-4-9-9-4Z"></path><path d="M22 2 11 13"></path></svg>',
        "Admin Dashboard": '<svg class="ai-pod-button-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="7" height="9" x="3" y="3" rx="1"></rect><rect width="7" height="5" x="14" y="3" rx="1"></rect><rect width="7" height="9" x="14" y="12" rx="1"></rect><rect width="7" height="5" x="3" y="16" rx="1"></rect></svg>'
    };

    const enhance = () => {
        root.querySelectorAll('button, [role="tab"]').forEach((el) => {
            const text = (el.textContent || "").trim();
            const icon = icons[text];
            if (!icon || el.dataset.aiPodIcon === "true") return;
            el.dataset.aiPodIcon = "true";
            const target = el.querySelector("p, span") || el;
            target.insertAdjacentHTML("afterbegin", icon);
            target.style.display = "inline-flex";
            target.style.alignItems = "center";
            target.style.justifyContent = "center";
            target.style.gap = "0.4rem";
        });
    };

    if (!root.getElementById("ai-pod-icon-style")) {
        const style = root.createElement("style");
        style.id = "ai-pod-icon-style";
        style.textContent = ".ai-pod-button-icon{width:16px;height:16px;flex:0 0 auto;}";
        root.head.appendChild(style);
    }

    enhance();
    window.parent.setTimeout(enhance, 250);
    window.parent.setTimeout(enhance, 900);
})();
</script>
""", height=0)

st.markdown("""
<style>
    .block-container {
        max-width: 980px;
        min-height: 100vh;
        overflow: visible;
        padding-top: 1.5rem;
        padding-bottom: 0.8rem;
        color: var(--text-main);
    }

    body.ai-pod-dashboard-active .block-container {
        max-width: 95vw;
        width: 95vw;
    }

    .chat-scroll {
        position: relative;
        height: calc(100vh - 27rem);
        min-height: 220px;
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

    .lucide-inline,
    .chat-meta-icon,
    .dashboard-title-icon,
    .dashboard-section-icon,
    .dashboard-card-icon {
        flex: 0 0 auto;
        vertical-align: -0.2em;
    }

    .dashboard-hero {
        display: flex;
        align-items: stretch;
        justify-content: space-between;
        gap: 1rem;
        margin: 0.5rem 0 1rem 0;
        padding: 1rem 1.1rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: var(--shadow);
    }

    .dashboard-eyebrow {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.25rem;
    }

    .dashboard-hero h2 {
        margin: 0;
        color: var(--text-main);
        font-size: 1.65rem;
        line-height: 1.15;
    }

    .dashboard-hero p {
        margin: 0.35rem 0 0 0;
        color: var(--text-muted);
        font-size: 0.95rem;
    }

    .dashboard-scope-card {
        min-width: 210px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 0.75rem 0.9rem;
        background: var(--surface-soft);
        border: 1px solid var(--border);
        border-radius: 8px;
    }

    .dashboard-scope-card span,
    .dashboard-panel-subtitle,
    .dashboard-metric-detail {
        color: var(--text-muted);
        font-size: 0.78rem;
    }

    .dashboard-scope-card strong {
        color: var(--text-main);
        font-size: 1rem;
        margin-top: 0.15rem;
    }

    .dashboard-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: var(--shadow);
        padding: 1rem;
        margin: 0.9rem 0;
        min-height: 100%;
    }

    .dashboard-panel-compact {
        padding-bottom: 0.5rem;
    }

    .dashboard-panel-heading {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 0.85rem;
    }

    .dashboard-panel-title {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-main);
        font-size: 1rem;
        font-weight: 800;
    }

    .dashboard-title-icon,
    .dashboard-section-icon {
        color: var(--accent);
    }

    .dashboard-metric-card {
        position: relative;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.85rem 0.9rem;
        min-height: 122px;
        box-shadow: var(--shadow);
        overflow: hidden;
    }

    .dashboard-metric-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: var(--accent);
        opacity: 0.9;
    }

    .dashboard-tone-warning::before {
        background: #EF4444;
    }

    .dashboard-metric-icon {
        width: 32px;
        height: 32px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: var(--surface-soft);
        color: var(--accent);
        margin-bottom: 0.65rem;
    }

    .dashboard-metric-label {
        color: var(--text-muted);
        font-size: 0.78rem;
        line-height: 1.25;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .dashboard-card-icon {
        width: 16px;
        height: 16px;
    }

    .dashboard-metric-value {
        color: var(--text-main);
        font-size: 1.7rem;
        line-height: 1.1;
        font-weight: 800;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.85rem 1rem;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-main) !important;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        opacity: 0.88;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
        background: var(--surface) !important;
    }

    div[data-testid="stDataFrame"] * {
        color: var(--text-main) !important;
        border-color: var(--border) !important;
    }

    div[data-testid="stDataFrame"] [role="grid"],
    div[data-testid="stDataFrame"] [role="row"],
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] [role="gridcell"] {
        background: var(--surface) !important;
    }

    div[data-testid="stDataFrame"] [role="columnheader"] {
        background: var(--surface-soft) !important;
        color: var(--text-muted) !important;
        font-weight: 700 !important;
    }

    div[data-testid="stDataFrame"] canvas,
    div[data-testid="stDataFrame"] .glideDataEditor {
        background: var(--surface) !important;
    }

    .dashboard-table-wrap {
        width: 100%;
        max-height: 360px;
        overflow: auto;
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface);
        margin: 0.5rem 0 0 0;
    }

    .dashboard-table {
        width: 100%;
        border-collapse: collapse;
        color: var(--text-main);
        font-size: 0.92rem;
    }

    .dashboard-table th,
    .dashboard-table td {
        padding: 0.7rem 0.8rem;
        border-bottom: 1px solid var(--border);
        border-right: 1px solid var(--border);
        vertical-align: top;
        overflow-wrap: anywhere;
    }

    .dashboard-table th {
        position: sticky;
        top: 0;
        z-index: 1;
        background: var(--surface-soft);
        color: var(--text-muted);
        text-align: left;
        font-weight: 700;
    }

    .dashboard-table td {
        background: var(--surface);
    }

    .dashboard-table tr:last-child td {
        border-bottom: none;
    }

    .dashboard-empty {
        color: var(--text-muted);
        background: var(--surface-soft);
        border: 1px dashed var(--border);
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.9rem;
    }

    .dashboard-status-badge,
    .dashboard-action-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        white-space: nowrap;
        border-radius: 999px;
        padding: 0.22rem 0.55rem;
        font-size: 0.76rem;
        font-weight: 800;
        border: 1px solid transparent;
    }

    .badge-matched {
        background: rgba(34, 197, 94, 0.14);
        color: #16A34A;
        border-color: rgba(34, 197, 94, 0.26);
    }

    .badge-none {
        background: rgba(100, 116, 139, 0.14);
        color: var(--text-muted);
        border-color: rgba(100, 116, 139, 0.22);
    }

    .badge-error {
        background: rgba(239, 68, 68, 0.14);
        color: #EF4444;
        border-color: rgba(239, 68, 68, 0.28);
    }

    .badge-low-confidence {
        background: rgba(245, 158, 11, 0.16);
        color: #D97706;
        border-color: rgba(245, 158, 11, 0.3);
    }

    .dashboard-action-pill {
        background: rgba(37, 99, 235, 0.12);
        color: var(--accent);
        border-color: rgba(37, 99, 235, 0.24);
    }

    .dashboard-bar-chart {
        min-height: 190px;
        display: flex;
        align-items: flex-end;
        gap: 0.45rem;
        padding: 0.65rem 0.25rem 0.25rem 0.25rem;
        overflow-x: auto;
    }

    .dashboard-bar-item {
        min-width: 34px;
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-end;
        gap: 0.25rem;
    }

    .dashboard-bar {
        width: 100%;
        max-width: 42px;
        border-radius: 6px 6px 2px 2px;
        background: linear-gradient(180deg, var(--accent), #1D4ED8);
        min-height: 8px;
    }

    .dashboard-bar-value {
        color: var(--text-muted);
        font-size: 0.72rem;
        font-weight: 800;
    }

    .dashboard-bar-label {
        color: var(--text-muted);
        font-size: 0.72rem;
        white-space: nowrap;
    }

    .dashboard-donut-wrap {
        display: grid;
        grid-template-columns: minmax(150px, 210px) 1fr;
        align-items: center;
        gap: 1rem;
        min-height: 220px;
    }

    .dashboard-donut {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        margin: 0 auto;
        box-shadow: inset 0 0 0 1px var(--border);
    }

    .dashboard-donut > div {
        width: 108px;
        height: 108px;
        border-radius: 50%;
        display: grid;
        place-items: center;
        background: var(--surface);
        border: 1px solid var(--border);
        text-align: center;
    }

    .dashboard-donut strong {
        color: var(--text-main);
        font-size: 1.65rem;
        line-height: 1;
    }

    .dashboard-donut span {
        color: var(--text-muted);
        font-size: 0.75rem;
        display: block;
    }

    .dashboard-donut-legend {
        display: grid;
        gap: 0.55rem;
    }

    .dashboard-donut-legend-item {
        display: grid;
        grid-template-columns: 12px 1fr auto;
        align-items: center;
        gap: 0.5rem;
        color: var(--text-main);
        font-size: 0.86rem;
    }

    .dashboard-donut-legend-item span {
        width: 12px;
        height: 12px;
        border-radius: 999px;
    }

    .dashboard-donut-legend-item em {
        color: var(--text-muted);
        font-style: normal;
        font-weight: 800;
    }

    body.ai-pod-dashboard-active [data-testid="stTabs"] button {
        padding-top: 0.7rem;
        padding-bottom: 0.7rem;
        font-weight: 800;
    }

    body.ai-pod-dashboard-active [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
    }

    @media (max-width: 1100px) {
        body.ai-pod-dashboard-active .block-container {
            max-width: 96vw;
            width: 96vw;
        }

        .dashboard-hero,
        .dashboard-donut-wrap {
            grid-template-columns: 1fr;
            display: grid;
        }

        .dashboard-scope-card {
            min-width: 0;
        }

        .dashboard-metric-card {
            min-height: 112px;
        }
    }

    @media (max-width: 700px) {
        body.ai-pod-dashboard-active .block-container {
            max-width: 100%;
            width: 100%;
        }

        .dashboard-hero,
        .dashboard-panel {
            padding: 0.85rem;
        }

        .dashboard-hero h2 {
            font-size: 1.35rem;
        }

        .dashboard-panel-heading {
            display: block;
        }

        .dashboard-panel-subtitle {
            margin-top: 0.25rem;
        }

        .dashboard-table {
            font-size: 0.82rem;
        }

        .dashboard-table th,
        .dashboard-table td {
            padding: 0.55rem 0.6rem;
        }

        .dashboard-donut {
            width: 150px;
            height: 150px;
        }

        .dashboard-donut > div {
            width: 92px;
            height: 92px;
        }
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

    .typing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        min-height: 1.5rem;
        padding: 0.25rem 0;
    }

    .typing-indicator span {
        width: 0.42rem;
        height: 0.42rem;
        border-radius: 999px;
        background: var(--text-muted);
        animation: typingPulse 1.2s infinite ease-in-out;
    }

    .typing-indicator span:nth-child(2) {
        animation-delay: 0.15s;
    }

    .typing-indicator span:nth-child(3) {
        animation-delay: 0.3s;
    }

    .typing-cursor {
        display: inline-block;
        width: 0.48rem;
        height: 1rem;
        margin-left: 0.15rem;
        border-radius: 999px;
        background: var(--accent);
        animation: typingBlink 0.9s infinite;
        vertical-align: -0.15rem;
    }

    .arabic-answer .typing-cursor {
        margin-left: 0;
        margin-right: 0.15rem;
    }

    @keyframes typingPulse {
        0%, 80%, 100% { opacity: 0.35; transform: translateY(0); }
        40% { opacity: 1; transform: translateY(-2px); }
    }

    @keyframes typingBlink {
        0%, 45% { opacity: 1; }
        46%, 100% { opacity: 0; }
    }

    .chat-meta {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin-top: 0.65rem;
        border-top: 1px solid var(--border);
        padding-top: 0.45rem;
        line-height: 1.4;
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem 0.8rem;
        align-items: center;
    }

    .chat-meta span {
        display: inline-flex;
        align-items: center;
        gap: 0.28rem;
    }

    .chat-meta-icon {
        width: 14px;
        height: 14px;
        color: var(--accent);
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
            height: calc(100vh - 24rem);
            min-height: 180px;
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
if check_permission(["admin"]):
    tab1, dashboard_tab = st.tabs(["Chat", "Admin Dashboard"])
else:
    tab1 = st.container()

with tab1:
    history_placeholder = st.empty()
    with history_placeholder.container():
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
                st.session_state.question = ""
                st.session_state.clear_question_input = True
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

                lang = detect_language(question)
                streamed_answer = ""
                result = None
                used_cache = False
                last_render = [0.0]

                def redraw_stream(force=False):
                    now = time.time()
                    if not force and now - last_render[0] < 0.05:
                        return
                    last_render[0] = now
                    transient_chat = {
                        "question": question,
                        "answer": streamed_answer,
                        "timestamp": datetime.now().isoformat(),
                        "language": lang,
                        "confidence": 0,
                        "answer_style": "detailed",
                        "match_type": "streaming",
                        "response_time": 0,
                        "sources": [],
                        "streaming": True,
                    }
                    st.session_state.force_chat_follow = True
                    with history_placeholder.container():
                        render_chat_history(transient_chat=transient_chat)

                redraw_stream(force=True)
                result = get_cached_answer(question, "detailed", ai_pod)
                if result:
                    used_cache = True
                    for delta in stream_text_chunks(result.get("answer", "")):
                        streamed_answer += delta
                        redraw_stream(force=True)
                        time.sleep(TYPE_DELAY_SECONDS)
                    if hasattr(ai_pod, "memory"):
                        ai_pod.memory.add(question, result.get("answer", ""))
                elif hasattr(ai_pod, "ask_stream"):
                    for event in ai_pod.ask_stream(question, answer_style="detailed"):
                        if not isinstance(event, dict):
                            continue
                        if event.get("type") == "delta":
                            for delta in stream_text_chunks(event.get("text", ""), chunk_size=8):
                                streamed_answer += delta
                                redraw_stream(force=True)
                                time.sleep(TYPE_DELAY_SECONDS)
                        elif event.get("type") == "done":
                            result = event.get("result")
                    if not result:
                        raise RuntimeError("Streaming ended without a final response.")
                    streamed_answer = result.get("answer", streamed_answer)
                    save_cached_answer(question, "detailed", ai_pod, result)
                else:
                    result = ai_pod.ask(question, answer_style="detailed")
                    full_answer = result.get("answer", "")
                    streamed_answer = ""
                    for delta in stream_text_chunks(full_answer):
                        streamed_answer += delta
                        redraw_stream(force=True)
                        time.sleep(TYPE_DELAY_SECONDS)
                    save_cached_answer(question, "detailed", ai_pod, result)

                redraw_stream(force=True)

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
                    "cached": result.get("cached", used_cache)
                })
                save_persisted_chat_history()
                st.session_state.question = ""
                st.session_state.force_chat_follow = True
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)[:200]}")

if check_permission(["admin"]):
    with dashboard_tab:
        render_admin_dashboard()
