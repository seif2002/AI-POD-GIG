"""
AI POD Configuration - Single Source of Truth
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def load_environment():
    """Load .env values, using python-dotenv when available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ModuleNotFoundError:
        pass

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# Load environment variables
load_environment()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_text(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default

class AIPodConfig:
    """Configuration for AI POD system - Single source of truth"""

    # ── GROQ CONFIGURATION ───────────────────────────────────────────────────
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Multilingual embedding model — supports Arabic + English in the same vector space
    # Replaces all-MiniLM-L6-v2 which was English-only and caused Arabic search to fail
    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    # Groq LLM models (were accidentally commented out — restored here)
    LLM_MODEL_FAST     = "llama-3.1-8b-instant"       # Default for all queries
    LLM_MODEL_ADVANCED = "llama-3.3-70b-versatile"    # Available for complex reasoning

    # ── PROJECT METADATA ─────────────────────────────────────────────────────
    COMPANY_NAME = "GIG EGYPT LIFE TAKAFUL"
    PROJECT_NAME = "AI POD"
    VERSION      = "1.3.0"
    DEPARTMENT   = "Multi-Department"

    # ── FILE PATHS ───────────────────────────────────────────────────────────
    BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
    DOCUMENTS_DIR = os.path.join(BASE_DIR, "HRandIT_documents")
    INDEX_DIR     = os.path.join(BASE_DIR, "indices")
    LOG_DIR       = os.path.join(BASE_DIR, "logs")

    # FAISS index files
    FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
    METADATA_PATH    = os.path.join(INDEX_DIR, "metadata.pkl")
    CHUNKS_PATH      = os.path.join(INDEX_DIR, "chunks.pkl")
    THRESHOLDS_PATH  = os.path.join(INDEX_DIR, "thresholds.pkl")

    # ── EMBEDDING / CHUNKING CONFIG ──────────────────────────────────────────
    CHUNK_SIZE    = 1000   # Characters per chunk
    CHUNK_OVERLAP = 200    # Overlap between chunks (active — bridges context at boundaries)
    BATCH_SIZE    = 32     # Batch size for embedding generation

    # ── SEARCH CONFIG ────────────────────────────────────────────────────────
    DEFAULT_SEARCH_K  = 15   # FAISS candidates retrieved per query
    TOP_K_RESULTS     = 3    # Top chunks sent to the LLM

    # ── SEMANTIC THRESHOLDS (overridden by auto-calibration at ingest time) ──
    HIGH_THRESHOLD   = 0.45
    MEDIUM_THRESHOLD = 0.30
    LOW_THRESHOLD    = 0.20

    # ── SECURITY CONFIG ──────────────────────────────────────────────────────
    ALLOWED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md']
    MAX_FILE_SIZE      = 10 * 1024 * 1024   # 10 MB

    # ── TOKEN LIMITS ─────────────────────────────────────────────────────────
    MAX_TOKENS_PER_QUERY = 4000
    MAX_CHUNKS_PER_QUERY = 5

    # ── CONVERSATION STYLE CONFIGURATION ────────────────────────────────────
    CONVERSATION_MAX_TURNS = _env_int("AI_POD_CONVERSATION_MAX_TURNS", 10)
    CONVERSATION_MAX_CHARS = _env_int("AI_POD_CONVERSATION_MAX_CHARS", 5000)
    GENERAL_CHAT_MAX_TOKENS = _env_int("AI_POD_GENERAL_CHAT_MAX_TOKENS", 650)
    GENERAL_CHAT_TEMPERATURE = _env_float("AI_POD_GENERAL_CHAT_TEMPERATURE", 0.75)
    CHAT_TITLE_MAX_CHARS = _env_int("AI_POD_CHAT_TITLE_MAX_CHARS", 56)

    CHAT_PERSONA_EN = _env_text(
        "AI_POD_CHAT_PERSONA_EN",
        "You speak like a warm, capable colleague: natural, concise, and present."
    )
    CHAT_PERSONA_AR = _env_text(
        "AI_POD_CHAT_PERSONA_AR",
        "تحدث كمساعد ودود وقريب من المستخدم: طبيعي، مختصر، وحاضر في الحوار."
    )
    CHAT_STYLE_RULES_EN = _env_text(
        "AI_POD_CHAT_STYLE_RULES_EN",
        "For casual conversation, acknowledge the user's tone before answering. "
        "Ask one short follow-up question when the message is vague or emotional. "
        "Do not over-explain, do not sound corporate unless the topic requires it, "
        "and never say you are just an AI language model."
    )
    CHAT_STYLE_RULES_AR = _env_text(
        "AI_POD_CHAT_STYLE_RULES_AR",
        "في المحادثات العادية، تفاعل مع نبرة المستخدم قبل الإجابة. "
        "اسأل سؤال متابعة قصير واحد عندما تكون الرسالة غير واضحة أو عاطفية. "
        "لا تكثر الشرح، ولا تستخدم أسلوباً رسمياً إلا إذا احتاج الموضوع لذلك."
    )

    # ── DIRECTORY CREATION ───────────────────────────────────────────────────
    @classmethod
    def create_directories(cls):
        """Create all required directories if they don't exist."""
        os.makedirs(cls.DOCUMENTS_DIR, exist_ok=True)
        os.makedirs(cls.INDEX_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        return True

    @classmethod
    def validate(cls) -> list:
        """
        Check for common misconfigurations.
        Returns a list of warning strings (empty = all good).
        """
        warnings = []
        if not cls.GROQ_API_KEY:
            warnings.append(
                "GROQ_API_KEY is not set — add it to your .env file. "
                "Running in basic mode (raw document excerpts, no LLM synthesis)."
            )
        if not os.path.exists(cls.FAISS_INDEX_PATH):
            warnings.append(
                "FAISS index not found — run: python ingest_documents.py"
            )
        return warnings


# Create directories and print any warnings on import
AIPodConfig.create_directories()

_warnings = AIPodConfig.validate()
for _w in _warnings:
    print(f"  [config] {_w}")
