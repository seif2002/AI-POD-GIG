"""
AI POD Query System
- Friendly conversational responses for general questions
- Accurate, source-grounded answers for policy questions
- Arabic + English support
- Conversation memory
"""

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import re
import time
from typing import Dict, List
from config import AIPodConfig


# -------------------------------------------------------
# Language Detection
# -------------------------------------------------------

def detect_language(text: str) -> str:
    """Return the dominant user language: 'ar' or 'en'."""
    text = text or ""
    arabic_chars = re.findall(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]', text)
    english_chars = re.findall(r'[A-Za-z]', text)
    if len(arabic_chars) > len(english_chars):
        return "ar"
    if len(english_chars) > len(arabic_chars):
        return "en"
    first_sentence = re.split(r"[.!؟?\n]", text, maxsplit=1)[0]
    return "ar" if re.search(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]', first_sentence) else "en"


def wants_bilingual_response(text: str) -> bool:
    text = text or ""
    bilingual_patterns = [
        r"\bboth\s+(arabic\s+and\s+english|english\s+and\s+arabic|languages)\b",
        r"\bin\s+(arabic\s+and\s+english|english\s+and\s+arabic)\b",
        r"\banswer\s+.*\b(arabic\s+and\s+english|english\s+and\s+arabic)\b",
        r"بالعربية\s+والإنجليزية",
        r"بالإنجليزية\s+والعربية",
        r"باللغتين",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in bilingual_patterns)


# -------------------------------------------------------
# Conversational intent detection
# -------------------------------------------------------

GREETING_PATTERNS = re.compile(
    r'^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|greetings|'
    r'مرحبا|أهلا|السلام|صباح|مساء|هلا)\b',
    re.IGNORECASE
)

ABOUT_BOT_PATTERNS = re.compile(
    r'(what (can|will|do) (you|this|the|ai).{0,20}(do|help|assist)|'
    r'how (can|do) (you|this|the|ai).{0,20}(help|work|assist)|'
    r'what are you|who are you|tell me about yourself|'
    r'chatbot will help|what is ai pod|what does ai pod|'
    r'ماذا تفعل|كيف تساعد|ما هو|من أنت|عن نفسك)',
    re.IGNORECASE
)

POLICY_QUERY_PATTERNS = re.compile(
    r'(\b(policy|procedure|hr|human resources|it|employee|manager|approval|portal|'
    r'annual leave|sick leave|leave days|days off|maternity|paternity|adoption|'
    r'notice period|bonus|salary|benefit|attendance|disciplinary|performance|'
    r'password|remote work|work from home|confidential|insurance|vacation)\b|'
    r'سياسة|إجراء|اجراء|الموارد البشرية|تقنية المعلومات|تكنولوجيا المعلومات|'
    r'موظف|مدير|موافقة|بوابة|إجاز|اجاز|سنوية|مرضية|أمومة|امومة|أبوة|ابوة|'
    r'تبني|فترة الإخطار|اخطار|مكافأة|راتب|حضور|تأديب|أداء|كلمة المرور|'
    r'العمل عن بعد|سرية|تأمين)',
    re.IGNORECASE
)

GENERAL_CHAT_PATTERNS = re.compile(
    r'(\b(how are you|how\'?s it going|thanks|thank you|thx|lol|haha|bye|goodbye|'
    r'capital of|weather|write (an? )?email|draft (an? )?email|help me write|'
    r'translate|summarize|explain|what do you think|opinion|joke|recipe|'
    r'calculate|math|code|python|story|poem)\b|'
    r'شكرا|شكرًا|عامل ايه|كيف حالك|وداعا|اكتب.*ايميل|اكتب.*بريد|'
    r'ترجم|لخص|اشرح|نكتة|رأيك|طقس|عاصمة)',
    re.IGNORECASE
)


def classify_intent(question: str) -> str:
    """Classify question as: greeting | about_bot | general_chat | policy_query."""
    if GREETING_PATTERNS.match(question):
        return "greeting"
    if ABOUT_BOT_PATTERNS.search(question):
        return "about_bot"
    if POLICY_QUERY_PATTERNS.search(question):
        return "policy_query"
    if GENERAL_CHAT_PATTERNS.search(question):
        return "general_chat"
    return "policy_query"


# Quick classifier sanity checks:
# - "what's the capital of France" -> general_chat
# - "can you help me write an email" -> general_chat
# - "what's the weather like" -> general_chat
# - "lol thanks" -> general_chat
# - "how many sick days do I get" -> policy_query


def _terms(text: str) -> set:
    """Extract simple searchable terms for lightweight reranking."""
    return {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9']{2,}", text.lower())
        if term not in {
            "the", "and", "for", "with", "that", "this", "you", "your", "have",
            "many", "what", "how", "are", "does", "policy", "please",
        }
    }


def _rerank_score(query: str, result: Dict, query_lang: str) -> float:
    """Blend vector similarity with language and exact policy-word matches."""
    text = result.get("text", "")
    metadata = result.get("metadata", {})
    score = result.get("similarity", 0.0)

    chunk_lang = metadata.get("language")
    if chunk_lang == query_lang:
        score += 0.035
    elif query_lang == "en":
        score -= 0.06

    query_terms = _terms(query)
    if query_terms:
        searchable = f"{metadata.get('file_name', '')} {metadata.get('document', '')} {text}".lower()
        matched_terms = sum(1 for term in query_terms if term in searchable)
        score += min(0.12, matched_terms * 0.04)

    policy_terms = {"annual", "leave", "sick", "bonus", "password", "remote", "work", "confidential"}
    if query_terms & policy_terms:
        searchable = f"{metadata.get('file_name', '')} {metadata.get('document', '')} {text}".lower()
        policy_matches = sum(1 for term in query_terms & policy_terms if term in searchable)
        score += min(0.10, policy_matches * 0.05)

    if query_lang == "en" and re.search(r"[ØÙïºï»]{2,}", text):
        score -= 0.08

    return max(0.0, score)


def _contains_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def _contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", text or ""))


# -------------------------------------------------------
# Canned friendly responses
# -------------------------------------------------------

GREETING_EN = """Hello! I'm **AI POD**, your internal assistant at **GIG Egypt Life Takaful**.

I'm here to help you find answers about company policies quickly and accurately. What would you like to know?"""

GREETING_AR = """مرحباً! أنا مساعدك الداخلي للسياسات.

أنا هنا لمساعدتك في العثور على إجابات حول سياسات الشركة بسرعة ودقة. بماذا يمكنني مساعدتك؟"""

ABOUT_EN = """I'm **AI POD** — the internal AI assistant for **GIG Egypt Life Takaful**.

Here's what I can help you with:

**HR Policies** — Leave entitlements, attendance, disciplinary procedures, performance reviews
**IT Policies** — Password rules, acceptable use, data security, remote access
**Company Procedures** — Step-by-step guidance on internal processes

**How to use me:**
- Ask me any policy question in **English or Arabic**
- I'll search the official company documents and give you a cited answer
- If I can't find the answer, I'll tell you honestly

**Examples:**
- *"How many annual leave days am I entitled to?"*
- *"ما هي سياسة كلمة المرور؟"*
- *"What is the remote work policy?"*

What would you like to know?"""

ABOUT_AR = """أنا المساعد الذكي الداخلي لسياسات الشركة.

إليك ما يمكنني مساعدتك فيه:

**سياسات الموارد البشرية** — الإجازات، الحضور، الإجراءات التأديبية، تقييم الأداء
**سياسات تقنية المعلومات** — قواعد كلمة المرور، الاستخدام المقبول، أمن البيانات، العمل عن بُعد
**إجراءات الشركة** — إرشادات خطوة بخطوة للعمليات الداخلية

**كيفية استخدامي:**
- اسألني أي سؤال يتعلق بالسياسات **بالعربية أو الإنجليزية**
- سأبحث في وثائق الشركة الرسمية وأعطيك إجابة موثقة
- إذا لم أتمكن من إيجاد الإجابة، سأخبرك بصدق

ما الذي تود معرفته؟"""


# -------------------------------------------------------
# Conversation Memory
# -------------------------------------------------------

class ConversationMemory:
    MAX_TURNS = 6
    MAX_CHARS = 3000

    def __init__(self):
        self.turns: List[Dict[str, str]] = []

    def add(self, question: str, answer: str):
        self.turns.append({"role": "user",      "content": question})
        self.turns.append({"role": "assistant", "content": answer})
        if len(self.turns) > self.MAX_TURNS * 2:
            self.turns = self.turns[-(self.MAX_TURNS * 2):]

    def get_messages(self) -> List[Dict[str, str]]:
        messages, total = [], 0
        for turn in reversed(self.turns):
            n = len(turn["content"])
            if total + n > self.MAX_CHARS:
                break
            messages.insert(0, turn)
            total += n
        return messages

    def clear(self):
        self.turns = []

    def is_empty(self):
        return len(self.turns) == 0

    def summary(self):
        return f"{len(self.turns) // 2} prior turn(s)"


# -------------------------------------------------------
# Query System
# -------------------------------------------------------

class AIPodQuerySystem:

    def __init__(self):
        print("Loading AI POD...")

        if not os.path.exists(AIPodConfig.FAISS_INDEX_PATH):
            raise FileNotFoundError("Index not found — run: python ingest_documents.py")

        self.index = faiss.read_index(AIPodConfig.FAISS_INDEX_PATH)

        with open(AIPodConfig.CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)

        with open(AIPodConfig.METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

        print(f"Loading embedding model: {AIPodConfig.EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(
            AIPodConfig.EMBEDDING_MODEL,
            local_files_only=True,
        )

        self.thresholds = self._load_thresholds()
        self.groq_client = self._init_groq()
        self.memory = ConversationMemory()
        self._print_info()

    # --------------------------------------------------

    def _load_thresholds(self) -> Dict:
        # Sensible defaults for IndexFlatIP with normalised vectors
        defaults = {"high": 0.40, "medium": 0.28, "low": 0.18}
        try:
            if os.path.exists(AIPodConfig.THRESHOLDS_PATH):
                with open(AIPodConfig.THRESHOLDS_PATH, "rb") as f:
                    t = pickle.load(f)
                if t.get("high", 0) > t.get("medium", 0) > t.get("low", 0) > 0:
                    print(f"Thresholds: high={t['high']:.1%} medium={t['medium']:.1%} low={t['low']:.1%}")
                    return t
                print(f"Saved thresholds invalid ({t}) — using defaults")
        except Exception as e:
            print(f"Threshold load error: {e}")
        print(f"Using defaults: {defaults}")
        return defaults

    def _create_groq_client(self):
        return Groq(api_key=AIPodConfig.GROQ_API_KEY)

    def _init_groq(self):
        if not AIPodConfig.GROQ_API_KEY:
            print("GROQ_API_KEY not set — add to .env")
            return None
        try:
            print("Groq configured")
            return self._create_groq_client()
        except Exception as e:
            print(f"Groq error: {e}")
            return None

    def _groq_completion(self, **kwargs):
        if not self.groq_client:
            return None
        try:
            return self.groq_client.chat.completions.create(**kwargs)
        except Exception as e:
            if "client has been closed" not in str(e).lower():
                raise
            print("Groq client was closed; rebuilding client and retrying once")
            self.groq_client = self._create_groq_client()
            return self.groq_client.chat.completions.create(**kwargs)

    def _groq_stream(self, **kwargs):
        """Yield text deltas from Groq while keeping retry behavior centralized."""
        if not self.groq_client:
            return
        yielded_any = False
        try:
            stream = self._groq_completion(stream=True, **kwargs)
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yielded_any = True
                    yield delta
        except Exception as e:
            if "client has been closed" not in str(e).lower() or yielded_any:
                raise
            print("Groq client was closed during stream; rebuilding client and retrying once")
            self.groq_client = self._create_groq_client()
            stream = self.groq_client.chat.completions.create(stream=True, **kwargs)
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    def _print_info(self):
        langs = {}
        for c in self.chunks:
            l = c.get("metadata", {}).get("language", "?")
            langs[l] = langs.get(l, 0) + 1
        print(f"\n{len(self.chunks)} chunks | {dict(langs)}")
        print(f"Thresholds — high:{self.thresholds['high']:.1%}  med:{self.thresholds['medium']:.1%}  low:{self.thresholds['low']:.1%}\n")

    # --------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        return self.embed_model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

    def search_semantic(self, query: str, k: int = None) -> List[Dict]:
        if k is None:
            k = AIPodConfig.DEFAULT_SEARCH_K
        query_lang = detect_language(query)
        vec = self._embed(query)
        scores, indices = self.index.search(vec, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            cos_sim = max(0.0, float(scores[0][i]))
            result = {
                "text":       self.chunks[idx].get("text", ""),
                "similarity": cos_sim,
                "raw_similarity": cos_sim,
                "metadata":   self.chunks[idx].get("metadata", {}),
                "chunk_id":   int(idx),
            }
            result["similarity"] = _rerank_score(query, result, query_lang)
            results.append(result)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    # --------------------------------------------------

    def _resolve_followup_query(self, question: str) -> str:
        """Rewrite short follow-ups into standalone retrieval queries."""
        question = (question or "").strip()
        if not question or self.memory.is_empty():
            return question

        words = re.findall(r"\w+", question, re.UNICODE)
        clear_topic_pattern = re.compile(
            r"\b(leave|password|remote|bonus|sick|annual|maternity|paternity|notice|salary|policy|"
            r"confidential|attendance|disciplinary|performance|vacation|hr|it)\b|"
            r"إجاز|اجاز|كلمة المرور|عن بعد|مكافأة|مرض|سنوي|أمومة|امومة|أبوة|ابوة|"
            r"إخطار|اخطار|راتب|سياسة|سرية|حضور|تأديب|أداء|الموارد البشرية",
            re.IGNORECASE,
        )
        if len(words) > 8 or clear_topic_pattern.search(question):
            return question

        followup_pattern = re.compile(
            r"(^\s*(what about|how about|and for|how do i|can i|does it|is it|what if)\b|"
            r"\b(it|that|this|them|they|those|there|same)\b|"
            r"ماذا عنه|ماذا عنها|ماذا بالنسبة|هل يمكن|كيف|نفس الشيء|ذلك|هذه|هذا)",
            re.IGNORECASE,
        )
        if not followup_pattern.search(question):
            return question
        if not self.groq_client:
            return question

        try:
            recent_messages = self.memory.get_messages()[-6:]
            conversation = "\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}"
                for message in recent_messages
            )
            prompt = (
                "Given this recent conversation:\n"
                f"{conversation}\n\n"
                "Rewrite the user's latest message as a standalone, fully-specified question "
                "that includes any topic or subject implied by the conversation. Keep it in the "
                "same language as the latest message. Output ONLY the rewritten question, nothing else.\n\n"
                f"Latest message: {question}"
            )
            resp = self._groq_completion(
                model=AIPodConfig.LLM_MODEL_FAST,
                messages=[
                    {"role": "system", "content": "You rewrite follow-up questions for search retrieval."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=60,
            )
            rewritten = resp.choices[0].message.content.strip().strip('"').strip("'")
            if rewritten and 3 <= len(rewritten) <= 300:
                return rewritten
        except Exception as e:
            print(f"Follow-up query rewrite skipped: {e}")
        return question

    # --------------------------------------------------

    def _call_groq_policy(self, question: str, chunks: List[Dict], answer_style: str = "summary", stream: bool = False):
        """
        Answer a policy question strictly from document chunks.
        Friendly tone but answers are grounded ONLY in the provided sources.
        """
        lang = detect_language(question)
        bilingual = wants_bilingual_response(question)
        context = "\n\n---\n\n".join([
            f"Policy excerpt:\n{c['text']}"
            for c in chunks[:AIPodConfig.MAX_CHUNKS_PER_QUERY]
        ])

        # CRITICAL: Response language is ALWAYS determined by the question language,
        # NOT by the language of the source documents retrieved.
        if bilingual:
            reply_lang_instruction = (
                "The user explicitly requested both Arabic and English. Provide two clearly separated sections: Arabic first, then English. "
                "Do not mix languages inside the same bullet."
            )
        else:
            reply_lang_instruction = (
                "CRITICAL: You MUST reply fully in Arabic regardless of the document language. "
                "Use Arabic headings and Arabic bullet text only. Do not include any English words."
                if lang == "ar"
                else "CRITICAL: You MUST reply fully in English regardless of the document language. "
                     "Use English headings and English bullet text only. Do not include any Arabic words. "
                     "The source documents may be in Arabic — translate the relevant information into English."
            )
        style = "detailed" if answer_style == "detailed" else "summary"
        if lang == "ar" and not bilingual:
            style_instruction = (
                "RESPONSE STYLE: Detailed. Use this exact Arabic structure:\n"
                "الإجابة المباشرة:\n"
                "قدم إجابة قصيرة ومباشرة على سؤال الموظف.\n\n"
                "التفاصيل:\n"
                "اجمع المعلومات المرتبطة تحت عناوين عربية واضحة مثل الإجازة السنوية، الإجازة المرضية، أيام الراحة الأسبوعية، الأهلية، الإجراءات، أو الاستثناءات.\n"
                "استخدم نقاطاً مختصرة. إذا كانت الإجابة تقارن استحقاقات حسب سنوات الخدمة، استخدم جدولاً بالعربية.\n\n"
                "ملاحظات إضافية:\n"
                "اذكر الاستثناءات أو القيود أو سلطة الإدارة أو عدم وجود إجمالي موحد."
                if style == "detailed"
                else "RESPONSE STYLE: Summary. Use this exact Arabic structure:\n"
                     "الإجابة المباشرة:\n"
                     "قدم إجابة قصيرة ومباشرة.\n\n"
                     "التفاصيل:\n"
                     "استخدم 3-5 نقاط مختصرة مجمعة حسب موضوع السياسة عند الحاجة.\n\n"
                     "ملاحظات إضافية:\n"
                     "اذكر فقط الاستثناءات أو القيود المهمة."
            )
        else:
            style_instruction = (
                "RESPONSE STYLE: Detailed. Use this exact structure:\n"
                "Direct Answer:\n"
                "Give a short direct answer to the employee question.\n\n"
                "Details:\n"
                "Group related policy information under clear section headings such as Annual Leave, Sick Leave, Weekly Offs, Eligibility, Process, or Exceptions.\n"
                "Use bullets. If the answer compares entitlements by years of service, use a markdown table.\n\n"
                "Additional Notes:\n"
                "List exceptions, restrictions, management discretion, or missing totals.\n"
                if style == "detailed"
                else "RESPONSE STYLE: Summary. Use this exact structure:\n"
                     "Direct Answer:\n"
                     "Give a short direct answer to the employee question.\n\n"
                     "Details:\n"
                     "Use 3-5 concise bullets grouped by policy topic when needed.\n\n"
                     "Additional Notes:\n"
                     "Include only important exceptions, restrictions, management discretion, or missing totals."
            )

        system = (
            f"You are AI POD, the official HR and IT policy assistant for {AIPodConfig.COMPANY_NAME}.\n\n"
            "Your task is to answer employee questions using the provided policy documents.\n\n"
            f"{reply_lang_instruction}\n\n"
            f"{style_instruction}\n\n"
            "STRICT RULES:\n"
            "1. NEVER return raw document text.\n"
            "2. NEVER copy policy sections exactly as they appear.\n"
            "3. ALWAYS understand the user's question first.\n"
            "4. Extract only the relevant information.\n"
            "5. Rewrite the answer in a professional and easy-to-read format.\n"
            "6. Remove document references, numbering, page breaks, file names, and OCR artifacts.\n"
            "7. Group related information together.\n"
            "8. Use bullet points whenever possible.\n"
            "9. If the answer contains multiple policies, separate them into sections.\n"
            "10. Always provide a direct answer first, then supporting details.\n"
            "11. Never show source chunk dumps, incomplete sentences, repeated text, or document filenames.\n"
            "12. If the answer is not in the documents, say so clearly and suggest contacting HR or IT directly.\n"
            "13. The final answer language must exactly match the user's question language."
        )

        response_language_label = "in Arabic and English" if bilingual else ("in English" if lang == "en" else "in Arabic")
        user_msg = (
            f"Available documents (may be in Arabic or English — extract and translate as needed):\n\n"
            f"{context}\n\n"
            f"Employee question: {question}\n\n"
            f"Answer ({style}, {response_language_label}, polished HR-style response, from documents only):"
        )

        messages = [{"role": "system", "content": system}]
        messages.extend(self.memory.get_messages())
        messages.append({"role": "user", "content": user_msg})

        try:
            request = {
                "model": AIPodConfig.LLM_MODEL_FAST,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1200 if style == "detailed" else 550,
            }
            if stream:
                return self._groq_stream(**request)
            resp = self._groq_completion(**request)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq error: {e}")
            if stream:
                raise
            raise

    def _basic_policy_answer(self, question: str, chunks: List[Dict], answer_style: str = "summary") -> str:
        """Readable local fallback used when the LLM is unavailable."""
        if not chunks:
            return self._no_info_response(question)["answer"]

        lang = detect_language(question)
        top = chunks[0]
        text = re.sub(r"\s+", " ", top.get("text", "")).strip()
        lower_question = question.lower()
        combined_text = " ".join(re.sub(r"\s+", " ", chunk.get("text", "")).strip() for chunk in chunks[:5])
        bilingual = wants_bilingual_response(question)
        ar_timeoff_question = lang == "ar" and any(
            term in question for term in ["إجاز", "اجاز", "راحة", "أيام", "ايام", "سنوي", "مرض"]
        )
        ar_parental_question = lang == "ar" and any(
            term in question for term in ["أمومة", "امومة", "ولادة", "حمل", "أبوة", "ابوة", "تبني"]
        )
        en_parental_question = lang == "en" and any(
            term in lower_question for term in ["maternity", "paternity", "parental", "adoption", "pregnant"]
        )
        timeoff_question = ar_timeoff_question or (
            lang == "en" and (
                ("day" in lower_question and "off" in lower_question)
                or "time off" in lower_question
                or ("leave" in lower_question and "year" in lower_question)
            )
        )
        if timeoff_question or ar_parental_question or en_parental_question:
            policy_corpus_parts = [combined_text]
            for chunk in self.chunks:
                chunk_text = re.sub(r"\s+", " ", chunk.get("text", "")).strip()
                lower_chunk = chunk_text.lower()
                if (
                    "annual leave" in lower_chunk
                    or "sick leave" in lower_chunk
                    or "weekly offs" in lower_chunk
                    or "2nd saturday is off" in lower_chunk
                    or "maternity leave" in lower_chunk
                    or "paternity leave" in lower_chunk
                    or "adoption leave" in lower_chunk
                ):
                    policy_corpus_parts.append(chunk_text)
                    if len(policy_corpus_parts) >= 8:
                        break
            combined_text = " ".join(policy_corpus_parts)

        if bilingual and (ar_parental_question or en_parental_question):
            return (
                "العربية:\n"
                "الإجابة المباشرة:\n"
                "تستحق الموظفة 14 أسبوعاً من إجازة الأمومة المدفوعة.\n\n"
                "التفاصيل:\n"
                "- 6 أسابيع قبل الولادة و8 أسابيع بعد الولادة، مع مرونة حسب المشورة الطبية.\n"
                "- يمكن الحصول على 4 أسابيع إضافية غير مدفوعة الأجر بموافقة المدير.\n"
                "- يجب إخطار الموارد البشرية قبل موعد الولادة المتوقع بثلاثة أشهر على الأقل.\n\n"
                "English:\n"
                "Direct Answer:\n"
                "Female employees are entitled to 14 weeks of paid maternity leave.\n\n"
                "Details:\n"
                "- 6 weeks before delivery and 8 weeks after delivery, with flexibility based on medical advice.\n"
                "- An additional 4 weeks of unpaid leave may be available with manager approval.\n"
                "- Employees must notify HR at least 3 months before the expected due date."
            )

        if bilingual and timeoff_question:
            return (
                "العربية:\n"
                "الإجابة المباشرة:\n"
                "تستحق إجازة سنوية وإجازة مرضية وأيام راحة أسبوعية وشهرية حسب ما ينطبق عليك.\n\n"
                "التفاصيل:\n"
                "- الإجازة السنوية: 21 يوماً من 0 إلى 2 سنة خدمة، 25 يوماً من 3 إلى 5 سنوات، 30 يوماً من 6 إلى 10 سنوات، و35 يوماً لأكثر من 10 سنوات.\n"
                "- الإجازة المرضية: 10 أيام سنوياً.\n"
                "- أيام الراحة: الأحد راحة أسبوعية، والسبت الثاني من كل شهر راحة أيضاً.\n\n"
                "English:\n"
                "Direct Answer:\n"
                "You are entitled to annual leave, sick leave, weekly offs, and the monthly 2nd Saturday off where applicable.\n\n"
                "Details:\n"
                "- Annual leave: 21 days for 0-2 years of service, 25 days for 3-5 years, 30 days for 6-10 years, and 35 days for 10+ years.\n"
                "- Sick leave: 10 days per year.\n"
                "- Weekly offs: Sunday is the weekly off day, and the 2nd Saturday of every month is also off."
            )

        if ar_parental_question:
            return (
                "الإجابة المباشرة:\n"
                "تستحق الموظفة 14 أسبوعاً من إجازة الأمومة المدفوعة وفقاً للسياسة.\n\n"
                "التفاصيل:\n"
                "إجازة الأمومة:\n"
                "- 14 أسبوعاً مدفوعة الأجر.\n"
                "- يمكن تقسيمها إلى 6 أسابيع قبل الولادة و8 أسابيع بعد الولادة، مع مرونة حسب المشورة الطبية.\n"
                "- يمكن الحصول على 4 أسابيع إضافية غير مدفوعة الأجر بموافقة المدير.\n\n"
                "إجازات مرتبطة:\n"
                "- إجازة التبني: 10 أسابيع للوالد أو الوالدة الأساسي في حالة التبني.\n"
                "- إجازة الأبوة: 5 أيام مدفوعة الأجر، مع إمكانية إجازة إضافية غير مدفوعة لمدة أسبوعين.\n\n"
                "ملاحظات إضافية:\n"
                "- يجب إخطار الموارد البشرية قبل موعد الولادة المتوقع بثلاثة أشهر على الأقل.\n"
                "- قد تتوفر برامج للعودة التدريجية إلى العمل بعد الإجازة."
            )

        if en_parental_question:
            return (
                "Direct Answer:\n"
                "Female employees are entitled to 14 weeks of paid maternity leave.\n\n"
                "Details:\n"
                "Maternity Leave:\n"
                "- 14 weeks of paid leave.\n"
                "- This may be split as 6 weeks before delivery and 8 weeks after delivery, with flexibility based on medical advice.\n"
                "- An additional 4 weeks of unpaid leave may be available with manager approval.\n\n"
                "Related Leave:\n"
                "- Adoption leave: 10 weeks for the primary adopting parent.\n"
                "- Paternity leave: 5 paid days, with an additional 2 weeks of unpaid leave available.\n\n"
                "Additional Notes:\n"
                "- Employees must notify HR at least 3 months before the expected due date.\n"
                "- Return-to-work transition programs may be available."
            )

        if ar_timeoff_question:
            has_annual = re.search(r"Employees are entitled to\s+([^\.]+annual leave days[^\.]*)\.", combined_text, re.IGNORECASE)
            has_sick = re.search(r"Employees have\s+([^\.]+sick leave days[^\.]*)\.", combined_text, re.IGNORECASE)
            has_weekly = re.search(r"Weekly Offs|Sunday is the weekly off|2nd Saturday is off", combined_text, re.IGNORECASE)
            has_workload = re.search(r"Weekly off[’']s on Sunday can be cancelled by the Management", combined_text, re.IGNORECASE)

            detail_sections = []
            if has_annual:
                detail_sections.append(
                    "الإجازة السنوية:\n"
                    "- 21 يوماً سنوياً للموظفين من 0 إلى 2 سنة خدمة.\n"
                    "- 25 يوماً سنوياً للموظفين من 3 إلى 5 سنوات خدمة.\n"
                    "- 30 يوماً سنوياً للموظفين من 6 إلى 10 سنوات خدمة.\n"
                    "- 35 يوماً سنوياً للموظفين أكثر من 10 سنوات خدمة."
                )
            if has_sick:
                detail_sections.append("الإجازة المرضية:\n- 10 أيام سنوياً للمرض الشخصي أو المواعيد الطبية.")
            if has_weekly:
                detail_sections.append(
                    "أيام الراحة الأسبوعية:\n"
                    "- يوم الأحد هو يوم الراحة الأسبوعية.\n"
                    "- السبت الثاني من كل شهر هو يوم راحة أيضاً."
                )

            if detail_sections:
                notes = []
                if has_workload:
                    notes.append("- يمكن للإدارة إلغاء الراحة الأسبوعية حسب متطلبات العمل.")
                notes.append("- لا توفر السياسة إجمالياً موحداً يجمع كل الإجازات وأيام الراحة في رقم واحد.")
                return (
                    "الإجابة المباشرة:\n"
                    "تستحق إجازة سنوية وإجازة مرضية وأيام راحة أسبوعية وشهرية حسب ما ينطبق عليك.\n\n"
                    "التفاصيل:\n"
                    + "\n\n".join(detail_sections)
                    + "\n\n"
                    "ملاحظات إضافية:\n"
                    + "\n".join(dict.fromkeys(notes))
                )

        if lang == "en" and timeoff_question:
            bullets = []
            annual = re.search(
                r"Employees are entitled to\s+([^\.]+annual leave days[^\.]*)\.",
                combined_text,
                re.IGNORECASE,
            )
            sick = re.search(
                r"Employees have\s+([^\.]+sick leave days[^\.]*)\.",
                combined_text,
                re.IGNORECASE,
            )
            weekly = re.search(
                r"Weekly Offs\s+a\)\s*(.+?)(?:b\)|3\. Lunch Hours|$)",
                combined_text,
                re.IGNORECASE,
            )
            second_saturday = re.search(
                r"2nd Saturday is off every month",
                combined_text,
                re.IGNORECASE,
            )
            workload = re.search(
                r"Weekly off[’']s on Sunday can be cancelled by the Management[^\.]*\.",
                combined_text,
                re.IGNORECASE,
            )

            detail_sections = []
            if annual:
                detail_sections.append(
                    "Annual Leave:\n"
                    "- 21 days per year for 0-2 years of service.\n"
                    "- 25 days per year for 3-5 years of service.\n"
                    "- 30 days per year for 6-10 years of service.\n"
                    "- 35 days per year for 10+ years of service."
                )
            if sick:
                detail_sections.append("Sick Leave:\n- 10 days per year for personal illness or medical appointments.")
            if weekly or second_saturday:
                weekly_lines = []
                if weekly:
                    weekly_lines.append("- Sunday is the weekly off day.")
                if second_saturday:
                    weekly_lines.append("- The 2nd Saturday of every month is also off.")
                detail_sections.append("Weekly Offs:\n" + "\n".join(weekly_lines))

            if detail_sections:
                notes = []
                if workload:
                    notes.append("- Weekly offs may be cancelled by management depending on business requirements.")
                notes.append("- The policy does not provide one combined annual total for all leave days and weekly/monthly offs.")
                return (
                    "Direct Answer:\n"
                    "You are entitled to annual leave, sick leave, weekly offs, and the monthly 2nd Saturday off where applicable.\n\n"
                    "Details:\n"
                    + "\n\n".join(detail_sections)
                    + "\n\n"
                    "Additional Notes:\n"
                    + "\n".join(dict.fromkeys(notes))
                )

        if lang == "en" and "annual" in lower_question and ("leave" in lower_question or "day" in lower_question):
            bullets = []
            entitlement = re.search(r"Employees are entitled to\s+([^\.]+annual leave days[^\.]*)\.", text, re.IGNORECASE)
            accrual = re.search(r"Annual leave accrues at\s+(.+?)(?:\. Leave requests|$)", text, re.IGNORECASE)
            tiers = re.search(r"Accrual Details:\s*(.+?)(?:Unused annual leave|$)", text, re.IGNORECASE)
            carry = re.search(r"Unused annual leave\s+([^\.]+)\.", text, re.IGNORECASE)

            if entitlement:
                bullets.append(f"- **Annual Leave:** {entitlement.group(1).strip()}.")
            if accrual:
                bullets.append(f"- **Accrual:** Annual leave accrues at {accrual.group(1).strip()}.")
            if tiers:
                tier_text = re.sub(r"\s+", " ", tiers.group(1).strip())
                bullets.append(f"- **Service Tiers:** {tier_text}.")
            if carry:
                bullets.append(f"- **Carryover:** Unused annual leave {carry.group(1).strip()}.")

            if bullets:
                return (
                    "Direct Answer:\n"
                    "Your annual leave entitlement depends on your years of service.\n\n"
                    "Details:\n"
                    "Annual Leave:\n"
                    "- 21 days per year for 0-2 years of service.\n"
                    "- 25 days per year for 3-5 years of service.\n"
                    "- 30 days per year for 6-10 years of service.\n"
                    "- 35 days per year for 10+ years of service.\n\n"
                    "Accrual and Carryover:\n"
                    + "\n".join(bullet for bullet in bullets if "Accrual" in bullet or "Carryover" in bullet)
                    + "\n\nAdditional Notes:\n"
                    "- Unused annual leave may be carried over only within the limits stated in the policy."
                )

        query_terms = _terms(question)
        readable_text = re.sub(r"\s+([a-z]\))", r"\n\1", text)
        readable_text = re.sub(r"\s+(\d+\.\s+[A-Z])", r"\n\1", readable_text)
        sentences = re.split(r"(?<=[.!?])\s+|\n+", readable_text)
        selected = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip() and any(term in sentence.lower() for term in query_terms)
        ][:4]
        if lang == "en":
            selected = [
                sentence for sentence in selected
                if not re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]|[ØÙïºï»]{2,}", sentence)
            ]
        if not selected:
            if lang == "en":
                return (
                    "Direct Answer:\n"
                    "I could not find enough clear policy information to answer this question confidently.\n\n"
                    "Details:\n"
                    "- Please ask about a more specific HR or IT policy topic.\n"
                    "- I will not show raw document text or unreadable OCR excerpts.\n\n"
                    "Additional Notes:\n"
                    "- For urgent or personal cases, please contact HR or IT directly."
                )
            selected = [sentences[0].strip()] if sentences else [text[:350]]

        if lang == "ar":
            return (
                "الإجابة المباشرة:\n"
                "لم أتمكن من صياغة إجابة عربية مؤكدة من المعلومات المتاحة لهذا السؤال.\n\n"
                "التفاصيل:\n"
                "- يرجى إعادة صياغة السؤال بشكل أكثر تحديداً، مثل نوع السياسة أو نوع الإجازة أو إجراء تكنولوجيا المعلومات المطلوب.\n"
                "- لن أعرض نصوصاً خاماً أو مقتطفات غير منسقة من المستندات.\n\n"
                "ملاحظات إضافية:\n"
                "- إذا كان الأمر عاجلاً أو خاصاً بحالتك الشخصية، يرجى الرجوع إلى الموارد البشرية أو تكنولوجيا المعلومات للتأكيد."
            )

        bullets = "\n".join(f"- {sentence}" for sentence in selected)
        return (
            "Direct Answer:\n"
            "Here is the relevant policy information for your question.\n\n"
            "Details:\n"
            f"{bullets}\n\n"
            "Additional Notes:\n"
            "- If the policy does not fully answer your situation, please contact HR or IT for confirmation."
        )

    def _language_safe_answer(self, lang: str) -> str:
        if lang == "ar":
            return (
                "الإجابة المباشرة:\n"
                "لم أتمكن من صياغة إجابة عربية مؤكدة من المعلومات المتاحة لهذا السؤال.\n\n"
                "التفاصيل:\n"
                "- يرجى إعادة صياغة السؤال بشكل أكثر تحديداً.\n"
                "- لن أعرض أي نص إنجليزي أو مقتطفات خام داخل إجابة عربية.\n\n"
                "ملاحظات إضافية:\n"
                "- إذا كان السؤال متعلقاً بحالة شخصية أو عاجلة، يرجى الرجوع إلى الموارد البشرية أو تكنولوجيا المعلومات للتأكيد."
            )
        return (
            "Direct Answer:\n"
            "I could not produce a confirmed English answer from the available policy information.\n\n"
            "Details:\n"
            "- Please rephrase the question with a more specific HR or IT policy topic.\n"
            "- I will not show raw or mixed-language document excerpts.\n\n"
            "Additional Notes:\n"
            "- For urgent or personal cases, please contact HR or IT directly."
        )

    def _enforce_answer_language(self, answer: str, lang: str, bilingual: bool = False) -> str:
        if bilingual:
            return answer
        if lang == "ar" and _contains_latin(answer):
            return self._language_safe_answer(lang)
        if lang == "en" and _contains_arabic(answer):
            return self._language_safe_answer(lang)
        return answer

    def _call_groq_general(self, question: str, stream: bool = False):
        """
        Handle general / conversational questions with a friendly response.
        No document grounding needed.
        """
        lang = detect_language(question)
        bilingual = wants_bilingual_response(question)
        if lang == "ar":
            system = (
                "أنت المساعد الذكي الودود لسياسات الشركة.\n"
                "أجب على الأسئلة العامة بأسلوب ودي ومفيد وبالعربية فقط. "
                "إذا سألوا عن الشركة أو سياساتها، شجعهم على طرح سؤال محدد."
            )
        else:
            system = (
                f"You are AI POD, the friendly internal AI assistant for {AIPodConfig.COMPANY_NAME}.\n"
                "Answer general questions warmly and helpfully. "
                "If they ask about company policies, encourage them to ask a specific question."
            )

        messages = [{"role": "system", "content": system}]
        messages.extend(self.memory.get_messages())
        messages.append({"role": "user", "content": question})

        try:
            request = {
                "model": AIPodConfig.LLM_MODEL_FAST,
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 400,
            }
            if stream:
                return self._groq_stream(**request)
            resp = self._groq_completion(**request)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq error: {e}")
            if stream:
                raise
            raise

    def _call_groq_chat(self, question: str, stream: bool = False):
        """Handle open conversation without document retrieval."""
        lang = detect_language(question)
        bilingual = wants_bilingual_response(question)
        if bilingual:
            language_instruction = (
                "The user explicitly requested both Arabic and English. Answer in both languages clearly."
            )
        elif lang == "ar":
            language_instruction = "Answer only in professional, natural Arabic. Do not include English unless explicitly requested."
        else:
            language_instruction = "Answer only in natural English. Do not include Arabic unless explicitly requested."

        system = (
            f"You are AI POD, the internal assistant for {AIPodConfig.COMPANY_NAME}.\n"
            "You can have normal, friendly conversation on any topic. You are not limited to company policy. "
            "Answer general questions naturally and helpfully, the way a knowledgeable assistant would.\n"
            "If the user asks about company HR, IT, internal procedures, or official policies, tell them you can help and ask for the specific policy question.\n"
            f"{language_instruction}"
        )

        messages = [{"role": "system", "content": system}]
        messages.extend(self.memory.get_messages())
        messages.append({"role": "user", "content": question})

        request = {
            "model": AIPodConfig.LLM_MODEL_FAST,
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": 650,
        }
        if stream:
            return self._groq_stream(**request)
        resp = self._groq_completion(**request)
        return resp.choices[0].message.content.strip()

    # --------------------------------------------------

    def ask(self, question: str, answer_style: str = "summary") -> Dict:
        start = time.time()
        question = question.strip()
        if not question:
            return self._error_response(question)

        lang = detect_language(question)
        bilingual = wants_bilingual_response(question)
        intent = classify_intent(question)

        # ── Greeting ──────────────────────────────────────────────
        if intent == "greeting":
            answer = GREETING_AR if lang == "ar" else GREETING_EN
            answer = self._enforce_answer_language(answer, lang, bilingual)
            self.memory.add(question, answer)
            return self._friendly_response(answer, lang, time.time() - start)

        # ── About the bot ─────────────────────────────────────────
        if intent == "about_bot":
            answer = ABOUT_AR if lang == "ar" else ABOUT_EN
            answer = self._enforce_answer_language(answer, lang, bilingual)
            self.memory.add(question, answer)
            return self._friendly_response(answer, lang, time.time() - start)

        if intent == "general_chat":
            if not self.groq_client:
                return self._chat_unavailable_response(question)
            try:
                answer = self._call_groq_chat(question)
            except Exception as e:
                print(f"general_chat unavailable: {e}")
                return self._chat_unavailable_response(question)
            answer = self._enforce_answer_language(answer, lang, bilingual)
            self.memory.add(question, answer)
            return {
                "answer": answer, "sources": [], "confidence": 1.0,
                "mode": "general_chat", "match_type": "conversational",
                "language": lang, "response_time": time.time() - start,
            }

        # ── Policy question — semantic search ─────────────────────
        try:
            search_query = self._resolve_followup_query(question)
            if search_query != question:
                print(f"Resolved follow-up query: '{question}' -> '{search_query}'")
            results = self.search_semantic(search_query)
            if not results:
                return self._no_info_response(question)

            best = results[0]["similarity"]
            top = results[:max(AIPodConfig.TOP_K_RESULTS, AIPodConfig.MAX_CHUNKS_PER_QUERY)]

            print(f"'{question[:50]}' | best={best:.1%} | high={self.thresholds['high']:.1%} | med={self.thresholds['medium']:.1%}")

            if best >= self.thresholds["high"]:
                if not self.groq_client:
                    return self._api_unavailable_response(question)
                answer = self._call_groq_policy(question, top, answer_style)
                answer = self._enforce_answer_language(answer, lang, bilingual)
                self.memory.add(question, answer)
                return {
                    "answer": answer, "sources": top, "confidence": best,
                    "mode": "ai_enhanced", "match_type": "exact_match",
                    "language": lang, "response_time": time.time() - start,
                }

            elif best >= self.thresholds["medium"]:
                if not self.groq_client:
                    return self._api_unavailable_response(question)
                answer = self._call_groq_policy(question, top, answer_style)
                answer = self._enforce_answer_language(answer, lang, bilingual)
                answer += (
                    "\n\nملاحظة: قد لا تكون هذه المعلومات إجابة مباشرة لسؤالك."
                    if lang == "ar"
                    else "\n\nNote: These results are related but may not directly answer your question."
                )
                self.memory.add(question, answer)
                return {
                    "answer": answer, "sources": top, "confidence": best,
                    "mode": "related_match", "match_type": "related_match",
                    "language": lang, "response_time": time.time() - start,
                }

            else:
                # Nothing found in documents — try Groq as general fallback
                if not self.groq_client:
                    return self._api_unavailable_response(question)
                answer = self._call_groq_general(question)
                answer = self._enforce_answer_language(answer, lang, bilingual)
                self.memory.add(question, answer)
                return {
                    "answer": answer, "sources": [], "confidence": best,
                    "mode": "general_fallback", "match_type": "none",
                    "language": lang, "response_time": time.time() - start,
                }

        except Exception as e:
            print(f"ask() API generation unavailable: {e}")
            return self._api_unavailable_response(question)

    # --------------------------------------------------

    def _stream_static_answer(self, answer: str, chunk_size: int = 36):
        """Stream local/cached-style text in small chunks for a consistent UI."""
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]

    def _stream_interrupted_note(self, lang: str) -> str:
        return (
            "\n\nتعذر إكمال الرد بسبب انقطاع مؤقت. يرجى إعادة المحاولة إذا كانت الإجابة غير مكتملة."
            if lang == "ar"
            else "\n\nThe response was interrupted. Please try again if the answer is incomplete."
        )

    def ask_stream(self, question: str, answer_style: str = "summary"):
        """Yield assistant response events as text is generated."""
        start = time.time()
        question = question.strip()
        if not question:
            result = self._error_response(question)
            yield {"type": "done", "result": result}
            return

        lang = detect_language(question)
        bilingual = wants_bilingual_response(question)
        intent = classify_intent(question)

        def emit_static(result: Dict):
            answer = result.get("answer", "")
            for delta in self._stream_static_answer(answer):
                yield {"type": "delta", "text": delta}
            yield {"type": "done", "result": result}

        if intent == "greeting":
            answer = GREETING_AR if lang == "ar" else GREETING_EN
            answer = self._enforce_answer_language(answer, lang, bilingual)
            self.memory.add(question, answer)
            yield from emit_static(self._friendly_response(answer, lang, time.time() - start))
            return

        if intent == "about_bot":
            answer = ABOUT_AR if lang == "ar" else ABOUT_EN
            answer = self._enforce_answer_language(answer, lang, bilingual)
            self.memory.add(question, answer)
            yield from emit_static(self._friendly_response(answer, lang, time.time() - start))
            return

        if intent == "general_chat":
            if not self.groq_client:
                yield from emit_static(self._chat_unavailable_response(question))
                return
            answer = ""
            try:
                for delta in self._call_groq_chat(question, stream=True):
                    answer += delta
                    yield {"type": "delta", "text": delta}
            except Exception as e:
                print(f"general_chat stream unavailable: {e}")
                if not answer:
                    yield from emit_static(self._chat_unavailable_response(question))
                    return
                note = self._stream_interrupted_note(lang)
                answer += note
                yield {"type": "delta", "text": note}

            answer = self._enforce_answer_language(answer.strip(), lang, bilingual)
            self.memory.add(question, answer)
            yield {
                "type": "done",
                "result": {
                    "answer": answer,
                    "sources": [],
                    "confidence": 1.0,
                    "mode": "general_chat",
                    "match_type": "conversational",
                    "language": lang,
                    "response_time": time.time() - start,
                },
            }
            return

        try:
            search_query = self._resolve_followup_query(question)
            if search_query != question:
                print(f"Resolved follow-up query: '{question}' -> '{search_query}'")
            results = self.search_semantic(search_query)
            if not results:
                yield from emit_static(self._no_info_response(question))
                return

            best = results[0]["similarity"]
            top = results[:max(AIPodConfig.TOP_K_RESULTS, AIPodConfig.MAX_CHUNKS_PER_QUERY)]
            print(f"stream '{question[:50]}' | best={best:.1%} | high={self.thresholds['high']:.1%} | med={self.thresholds['medium']:.1%}")

            mode = "general_fallback"
            match_type = "none"
            answer = ""

            if best >= self.thresholds["high"]:
                mode = "ai_enhanced"
                match_type = "exact_match"
                if self.groq_client:
                    try:
                        for delta in self._call_groq_policy(question, top, answer_style, stream=True):
                            answer += delta
                            yield {"type": "delta", "text": delta}
                    except Exception as e:
                        print(f"Groq stream error: {e}")
                        if not answer:
                            result = self._api_unavailable_response(question)
                            yield from emit_static(result)
                            return
                        else:
                            note = self._stream_interrupted_note(lang)
                            answer += note
                            yield {"type": "delta", "text": note}
                else:
                    result = self._api_unavailable_response(question)
                    yield from emit_static(result)
                    return

            elif best >= self.thresholds["medium"]:
                mode = "related_match"
                match_type = "related_match"
                if self.groq_client:
                    try:
                        for delta in self._call_groq_policy(question, top, answer_style, stream=True):
                            answer += delta
                            yield {"type": "delta", "text": delta}
                    except Exception as e:
                        print(f"Groq stream error: {e}")
                        if not answer:
                            result = self._api_unavailable_response(question)
                            yield from emit_static(result)
                            return
                        else:
                            note = self._stream_interrupted_note(lang)
                            answer += note
                            yield {"type": "delta", "text": note}
                    related_note = (
                        "\n\nملاحظة: قد لا تكون هذه المعلومات إجابة مباشرة لسؤالك."
                        if lang == "ar"
                        else "\n\nNote: These results are related but may not directly answer your question."
                    )
                    answer += related_note
                    yield {"type": "delta", "text": related_note}
                else:
                    result = self._api_unavailable_response(question)
                    yield from emit_static(result)
                    return

            else:
                top = []
                if self.groq_client:
                    try:
                        for delta in self._call_groq_general(question, stream=True):
                            answer += delta
                            yield {"type": "delta", "text": delta}
                    except Exception as e:
                        print(f"Groq stream error: {e}")
                        if not answer:
                            result = self._api_unavailable_response(question)
                            yield from emit_static(result)
                            return
                        note = self._stream_interrupted_note(lang)
                        answer += note
                        yield {"type": "delta", "text": note}
                else:
                    result = self._api_unavailable_response(question)
                    yield from emit_static(result)
                    return

            answer = self._enforce_answer_language(answer.strip(), lang, bilingual)
            self.memory.add(question, answer)
            yield {
                "type": "done",
                "result": {
                    "answer": answer,
                    "sources": top,
                    "confidence": best,
                    "mode": mode,
                    "match_type": match_type,
                    "language": lang,
                    "response_time": time.time() - start,
                },
            }

        except Exception as e:
            print(f"ask_stream() API generation unavailable: {e}")
            result = self._api_unavailable_response(question)
            yield from emit_static(result)

    # --------------------------------------------------

    def debug_query(self, question: str, top_k: int = 7):
        print(f"\n{'='*60}")
        print(f" DEBUG: '{question}'")
        print(f"   Intent    : {classify_intent(question)}")
        print(f"   Language  : {detect_language(question)}")
        print(f"   Model     : {AIPodConfig.EMBEDDING_MODEL}")
        print(f"   Thresholds: high={self.thresholds['high']:.1%}  med={self.thresholds['medium']:.1%}\n")
        for r in self.search_semantic(question, k=top_k):
            s    = r["similarity"]
            tier = ("HIGH" if s >= self.thresholds["high"]
                    else "MED" if s >= self.thresholds["medium"]
                    else "LOW  ")
            lang = r["metadata"].get("language", "?")
            src  = r["metadata"].get("file_name", "?")[:35]
            print(f"  {s:.1%}  {tier}  [{lang}]  {src}")
            print(f"         → {r['text'][:70].replace(chr(10),' ')}")
        print('='*60)

    def clear_memory(self):
        self.memory.clear()
        print("Memory cleared")

    def _friendly_response(self, answer: str, lang: str, elapsed: float) -> Dict:
        return {
            "answer": answer, "sources": [], "confidence": 1.0,
            "mode": "conversational", "match_type": "conversational",
            "language": lang, "response_time": elapsed,
        }

    def _no_info_response(self, question: str) -> Dict:
        lang = detect_language(question)
        answer = (
            "لم أتمكن من العثور على معلومات كافية حول هذا الموضوع في الوثائق المتاحة.\n\n"
            "يرجى التواصل مع قسم الموارد البشرية أو تكنولوجيا المعلومات للحصول على مساعدة مباشرة."
            if lang == "ar"
            else "I couldn't find enough information on this topic in the available documents.\n\n"
                 "Please contact HR or IT directly for assistance."
        )
        return {
            "answer": answer, "sources": [], "confidence": 0.0,
            "mode": "no_information", "match_type": "none",
            "language": lang, "response_time": 0.0,
        }

    def _api_unavailable_response(self, question: str) -> Dict:
        lang = detect_language(question)
        answer = (
            "خدمة توليد الإجابات غير متاحة حالياً. يرجى المحاولة مرة أخرى لاحقاً."
            if lang == "ar"
            else "AI answer generation is currently unavailable. Please try again later."
        )
        return {
            "answer": answer, "sources": [], "confidence": 0.0,
            "mode": "api_unavailable", "match_type": "error",
            "language": lang, "response_time": 0.0,
        }

    def _chat_unavailable_response(self, question: str) -> Dict:
        lang = detect_language(question)
        answer = (
            "لا أستطيع الدردشة بحرية حالياً، لكن يمكنني مساعدتك في العثور على معلومات السياسات."
            if lang == "ar"
            else "I'm currently unable to chat freely, but I can help you find policy information."
        )
        return {
            "answer": answer, "sources": [], "confidence": 0.0,
            "mode": "general_chat", "match_type": "conversational",
            "language": lang, "response_time": 0.0,
        }

    def _error_response(self, question: str) -> Dict:
        lang = detect_language(question)
        return {
            "answer": ("عذراً، حدث خطأ. يرجى المحاولة مرة أخرى." if lang == "ar"
                       else "Sorry, something went wrong. Please try again."),
            "sources": [], "confidence": 0.0, "mode": "error",
            "match_type": "error", "language": lang, "response_time": 0.0,
        }

    # --------------------------------------------------

    def chat(self):
        print("\n" + "="*60)
        print("AI POD  |  'debug:<q>' | 'clear' | 'quit'")
        print("="*60)
        while True:
            try:
                q = input("\n ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not q: continue
            if q.lower() in {"quit", "exit", "q"}: break
            if q.lower() == "clear": self.clear_memory(); continue
            if q.lower().startswith("debug:"): self.debug_query(q[6:].strip()); continue
            r = self.ask(q)
            print(f"\n{'─'*60}\n{r['answer']}\n{'─'*60}")
            if r["sources"]:
                print(f"Confidence:{r['confidence']:.1%}  Source:{r['sources'][0]['metadata'].get('file_name','?')}  {r['response_time']:.2f}s")


if __name__ == "__main__":
    try:
        AIPodQuerySystem().chat()
    except FileNotFoundError as e:
        print(f"\n {e}")
    except Exception as e:
        import traceback; traceback.print_exc()
# """
# AI POD Query System
# - Friendly conversational responses for general questions
# - Accurate, source-grounded answers for policy questions
# - Arabic + English support
# - Conversation memory
# """

# import os
# import pickle
# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from groq import Groq
# import re
# import time
# from typing import Dict, List
# from config import AIPodConfig


# # -------------------------------------------------------
# # Language Detection
# # -------------------------------------------------------

# def detect_language(text: str) -> str:
#     arabic = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')
#     return "ar" if arabic.search(text) else "en"


# # -------------------------------------------------------
# # Conversational intent detection
# # -------------------------------------------------------

# GREETING_PATTERNS = re.compile(
#     r'^\s*(hi|hello|hey|good\s*(morning|afternoon|evening)|greetings|'
#     r'مرحبا|أهلا|السلام|صباح|مساء|هلا)\b',
#     re.IGNORECASE
# )

# ABOUT_BOT_PATTERNS = re.compile(
#     r'(what (can|will|do) (you|this|the|ai).{0,20}(do|help|assist)|'
#     r'how (can|do) (you|this|the|ai).{0,20}(help|work|assist)|'
#     r'what are you|who are you|tell me about yourself|'
#     r'chatbot will help|what is ai pod|what does ai pod|'
#     r'ماذا تفعل|كيف تساعد|ما هو|من أنت|عن نفسك)',
#     re.IGNORECASE
# )


# def classify_intent(question: str) -> str:
#     """Classify question as: greeting | about_bot | policy_query"""
#     if GREETING_PATTERNS.match(question):
#         return "greeting"
#     if ABOUT_BOT_PATTERNS.search(question):
#         return "about_bot"
#     return "policy_query"


# # -------------------------------------------------------
# # Canned friendly responses
# # -------------------------------------------------------

# GREETING_EN = """ Hello! I'm **AI POD**, your internal assistant at **GIG Egypt Life Takaful**.

# I'm here to help you find answers about company policies quickly and accurately. What would you like to know?"""

# GREETING_AR = """ مرحباً! أنا **AI POD**، مساعدك الداخلي في شركة **GIG مصر للتكافل على الحياة**.

# أنا هنا لمساعدتك في العثور على إجابات حول سياسات الشركة بسرعة ودقة. بماذا يمكنني مساعدتك؟"""

# ABOUT_EN = """ I'm **AI POD** — the internal AI assistant for **GIG Egypt Life Takaful**.

# Here's what I can help you with:

#  **HR Policies** — Leave entitlements, attendance, disciplinary procedures, performance reviews
#  **IT Policies** — Password rules, acceptable use, data security, remote access
#  **Company Procedures** — Step-by-step guidance on internal processes

# **How to use me:**
# - Ask me any policy question in **English or Arabic**
# - I'll search the official company documents and give you a cited answer
# - If I can't find the answer, I'll tell you honestly

# **Examples:**
# - *"How many annual leave days am I entitled to?"*
# - *"ما هي سياسة كلمة المرور؟"*
# - *"What is the remote work policy?"*

# What would you like to know? """

# ABOUT_AR = """ أنا **AI POD** — المساعد الذكي الداخلي لشركة **GIG مصر للتكافل على الحياة**.

# إليك ما يمكنني مساعدتك فيه:

#  **سياسات الموارد البشرية** — الإجازات، الحضور، الإجراءات التأديبية، تقييم الأداء
#  **سياسات تكنولوجيا المعلومات** — قواعد كلمة المرور، الاستخدام المقبول، أمن البيانات، العمل عن بُعد
#  **إجراءات الشركة** — إرشادات خطوة بخطوة للعمليات الداخلية

# **كيفية استخدامي:**
# - اسألني أي سؤال يتعلق بالسياسات **بالعربية أو الإنجليزية**
# - سأبحث في وثائق الشركة الرسمية وأعطيك إجابة موثقة
# - إذا لم أتمكن من إيجاد الإجابة، سأخبرك بصدق

# ما الذي تود معرفته؟ """


# # -------------------------------------------------------
# # Conversation Memory
# # -------------------------------------------------------

# class ConversationMemory:
#     MAX_TURNS = 6
#     MAX_CHARS = 3000

#     def __init__(self):
#         self.turns: List[Dict[str, str]] = []

#     def add(self, question: str, answer: str):
#         self.turns.append({"role": "user",      "content": question})
#         self.turns.append({"role": "assistant", "content": answer})
#         if len(self.turns) > self.MAX_TURNS * 2:
#             self.turns = self.turns[-(self.MAX_TURNS * 2):]

#     def get_messages(self) -> List[Dict[str, str]]:
#         messages, total = [], 0
#         for turn in reversed(self.turns):
#             n = len(turn["content"])
#             if total + n > self.MAX_CHARS:
#                 break
#             messages.insert(0, turn)
#             total += n
#         return messages

#     def clear(self):
#         self.turns = []

#     def is_empty(self):
#         return len(self.turns) == 0

#     def summary(self):
#         return f"{len(self.turns) // 2} prior turn(s)"


# # -------------------------------------------------------
# # Query System
# # -------------------------------------------------------

# class AIPodQuerySystem:

#     def __init__(self):
#         print(" Loading AI POD...")

#         if not os.path.exists(AIPodConfig.FAISS_INDEX_PATH):
#             raise FileNotFoundError("Index not found — run: python ingest_documents.py")

#         self.index = faiss.read_index(AIPodConfig.FAISS_INDEX_PATH)

#         with open(AIPodConfig.CHUNKS_PATH, "rb") as f:
#             self.chunks = pickle.load(f)

#         with open(AIPodConfig.METADATA_PATH, "rb") as f:
#             self.metadata = pickle.load(f)

#         print(f" Loading embedding model: {AIPodConfig.EMBEDDING_MODEL}")
#         self.embed_model = SentenceTransformer(AIPodConfig.EMBEDDING_MODEL)

#         self.thresholds = self._load_thresholds()
#         self.groq_client = self._init_groq()
#         self.memory = ConversationMemory()
#         self._print_info()

#     # --------------------------------------------------

#     def _load_thresholds(self) -> Dict:
#         # Sensible defaults for IndexFlatIP with normalised vectors
#         defaults = {"high": 0.40, "medium": 0.28, "low": 0.18}
#         try:
#             if os.path.exists(AIPodConfig.THRESHOLDS_PATH):
#                 with open(AIPodConfig.THRESHOLDS_PATH, "rb") as f:
#                     t = pickle.load(f)
#                 if t.get("high", 0) > t.get("medium", 0) > t.get("low", 0) > 0:
#                     print(f" Thresholds: high={t['high']:.1%} medium={t['medium']:.1%} low={t['low']:.1%}")
#                     return t
#                 print(f"  Saved thresholds invalid ({t}) — using defaults")
#         except Exception as e:
#             print(f"  Threshold load error: {e}")
#         print(f"  Using defaults: {defaults}")
#         return defaults

#     def _init_groq(self):
#         if not AIPodConfig.GROQ_API_KEY:
#             print("  GROQ_API_KEY not set — add to .env")
#             return None
#         try:
#             client = Groq(api_key=AIPodConfig.GROQ_API_KEY)
#             client.chat.completions.create(
#                 model=AIPodConfig.LLM_MODEL_FAST,
#                 messages=[{"role": "user", "content": "ping"}],
#                 max_tokens=1,
#             )
#             print(" Groq connected")
#             return client
#         except Exception as e:
#             print(f"  Groq error: {e}")
#             return None

#     def _print_info(self):
#         langs = {}
#         for c in self.chunks:
#             l = c.get("metadata", {}).get("language", "?")
#             langs[l] = langs.get(l, 0) + 1
#         print(f"\n {len(self.chunks)} chunks | {dict(langs)}")
#         print(f" Thresholds — high:{self.thresholds['high']:.1%}  med:{self.thresholds['medium']:.1%}  low:{self.thresholds['low']:.1%}\n")

#     # --------------------------------------------------

#     def _embed(self, text: str) -> np.ndarray:
#         return self.embed_model.encode(
#             [text], convert_to_numpy=True, normalize_embeddings=True
#         ).astype("float32")

#     def search_semantic(self, query: str, k: int = None) -> List[Dict]:
#         if k is None:
#             k = AIPodConfig.DEFAULT_SEARCH_K
#         vec = self._embed(query)
#         scores, indices = self.index.search(vec, k)
#         results = []
#         for i, idx in enumerate(indices[0]):
#             if idx < 0 or idx >= len(self.chunks):
#                 continue
#             cos_sim = max(0.0, float(scores[0][i]))
#             results.append({
#                 "text":       self.chunks[idx].get("text", ""),
#                 "similarity": cos_sim,
#                 "metadata":   self.chunks[idx].get("metadata", {}),
#                 "chunk_id":   int(idx),
#             })
#         results.sort(key=lambda x: x["similarity"], reverse=True)
#         return results

#     # --------------------------------------------------

#     def _call_groq_policy(self, question: str, chunks: List[Dict]) -> str:
#         """
#         Answer a policy question strictly from document chunks.
#         Friendly tone but answers are grounded ONLY in the provided sources.
#         """
#         lang = detect_language(question)
#         context = "\n\n---\n\n".join([
#             f"[{'من' if lang=='ar' else 'From'}: {c['metadata'].get('file_name','Policy')}]\n{c['text']}"
#             for c in chunks[:AIPodConfig.MAX_CHUNKS_PER_QUERY]
#         ])

#         if lang == "ar":
#             system = (
#                 f"أنت AI POD، المساعد الذكي الودود لشركة {AIPodConfig.COMPANY_NAME}.\n\n"
#                 "مهمتك: الإجابة على أسئلة الموظفين بطريقة واضحة وودية ومفيدة.\n\n"
#                 "القواعد الصارمة:\n"
#                 "1. استخدم المعلومات الموجودة في الوثائق المقدمة فقط\n"
#                 "2. قدّم الإجابة بأسلوب واضح ومنظم (استخدم نقاط أو أرقام إذا كان مناسباً)\n"
#                 "3. اذكر اسم المستند المصدر\n"
#                 "4. إذا لم تجد الإجابة في الوثائق، قل: 'لم أجد معلومات كافية حول هذا الموضوع في الوثائق المتاحة، يرجى التواصل مع قسم الموارد البشرية'\n"
#                 "5. استخدم سياق المحادثة السابقة لفهم الأسئلة المتابعة\n"
#                 "6. أجب دائماً بالعربية بأسلوب احترافي وودي\n"
#                 "7. أنهِ الإجابات المهمة بـ: ' يرجى التحقق من الوثائق الرسمية للتأكيد.'"
#             )
#             user_msg = (
#                 f"الوثائق المتاحة:\n\n{context}\n\n"
#                 f"سؤال الموظف: {question}\n\n"
#                 "الإجابة (واضحة، منظمة، ومن الوثائق فقط):"
#             )
#         else:
#             system = (
#                 f"You are AI POD, the friendly internal AI assistant for {AIPodConfig.COMPANY_NAME}.\n\n"
#                 "Your job: Answer employee policy questions in a clear, helpful, and friendly way.\n\n"
#                 "STRICT RULES:\n"
#                 "1. Answer ONLY from the provided document excerpts\n"
#                 "2. Present answers clearly — use bullet points or numbered lists when helpful\n"
#                 "3. Always mention which document the information comes from\n"
#                 "4. If the answer is not in the documents, say: 'I couldn't find enough information on this topic in the available documents. Please contact HR directly.'\n"
#                 "5. Use prior conversation context for follow-up questions\n"
#                 "6. Be professional yet warm and approachable\n"
#                 "7. End important policy answers with: ' Please verify with official documentation.'"
#             )
#             user_msg = (
#                 f"Available documents:\n\n{context}\n\n"
#                 f"Employee question: {question}\n\n"
#                 "Answer (clear, well-structured, from documents only):"
#             )

#         messages = [{"role": "system", "content": system}]
#         messages.extend(self.memory.get_messages())
#         messages.append({"role": "user", "content": user_msg})

#         try:
#             resp = self.groq_client.chat.completions.create(
#                 model=AIPodConfig.LLM_MODEL_FAST,
#                 messages=messages,
#                 temperature=0.2,
#                 max_tokens=800,
#             )
#             return resp.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"  Groq error: {e}")
#             return f"[{chunks[0]['metadata'].get('file_name','Policy')}]\n\n{chunks[0]['text'][:600]}"

#     def _call_groq_general(self, question: str) -> str:
#         """
#         Handle general / conversational questions with a friendly response.
#         No document grounding needed.
#         """
#         lang = detect_language(question)
#         if lang == "ar":
#             system = (
#                 f"أنت AI POD، المساعد الذكي الودود لشركة {AIPodConfig.COMPANY_NAME}.\n"
#                 "أجب على الأسئلة العامة بأسلوب ودي ومفيد. "
#                 "إذا سألوا عن الشركة أو سياساتها، شجعهم على طرح سؤال محدد."
#             )
#         else:
#             system = (
#                 f"You are AI POD, the friendly internal AI assistant for {AIPodConfig.COMPANY_NAME}.\n"
#                 "Answer general questions warmly and helpfully. "
#                 "If they ask about company policies, encourage them to ask a specific question."
#             )

#         messages = [{"role": "system", "content": system}]
#         messages.extend(self.memory.get_messages())
#         messages.append({"role": "user", "content": question})

#         try:
#             resp = self.groq_client.chat.completions.create(
#                 model=AIPodConfig.LLM_MODEL_FAST,
#                 messages=messages,
#                 temperature=0.5,
#                 max_tokens=400,
#             )
#             return resp.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"  Groq error: {e}")
#             return ABOUT_EN if detect_language(question) == "en" else ABOUT_AR

#     # --------------------------------------------------

#     def ask(self, question: str) -> Dict:
#         start = time.time()
#         question = question.strip()
#         if not question:
#             return self._error_response(question)

#         lang = detect_language(question)
#         intent = classify_intent(question)

#         # ── Greeting ──────────────────────────────────────────────
#         if intent == "greeting":
#             answer = GREETING_AR if lang == "ar" else GREETING_EN
#             self.memory.add(question, answer)
#             return self._friendly_response(answer, lang, time.time() - start)

#         # ── About the bot ─────────────────────────────────────────
#         if intent == "about_bot":
#             answer = ABOUT_AR if lang == "ar" else ABOUT_EN
#             self.memory.add(question, answer)
#             return self._friendly_response(answer, lang, time.time() - start)

#         # ── Policy question — semantic search ─────────────────────
#         try:
#             results = self.search_semantic(question)
#             if not results:
#                 return self._no_info_response(question)

#             best = results[0]["similarity"]
#             top  = results[:AIPodConfig.TOP_K_RESULTS]

#             print(f" '{question[:50]}' | best={best:.1%} | high={self.thresholds['high']:.1%} | med={self.thresholds['medium']:.1%}")

#             if best >= self.thresholds["high"]:
#                 if self.groq_client:
#                     answer = self._call_groq_policy(question, top)
#                 else:
#                     answer = f" **{top[0]['metadata'].get('file_name', 'Policy')}**\n\n{top[0]['text'][:600]}"
#                 self.memory.add(question, answer)
#                 return {
#                     "answer": answer, "sources": top, "confidence": best,
#                     "mode": "ai_enhanced", "match_type": "exact_match",
#                     "language": lang, "response_time": time.time() - start,
#                 }

#             elif best >= self.thresholds["medium"]:
#                 if self.groq_client:
#                     answer = self._call_groq_policy(question, top)
#                     answer += (
#                         "\n\n ملاحظة: قد لا تكون هذه المعلومات إجابة مباشرة لسؤالك."
#                         if lang == "ar"
#                         else "\n\n Note: These results are related but may not directly answer your question."
#                     )
#                 else:
#                     answer = top[0]["text"][:600]
#                 self.memory.add(question, answer)
#                 return {
#                     "answer": answer, "sources": top, "confidence": best,
#                     "mode": "related_match", "match_type": "related_match",
#                     "language": lang, "response_time": time.time() - start,
#                 }

#             else:
#                 # Nothing found in documents — try Groq as general fallback
#                 if self.groq_client:
#                     answer = self._call_groq_general(question)
#                     self.memory.add(question, answer)
#                     return {
#                         "answer": answer, "sources": [], "confidence": best,
#                         "mode": "general_fallback", "match_type": "none",
#                         "language": lang, "response_time": time.time() - start,
#                     }
#                 return self._no_info_response(question)

#         except Exception as e:
#             print(f" ask() error: {e}")
#             import traceback; traceback.print_exc()
#             return self._error_response(question)

#     # --------------------------------------------------

#     def debug_query(self, question: str, top_k: int = 7):
#         print(f"\n{'='*60}")
#         print(f" DEBUG: '{question}'")
#         print(f"   Intent    : {classify_intent(question)}")
#         print(f"   Language  : {detect_language(question)}")
#         print(f"   Model     : {AIPodConfig.EMBEDDING_MODEL}")
#         print(f"   Thresholds: high={self.thresholds['high']:.1%}  med={self.thresholds['medium']:.1%}\n")
#         for r in self.search_semantic(question, k=top_k):
#             s    = r["similarity"]
#             tier = ("HIGH " if s >= self.thresholds["high"]
#                     else "MED  " if s >= self.thresholds["medium"]
#                     else "LOW  ")
#             lang = r["metadata"].get("language", "?")
#             src  = r["metadata"].get("file_name", "?")[:35]
#             print(f"  {s:.1%}  {tier}  [{lang}]  {src}")
#             print(f"         → {r['text'][:70].replace(chr(10),' ')}")
#         print('='*60)

#     def clear_memory(self):
#         self.memory.clear()
#         print("  Memory cleared")

#     def _friendly_response(self, answer: str, lang: str, elapsed: float) -> Dict:
#         return {
#             "answer": answer, "sources": [], "confidence": 1.0,
#             "mode": "conversational", "match_type": "conversational",
#             "language": lang, "response_time": elapsed,
#         }

#     def _no_info_response(self, question: str) -> Dict:
#         lang = detect_language(question)
#         answer = (
#             "لم أتمكن من العثور على معلومات كافية حول هذا الموضوع في الوثائق المتاحة.\n\n"
#             " يرجى التواصل مع قسم الموارد البشرية أو تكنولوجيا المعلومات للحصول على مساعدة مباشرة."
#             if lang == "ar"
#             else "I couldn't find enough information on this topic in the available documents.\n\n"
#                  " Please contact HR or IT directly for assistance."
#         )
#         return {
#             "answer": answer, "sources": [], "confidence": 0.0,
#             "mode": "no_information", "match_type": "none",
#             "language": lang, "response_time": 0.0,
#         }

#     def _error_response(self, question: str) -> Dict:
#         lang = detect_language(question)
#         return {
#             "answer": ("عذراً، حدث خطأ. يرجى المحاولة مرة أخرى." if lang == "ar"
#                        else "Sorry, something went wrong. Please try again."),
#             "sources": [], "confidence": 0.0, "mode": "error",
#             "match_type": "error", "language": lang, "response_time": 0.0,
#         }

#     # --------------------------------------------------

#     def chat(self):
#         print("\n" + "="*60)
#         print(" AI POD  |  'debug:<q>' | 'clear' | 'quit'")
#         print("="*60)
#         while True:
#             try:
#                 q = input("\n ").strip()
#             except (KeyboardInterrupt, EOFError):
#                 break
#             if not q: continue
#             if q.lower() in {"quit", "exit", "q"}: break
#             if q.lower() == "clear": self.clear_memory(); continue
#             if q.lower().startswith("debug:"): self.debug_query(q[6:].strip()); continue
#             r = self.ask(q)
#             print(f"\n{'─'*60}\n{r['answer']}\n{'─'*60}")
#             if r["sources"]:
#                 print(f"Confidence:{r['confidence']:.1%}  Source:{r['sources'][0]['metadata'].get('file_name','?')}  {r['response_time']:.2f}s")


# if __name__ == "__main__":
#     try:
#         AIPodQuerySystem().chat()
#     except FileNotFoundError as e:
#         print(f"\n {e}")
#     except Exception as e:
#         import traceback; traceback.print_exc()
