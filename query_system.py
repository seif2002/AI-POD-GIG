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
    arabic = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')
    return "ar" if arabic.search(text) else "en"


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


def classify_intent(question: str) -> str:
    """Classify question as: greeting | about_bot | policy_query"""
    if GREETING_PATTERNS.match(question):
        return "greeting"
    if ABOUT_BOT_PATTERNS.search(question):
        return "about_bot"
    return "policy_query"


# -------------------------------------------------------
# Canned friendly responses
# -------------------------------------------------------

GREETING_EN = """Hello! I'm **AI POD**, your internal assistant at **GIG Egypt Life Takaful**.

I'm here to help you find answers about company policies quickly and accurately. What would you like to know?"""

GREETING_AR = """مرحباً! أنا **AI POD**، مساعدك الداخلي في شركة **GIG مصر للتكافل على الحياة**.

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

ABOUT_AR = """أنا **AI POD** — المساعد الذكي الداخلي لشركة **GIG مصر للتكافل على الحياة**.

إليك ما يمكنني مساعدتك فيه:

**سياسات الموارد البشرية** — الإجازات، الحضور، الإجراءات التأديبية، تقييم الأداء
**سياسات تكنولوجيا المعلومات** — قواعد كلمة المرور، الاستخدام المقبول، أمن البيانات، العمل عن بُعد
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
        vec = self._embed(query)
        scores, indices = self.index.search(vec, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            cos_sim = max(0.0, float(scores[0][i]))
            results.append({
                "text":       self.chunks[idx].get("text", ""),
                "similarity": cos_sim,
                "metadata":   self.chunks[idx].get("metadata", {}),
                "chunk_id":   int(idx),
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    # --------------------------------------------------

    def _call_groq_policy(self, question: str, chunks: List[Dict], answer_style: str = "summary") -> str:
        """
        Answer a policy question strictly from document chunks.
        Friendly tone but answers are grounded ONLY in the provided sources.
        """
        lang = detect_language(question)
        context = "\n\n---\n\n".join([
            f"[{'من' if lang=='ar' else 'From'}: {c['metadata'].get('file_name','Policy')}]\n{c['text']}"
            for c in chunks[:AIPodConfig.MAX_CHUNKS_PER_QUERY]
        ])

        # CRITICAL: Response language is ALWAYS determined by the question language,
        # NOT by the language of the source documents retrieved.
        reply_lang_instruction = (
            "CRITICAL: You MUST reply in Arabic regardless of the document language."
            if lang == "ar"
            else "CRITICAL: You MUST reply in English regardless of the document language. "
                 "The source documents may be in Arabic — translate the relevant information into English."
        )
        style = "detailed" if answer_style == "detailed" else "summary"
        style_instruction = (
            "RESPONSE STYLE: Detailed. Use this exact structure:\n"
            "Policy Title\n"
            "- Use a short title if the question covers a policy area, for example 'Leave and Time-Off Policy'.\n\n"
            "Details:\n"
            "- Use clear section headings such as Annual Leave, Sick Leave, Weekly Off Days, Eligibility, Process, Exceptions.\n"
            "- If the answer compares entitlements by years of service, use a markdown table with columns such as 'Years of Service' and 'Annual Leave Entitlement'.\n"
            "- Put conditions under a bold/clear subheading such as Additional Conditions: or Conditions:.\n"
            "- Use bullets for conditions and important notes.\n"
            "- Keep paragraphs short and readable.\n\n"
            "Source:\n"
            "- Mention the document name(s).\n"
            "Do not start with phrases like 'Based on the provided documents'."
            if style == "detailed"
            else "RESPONSE STYLE: Summary. Use this exact structure:\n"
                 "Summary:\n"
                 "- Give 3-5 concise bullets only. Start each bullet with a bold label, for example **Annual Leave:** normal explanatory text.\n\n"
                 "Source:\n"
                 "- Mention the document name(s).\n"
                 "Do not start with phrases like 'Based on the provided documents'."
        )

        system = (
            f"You are AI POD, the friendly internal AI assistant for {AIPodConfig.COMPANY_NAME}.\n\n"
            "Your job: Answer employee policy questions in a clear, helpful, and friendly way.\n\n"
            f"{reply_lang_instruction}\n\n"
            f"{style_instruction}\n\n"
            "STRICT RULES:\n"
            "1. Answer ONLY from the provided document excerpts\n"
            "2. Present answers in the requested structure with short headings and clean bullets\n"
            "3. Use normal readable text after each bold label. Do not wrap whole sentences in bold.\n"
            "4. Do NOT put multiple policies in one long bullet. Use separate headings and separate bullets.\n"
            "5. Use markdown tables when the source contains tiered values, levels, years, or categories.\n"
            "6. Always mention which document the information comes from\n"
            "7. If the answer is not in the documents, say so and suggest contacting HR directly\n"
            "8. Use prior conversation context for follow-up questions\n"
            "9. Be professional yet warm and approachable\n"
            "10. Avoid long paragraphs, duplicated source names inside every bullet, and generic preambles\n"
            "11. End important policy answers with the appropriate disclaimer:\n"
            "   - English: 'Please verify with official documentation.'\n"
            "   - Arabic: 'يرجى التحقق من الوثائق الرسمية للتأكيد.'"
        )

        user_msg = (
            f"Available documents (may be in Arabic or English — extract and translate as needed):\n\n"
            f"{context}\n\n"
            f"Employee question: {question}\n\n"
            f"Answer ({style}, {'in English' if lang == 'en' else 'in Arabic'}, clear, well-structured, from documents only):"
        )

        messages = [{"role": "system", "content": system}]
        messages.extend(self.memory.get_messages())
        messages.append({"role": "user", "content": user_msg})

        try:
            resp = self._groq_completion(
                model=AIPodConfig.LLM_MODEL_FAST,
                messages=messages,
                temperature=0.2,
                max_tokens=1200 if style == "detailed" else 550,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq error: {e}")
            return f"[{chunks[0]['metadata'].get('file_name','Policy')}]\n\n{chunks[0]['text'][:600]}"

    def _call_groq_general(self, question: str) -> str:
        """
        Handle general / conversational questions with a friendly response.
        No document grounding needed.
        """
        lang = detect_language(question)
        if lang == "ar":
            system = (
                f"أنت AI POD، المساعد الذكي الودود لشركة {AIPodConfig.COMPANY_NAME}.\n"
                "أجب على الأسئلة العامة بأسلوب ودي ومفيد. "
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
            resp = self._groq_completion(
                model=AIPodConfig.LLM_MODEL_FAST,
                messages=messages,
                temperature=0.5,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq error: {e}")
            return ABOUT_EN if detect_language(question) == "en" else ABOUT_AR

    # --------------------------------------------------

    def ask(self, question: str, answer_style: str = "summary") -> Dict:
        start = time.time()
        question = question.strip()
        if not question:
            return self._error_response(question)

        lang = detect_language(question)
        intent = classify_intent(question)

        # ── Greeting ──────────────────────────────────────────────
        if intent == "greeting":
            answer = GREETING_AR if lang == "ar" else GREETING_EN
            self.memory.add(question, answer)
            return self._friendly_response(answer, lang, time.time() - start)

        # ── About the bot ─────────────────────────────────────────
        if intent == "about_bot":
            answer = ABOUT_AR if lang == "ar" else ABOUT_EN
            self.memory.add(question, answer)
            return self._friendly_response(answer, lang, time.time() - start)

        # ── Policy question — semantic search ─────────────────────
        try:
            results = self.search_semantic(question)
            if not results:
                return self._no_info_response(question)

            best = results[0]["similarity"]
            top  = results[:AIPodConfig.TOP_K_RESULTS]

            print(f"'{question[:50]}' | best={best:.1%} | high={self.thresholds['high']:.1%} | med={self.thresholds['medium']:.1%}")

            if best >= self.thresholds["high"]:
                if self.groq_client:
                    answer = self._call_groq_policy(question, top, answer_style)
                else:
                    answer = f"**{top[0]['metadata'].get('file_name', 'Policy')}**\n\n{top[0]['text'][:600]}"
                self.memory.add(question, answer)
                return {
                    "answer": answer, "sources": top, "confidence": best,
                    "mode": "ai_enhanced", "match_type": "exact_match",
                    "language": lang, "response_time": time.time() - start,
                }

            elif best >= self.thresholds["medium"]:
                if self.groq_client:
                    answer = self._call_groq_policy(question, top, answer_style)
                    answer += (
                        "\n\nملاحظة: قد لا تكون هذه المعلومات إجابة مباشرة لسؤالك."
                        if lang == "ar"
                        else "\n\nNote: These results are related but may not directly answer your question."
                    )
                else:
                    answer = top[0]["text"][:600]
                self.memory.add(question, answer)
                return {
                    "answer": answer, "sources": top, "confidence": best,
                    "mode": "related_match", "match_type": "related_match",
                    "language": lang, "response_time": time.time() - start,
                }

            else:
                # Nothing found in documents — try Groq as general fallback
                if self.groq_client:
                    answer = self._call_groq_general(question)
                    self.memory.add(question, answer)
                    return {
                        "answer": answer, "sources": [], "confidence": best,
                        "mode": "general_fallback", "match_type": "none",
                        "language": lang, "response_time": time.time() - start,
                    }
                return self._no_info_response(question)

        except Exception as e:
            print(f"ask() error: {e}")
            import traceback; traceback.print_exc()
            return self._error_response(question)

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
