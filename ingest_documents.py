"""
AI POD Document Ingestion — Enhanced with pymupdf
Pure semantic chunking, no keyword dependencies.
Switched from PyPDF2 → pymupdf (fitz) for better Arabic text extraction.
"""

import os
import pickle
import hashlib
import datetime
import random
from typing import Dict, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

# ── PDF extraction: pymupdf (fitz) replaces PyPDF2 ──────────────────────────
# pymupdf handles:
#   • Arabic / RTL text correctly (proper Unicode, correct reading order)
#   • Complex multi-column layouts
#   • Embedded fonts that PyPDF2 cannot decode
# Install: pip install pymupdf
try:
    import fitz  # pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("  pymupdf not installed — PDF extraction will be skipped.")
    print("    Run: pip install pymupdf")

from config import AIPodConfig


# ============================================================
# Document Processor
# ============================================================

class DocumentProcessor:
    """Process documents and build FAISS index — pure semantic approach."""

    def __init__(self):
        AIPodConfig.create_directories()

        print(" Loading embedding model...")
        self.embedder = SentenceTransformer(AIPodConfig.EMBEDDING_MODEL)

        self.chunks: List[Dict] = []
        self.total_files = 0

        print(" Document Processor initialised")
        print(f"   Source dir    : {AIPodConfig.DOCUMENTS_DIR}")
        print(f"   Chunk size    : {AIPodConfig.CHUNK_SIZE} chars")
        print(f"   Chunk overlap : {AIPodConfig.CHUNK_OVERLAP} chars")
        print(f"   Embedding     : {AIPodConfig.EMBEDDING_MODEL}")
        print(f"   PDF backend   : {'pymupdf (fitz)' if PYMUPDF_AVAILABLE else 'UNAVAILABLE'}")

    # --------------------------------------------------------

    def get_all_documents(self) -> List[str]:
        """
        Return all .txt and .pdf files from the documents directory.
        Uses os.scandir() which handles Unicode / Arabic filenames correctly
        on Windows, unlike os.listdir() which can silently skip them.
        Files are sorted by modification time (oldest first) so newly added
        files are always processed.
        """
        all_files = []
        try:
            with os.scandir(AIPodConfig.DOCUMENTS_DIR) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    # .lower() on entry.name works correctly for Unicode names
                    if entry.name.lower().endswith((".txt", ".pdf")):
                        all_files.append((entry.path, entry.stat().st_mtime))
                        # Print name safely — encode to utf-8 then back to handle
                        # any console that can't display Arabic
                        try:
                            print(f"    Found: {entry.name}")
                        except UnicodeEncodeError:
                            print(f"    Found: [Arabic filename] {entry.path}")
        except PermissionError as e:
            print(f"    Cannot read documents directory: {e}")
            return []

        # Sort by modification time so nothing is missed regardless of order
        all_files.sort(key=lambda x: x[1])
        paths = [p for p, _ in all_files]
        self.total_files = len(paths)
        print(f"    Total eligible files: {self.total_files}")
        return paths

    # --------------------------------------------------------
    # PDF extraction — pymupdf
    # --------------------------------------------------------

    def extract_text_from_pdf(self, path: str) -> str:
        """
        Extract text using pymupdf with Arabic presentation-form normalisation.
        Many Arabic PDFs store text as isolated glyph presentation forms (U+FE70-FEFF)
        with spaces between every character. We normalise these to proper Arabic Unicode
        so the embedding model can understand them.
        """
        if not PYMUPDF_AVAILABLE:
            print(f"    pymupdf not available — skipping {os.path.basename(path)}")
            return ""

        try:
            text_parts = []
            with fitz.open(path) as doc:
                for page_num, page in enumerate(doc):
                    # Use "rawdict" to get raw character data, then reconstruct
                    # This handles RTL and presentation-form Arabic better
                    page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
                    if page_text.strip():
                        text_parts.append(page_text)

            full_text = "\n".join(text_parts)

            # Normalise Arabic presentation forms to standard Unicode
            # Many scanned/old Arabic PDFs use isolated/initial/medial/final forms
            full_text = self._normalise_arabic(full_text)

            print(f"    pymupdf extracted {len(full_text):,} chars from {os.path.basename(path)}")
            return full_text

        except Exception as e:
            print(f"    pymupdf error reading {os.path.basename(path)}: {e}")
            return ""

    def _normalise_arabic(self, text: str) -> str:
        """
        Normalise Arabic text extracted from PDFs:
        1. Remove zero-width characters injected between letters
        2. Normalise Unicode to NFC (composes ligatures into single codepoints)
        3. Collapse runs of whitespace inside Arabic words
        """
        import unicodedata

        # Remove zero-width non-joiner, zero-width joiner, and other invisible chars
        text = re.sub(r'[​-‏‪-‮﻿]', '', text)

        # NFC normalisation: compose ligatures and diacritics
        text = unicodedata.normalize("NFC", text)

        # Fix the most common garbling pattern: single Arabic letters separated by spaces
        # e.g. "ا ل س ي ا س ة" → "السياسة"
        # Detect runs of single Arabic letters separated by spaces and rejoin them
        arabic_letter = r'[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]'
        # Match 4+ consecutive (single Arabic letter + space) and remove the spaces
        def rejoin(m):
            return m.group(0).replace(' ', '')
        text = re.sub(f'(?:{arabic_letter} ){{4,}}{arabic_letter}', rejoin, text)

        return text

    # --------------------------------------------------------

    def extract_text(self, path: str) -> str:
        """Dispatch text extraction by file type."""
        if path.lower().endswith(".pdf"):
            return self.extract_text_from_pdf(path)
        # .txt
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"    Error reading {os.path.basename(path)}: {e}")
            return ""

    # --------------------------------------------------------

    def split_by_language(self, text: str, filename: str = "") -> Dict[str, str]:
        """
        Split text into language sections.
        Handles (1) filename hints, (2) inline markers, (3) fallback.
        """
        sections: Dict[str, str] = {}
        filename_lower = filename.lower()

        # Case 1: Filename contains language hint
        if any(k in filename_lower for k in ("english", "_en_", "-en-", "_eng_")):
            print("    Language inferred from filename: English")
            sections["en"] = text
            return sections
        if any(k in filename_lower for k in ("arabic", "_ar_", "-ar-", "_ara_")):
            print("    Language inferred from filename: Arabic")
            sections["ar"] = text
            return sections

        # Case 2: Inline markers
        en_start = text.find("=== LANGUAGE: EN ===")
        ar_start = text.find("=== LANGUAGE: AR ===")

        if en_start >= 0 and ar_start >= 0:
            if en_start < ar_start:
                en_content = text[en_start + 20: ar_start].strip()
                ar_content = text[ar_start + 20:].strip()
            else:
                ar_content = text[ar_start + 20: en_start].strip()
                en_content = text[en_start + 20:].strip()
            if en_content:
                sections["en"] = en_content
            if ar_content:
                sections["ar"] = ar_content
        elif en_start >= 0:
            sections["en"] = text[en_start + 20:].strip()
        elif ar_start >= 0:
            sections["ar"] = text[ar_start + 20:].strip()
        else:
            # Case 3: No hints — check character set majority
            arabic_chars = len(re.findall(
                r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]',
                text
            ))
            total_alpha = len(re.findall(r'[A-Za-z\u0600-\u06FF]', text))
            if total_alpha > 0 and arabic_chars / total_alpha > 0.5:
                print(f"     No markers — auto-detected Arabic (rename file to include 'arabic' to be explicit)")
                sections["ar"] = text
            else:
                print(f"     No markers — defaulting to English (rename file to include 'english' if wrong)")
                sections["en"] = text

        return sections

    # --------------------------------------------------------
    # Semantic chunking WITH overlap
    # --------------------------------------------------------

    def chunk_text_semantic(self, text: str, base_metadata: Dict) -> List[Dict]:
        """
        Semantic chunking that respects paragraph/sentence boundaries AND
        implements CHUNK_OVERLAP so no context is lost at chunk edges.
        """
        if not text.strip():
            return []

        chunk_size    = AIPodConfig.CHUNK_SIZE
        chunk_overlap = AIPodConfig.CHUNK_OVERLAP

        # ── Step 1: collect all sentence-level units ──────────────────────
        units: List[str] = []
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if len(para) > chunk_size:
                # Split long paragraphs at sentence boundaries
                for sentence in re.split(r'(?<=[.!?])\s+', para):
                    sentence = sentence.strip()
                    if sentence:
                        units.append(sentence)
            else:
                units.append(para)

        # ── Step 2: pack units into chunks with overlap ────────────────────
        chunks: List[Dict] = []
        current_units: List[str] = []
        current_size = 0

        for unit in units:
            unit_len = len(unit)

            if current_size + unit_len > chunk_size and current_units:
                # Emit chunk
                chunk_text = " ".join(current_units)
                chunks.append(self._build_chunk(chunk_text, base_metadata))

                # ── Overlap: carry forward enough trailing units to fill
                # the overlap window so context bridges chunk boundaries ──
                overlap_units: List[str] = []
                overlap_size = 0
                for prev_unit in reversed(current_units):
                    if overlap_size + len(prev_unit) > chunk_overlap:
                        break
                    overlap_units.insert(0, prev_unit)
                    overlap_size += len(prev_unit)

                current_units = overlap_units + [unit]
                current_size  = overlap_size + unit_len
            else:
                current_units.append(unit)
                current_size += unit_len

        # Emit final chunk
        if current_units:
            chunks.append(self._build_chunk(" ".join(current_units), base_metadata))

        return chunks

    # --------------------------------------------------------

    def _build_chunk(self, text: str, metadata: Dict) -> Dict:
        meta = metadata.copy()
        meta.update({
            "chunk_id":   len(self.chunks),
            "chunk_hash": hashlib.md5(text.encode()).hexdigest()[:8],
            "char_length": len(text),
            "word_count": len(text.split()),
            "created_at": datetime.datetime.now().isoformat(),
        })
        return {"text": text, "metadata": meta}

    # --------------------------------------------------------

    def process_all_files(self) -> bool:
        all_files = self.get_all_documents()
        if not all_files:
            print(" No files found in documents directory")
            return False

        print(f"\n Found {len(all_files)} file(s)")
        total_chunks = 0
        start_time   = time.time()

        for file_idx, file_path in enumerate(all_files, 1):
            file_name = os.path.basename(file_path)
            file_ext  = os.path.splitext(file_name)[1].lower()

            print(f"\n[{file_idx}/{len(all_files)}]  Processing: {file_name}")

            raw_text = self.extract_text(file_path)
            if not raw_text:
                continue

            sections = self.split_by_language(raw_text, file_name)
            if not sections:
                print("     No language sections found")
                continue

            print(f"    Languages: {list(sections.keys())}")

            base_metadata = {
                "department":  "Multi-Department",
                "document":    file_name.replace(".txt", "").replace(".pdf", ""),
                "file_name":   file_name,
                "file_type":   file_ext[1:],
                "ingested_at": datetime.datetime.now().isoformat(),
                "source":      "Internal Documents",
            }

            file_chunks = 0
            for lang, content in sections.items():
                if not content.strip():
                    continue
                lang_meta          = base_metadata.copy()
                lang_meta["language"] = lang
                lang_chunks        = self.chunk_text_semantic(content, lang_meta)
                if lang_chunks:
                    self.chunks.extend(lang_chunks)
                    file_chunks += len(lang_chunks)
                    print(f"   • {lang.upper()}: {len(lang_chunks)} chunks")

            total_chunks += file_chunks
            print(f"    Added: {file_chunks} chunks")

            elapsed   = time.time() - start_time
            avg_time  = elapsed / file_idx
            remaining = avg_time * (len(all_files) - file_idx)
            print(f"   Progress: {total_chunks} chunks total, ~{remaining:.0f}s remaining")

        if total_chunks == 0:
            print("\n No valid content found")
            return False

        print(f"\n Processing completed in {time.time() - start_time:.1f}s")
        print(f" Total chunks: {total_chunks}")
        return True

    # --------------------------------------------------------

    def build_index(self) -> bool:
        if not self.chunks:
            print(" No chunks to index")
            return False

        print(f"\n Building FAISS index for {len(self.chunks)} chunks...")
        start_time = time.time()

        texts = [c["text"] for c in self.chunks]

        print(" Generating embeddings...")
        embeddings_list = []
        for i in tqdm(range(0, len(texts), AIPodConfig.BATCH_SIZE), desc="Embedding", unit="batch"):
            batch = texts[i: i + AIPodConfig.BATCH_SIZE]
            batch_emb = self.embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            embeddings_list.append(batch_emb)

        all_embeddings = np.vstack(embeddings_list)
        print(f" Embedding dimension : {all_embeddings.shape[1]}")
        print(f" Total vectors       : {all_embeddings.shape[0]}")

        print("  Creating FAISS index...")
        # IndexFlatIP = cosine similarity on normalised vectors (values 0-1, higher = more similar)
        index = faiss.IndexFlatIP(all_embeddings.shape[1])
        index.add(all_embeddings.astype("float32"))
        faiss.write_index(index, AIPodConfig.FAISS_INDEX_PATH)

        with open(AIPodConfig.CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

        metadata_summary = {
            "total_chunks":    len(self.chunks),
            "languages":       list(set(c["metadata"].get("language", "?") for c in self.chunks)),
            "documents":       list(set(c["metadata"].get("file_name", "?") for c in self.chunks)),
            "built_at":        datetime.datetime.now().isoformat(),
            "version":         AIPodConfig.VERSION,
            "embedding_model": AIPodConfig.EMBEDDING_MODEL,
            "chunk_size":      AIPodConfig.CHUNK_SIZE,
            "chunk_overlap":   AIPodConfig.CHUNK_OVERLAP,
            "pdf_backend":     "pymupdf" if PYMUPDF_AVAILABLE else "unavailable",
        }
        with open(AIPodConfig.METADATA_PATH, "wb") as f:
            pickle.dump(metadata_summary, f)

        print("\n Calibrating semantic thresholds...")
        thresholds = self.calibrate_thresholds(all_embeddings)
        with open(AIPodConfig.THRESHOLDS_PATH, "wb") as f:
            pickle.dump(thresholds, f)

        print(f"\n Index built in {time.time() - start_time:.1f}s")
        print(f" Thresholds calibrated and saved")
        return True

    # --------------------------------------------------------

    def calibrate_thresholds(self, embeddings: np.ndarray) -> Dict:
        print("   Analysing similarity distribution...")

        if embeddings.shape[0] < 10:
            print("     Too few chunks for calibration — using defaults")
            return {"high": 0.45, "medium": 0.30, "low": 0.20}

        sample_size    = min(50, len(self.chunks))
        sample_indices = random.sample(range(len(self.chunks)), sample_size)
        sample_embs    = embeddings[sample_indices].astype("float32")

        # Use IndexFlatIP — scores are cosine similarities (0 to 1) on normalised vectors
        temp_index = faiss.IndexFlatIP(embeddings.shape[1])
        temp_index.add(sample_embs)

        self_similarities = []
        for emb in sample_embs:
            emb_q = emb.reshape(1, -1)
            scores, indices = temp_index.search(emb_q, 2)
            if indices.shape[1] >= 2:
                # scores[0][0] is self (=1.0), scores[0][1] is nearest neighbour
                self_similarities.append(float(scores[0][1]))

        # Cosine similarity between random pairs = dot product on normalised vectors
        cross_similarities = []
        for _ in range(min(100, sample_size * 2)):
            i, j = random.sample(range(sample_size), 2)
            cos_sim = float(np.dot(sample_embs[i], sample_embs[j]))
            cross_similarities.append(cos_sim)

        avg_self  = sum(self_similarities)  / len(self_similarities)  if self_similarities  else 0.5
        min_self  = min(self_similarities)                             if self_similarities  else 0.4
        avg_cross = sum(cross_similarities) / len(cross_similarities) if cross_similarities else 0.2
        max_cross = max(cross_similarities)                            if cross_similarities else 0.3

        print(f"   • Similar (same topic): avg={avg_self:.2%}, min={min_self:.2%}")
        print(f"   • Random (different)  : avg={avg_cross:.2%}, max={max_cross:.2%}")

        # Place thresholds between noise floor (avg_cross) and genuine matches (min_self)
        raw_high   = min_self  * 0.85   # just below worst genuine match
        raw_medium = avg_cross * 1.4    # comfortably above average noise
        raw_low    = avg_cross * 1.1    # just above noise floor

        # Clamp to absolute sane ranges
        high   = max(0.40, min(0.70, raw_high))
        medium = max(0.25, min(0.55, raw_medium))
        low    = max(0.15, min(0.35, raw_low))

        # Enforce strict ordering: high > medium > low
        medium = min(medium, high   - 0.05)
        low    = min(low,    medium - 0.05)

        thresholds = {"high": high, "medium": medium, "low": low}

        print(f"\n Calibrated Thresholds:")
        print(f"   • High relevance  : > {thresholds['high']:.1%}")
        print(f"   • Medium relevance: {thresholds['medium']:.1%} – {thresholds['high']:.1%}")
        print(f"   • Low relevance   : {thresholds['low']:.1%} – {thresholds['medium']:.1%}")
        print(f"   • Off-topic       : < {thresholds['low']:.1%}")
        return thresholds


# ============================================================
# Main
# ============================================================

import re

if __name__ == "__main__":
    print("=" * 60)
    print(" AI POD — Semantic Document Ingestion (pymupdf edition)")
    print("=" * 60)
    print(f"Company   : {AIPodConfig.COMPANY_NAME}")
    print(f"Version   : {AIPodConfig.VERSION}")
    print(f"Embedding : {AIPodConfig.EMBEDDING_MODEL}")
    print(f"PDF engine: {'pymupdf (fitz)' if PYMUPDF_AVAILABLE else '  NOT AVAILABLE — install pymupdf'}")
    print("=" * 60)

    total_start = time.time()
    try:
        processor = DocumentProcessor()

        if processor.process_all_files():
            if processor.build_index():
                total_elapsed = time.time() - total_start
                print("\n" + "=" * 60)
                print(" INGESTION COMPLETED SUCCESSFULLY!")
                print("=" * 60)
                print(f" Final Statistics:")
                print(f"   • Files processed : {processor.total_files}")
                print(f"   • Total chunks    : {len(processor.chunks)}")
                print(f"   • Total time      : {total_elapsed:.1f}s")
                print(f"   • Index saved to  : {AIPodConfig.INDEX_DIR}")
                print(f"   • Chunk overlap   : {AIPodConfig.CHUNK_OVERLAP} chars (active)")
                print(f"   • PDF backend     : {'pymupdf' if PYMUPDF_AVAILABLE else 'unavailable'}")
                print("\n  Next step: python query_system.py")
                print("=" * 60)
            else:
                print("\n Failed to build index")
        else:
            print("\n Failed to process files")

    except KeyboardInterrupt:
        print("\n\n  Ingestion interrupted by user")
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
