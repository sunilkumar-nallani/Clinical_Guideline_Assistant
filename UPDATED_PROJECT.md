# 📋 UPDATED_PROJECT — Complete Change Report

## Feature: Dual Evidence Source Support (Guidelines + Publications)

**Date:** September 10, 2026  
**Goal:** Enable doctors to search **guidelines**, **publications**, or **both** — so the app becomes a comprehensive clinical evidence assistant (similar to UpToDate), not just a single-guideline lookup tool.

**Doctor's Feedback That Triggered This:**  
> "Add publications too, not only guidelines — many doctors use publications for up-to-date evidence."

---

## 🧠 The Core Problem & Intuition

### Before (What We Had)
The app could ONLY search the KDIGO 2026 guideline PDF. One folder (`Nephrology_Guidelines/`), one FAISS index (`faiss_index/`), one retriever, one prompt, one answer — all hardcoded to guidelines.

### After (What We Built)
The app can now search **two types of clinical evidence independently**:

| Source | What It Contains | How Doctors Use It |
|--------|-----------------|-------------------|
| **Guidelines** | KDIGO consensus recommendations, staging criteria, treatment algorithms | "What **should** I do?" — Standard of care |
| **Publications** | Research papers, clinical trials (RCTs), meta-analyses | "What does the **latest evidence** say?" — Nuance, emerging therapies |
| **Both** | Merged results from both | "Give me the **complete picture**" — Guideline + supporting/contrasting evidence |

### Why Separate Indexes Instead of One Big Index?
**FAISS does not support metadata filtering natively.** If we put everything in one index, we can't tell the retriever "only search guidelines" — it would pull random publication chunks when the doctor specifically wants guideline recommendations. Separate indexes = clean, precise, targeted retrieval.

---

## 📁 Change 1: New Folder Structure

### What Was Created
```
documents/                          ← NEW parent folder
├── guidelines/                     ← NEW: guideline PDFs go here
│   └── KDIGO-2026-AKI-AKD-Guideline-Public-Review-Draft-March-2026.pdf  (copied from old location)
└── publications/                   ← NEW: research paper PDFs go here
    └── README.md                   ← NEW: instructions for adding papers
```

### Why
- **Clean separation**: Guidelines and publications are fundamentally different document types. Mixing them in one folder makes it impossible to build separate indexes.
- **Scalability**: Tomorrow, drop a cardiology guideline in `documents/guidelines/` or a new trial paper in `documents/publications/` — just re-run `Build_Index.py`, no code changes.
- **Doctor-friendly workflow**: The doctor sends a PDF → you drop it in the right folder → rebuild → done.

### The `documents/publications/README.md` File
This is a helper file so anyone (you, the doctor, a collaborator) knows:
- Where to put publication PDFs
- What format is supported (PDF only)
- Naming tips (e.g., `STARRT-AKI-Trial-Bagshaw-2020.pdf`)
- That reference pages are auto-filtered

### Backward Compatibility
The old `Nephrology_Guidelines/` folder was **NOT deleted**. The code falls back to it if `documents/guidelines/` doesn't exist. Nothing breaks if someone clones the old repo.

---

## 📁 Change 2: Build_Index.py — Separate Index Builder

**File:** [`Build_Index.py`](file:///c:/Users/sunil/OneDrive/Desktop/Lang_Chain/Clinical_Guideline_Assistant_RAG_Project/Build_Index.py)  
**Lines changed:** Entire file rewritten (156 → 219 lines)

### What Changed (Step by Step)

#### 2.1 — `extract_pdf_with_tables()` Now Accepts `source_type` Parameter

**Before (Line 46, old):**
```python
def extract_pdf_with_tables(pdf_path):
```

**After (Line 46, new):**
```python
def extract_pdf_with_tables(pdf_path, source_type="guideline"):
```

**Why:** Every chunk extracted from a PDF now gets a `source_type` metadata tag (`"guideline"` or `"publication"`). This tag flows through the entire pipeline — from indexing → retrieval → context formatting → LLM prompt. Without this tag, the system can't tell which evidence type a chunk came from.

**The metadata change (Lines 79-86):**
```python
metadata={
    "source": os.path.basename(pdf_path),
    "page_label": page_label,
    "source_type": source_type,    # ← NEW: tags every chunk
}
```

#### 2.2 — Figure Annotations Also Get `source_type` Tag

**Lines 118-120 (new):**
```python
"chunk_type": "figure_annotation",
"source_type": "guideline",         # ← Figure annotations are always guidelines
```

**Why:** Figure annotations (the manually written flowchart descriptions from `figure_annotations.json`) are descriptions of KDIGO guideline figures. They should ONLY appear in the guidelines index, never in publications.

#### 2.3 — New `_build_single_index()` Helper Function

**Lines 126-165 (entirely new):**
```python
def _build_single_index(pdf_folder, source_type, save_path, figure_annotations_path=None):
```

**Why:** Instead of duplicating the same "scan folder → extract PDFs → chunk → embed → save" logic twice (once for guidelines, once for publications), we extracted it into a reusable function. This is the **DRY principle** (Don't Repeat Yourself).

**What it does:**
1. Checks if the folder exists → if not, prints a warning and returns `False`
2. Checks if the folder has any `.pdf` files → if not, prints info and returns `False`
3. Extracts all PDFs with the given `source_type` tag
4. Injects figure annotations **only if** `source_type == "guideline"` (Line 147)
5. Chunks, embeds, and saves to the specified FAISS index path

**The `figure_annotations_path` guard (Line 147):**
```python
if figure_annotations_path and source_type == "guideline":
```
**Intuition:** Figure annotations describe KDIGO flowcharts. Injecting them into the publications index would contaminate it with guideline content — a doctor searching only publications would get guideline figures mixed in, which is wrong.

#### 2.4 — `build_and_save_index()` Now Builds Two Indexes

**Before (old function):**
```python
def build_and_save_index(pdf_folder="Nephrology_Guidelines/", save_path="faiss_index"):
    # → builds ONE index from ONE folder
```

**After (Lines 168-215):**
```python
def build_and_save_index(
    guidelines_folder="documents/guidelines/",
    publications_folder="documents/publications/",
    guidelines_index_path="faiss_index_guidelines",
    publications_index_path="faiss_index_publications",
    legacy_guidelines_folder="Nephrology_Guidelines/",
):
```

**What it does now:**
1. **Guidelines Index**: Scans `documents/guidelines/` → builds `faiss_index_guidelines/`
2. **Publications Index**: Scans `documents/publications/` → builds `faiss_index_publications/`
3. **Legacy fallback** (Line 188): If `documents/guidelines/` doesn't exist, it tries the old `Nephrology_Guidelines/` folder

**Why the legacy fallback?**
```python
gf = guidelines_folder if os.path.exists(guidelines_folder) else legacy_guidelines_folder
```
If someone clones the repo but hasn't reorganized their folders yet, the build still works. No breaking changes.

#### 2.5 — Console Output With Source Labels

**Lines 144, 159, 164 (new):**
```python
print(f"[{source_type.upper()}] Loaded {len(all_documents)} content pages from {pdf_folder}")
print(f"[{source_type.upper()}] Created {len(chunks)} structural chunks")
print(f"[{source_type.upper()}] Saved FAISS index to {save_path}/")
```

**Why:** When building two indexes, you need to know which one's output you're reading. The `[GUIDELINE]` and `[PUBLICATION]` prefixes make the console output unambiguous.

---

## 📁 Change 3: Rag_Pipeline.py — Source-Aware Retrieval

**File:** [`Rag_Pipeline.py`](file:///c:/Users/sunil/OneDrive/Desktop/Lang_Chain/Clinical_Guideline_Assistant_RAG_Project/Rag_Pipeline.py)  
**Lines changed:** Entire file rewritten (154 → 291 lines)

### 3.1 — Loading Two Separate FAISS Indexes

**Before (old):**
```python
vector_store = FAISS.load_local("faiss_index", embeddings, ...)
base_retriever = vector_store.as_retriever(...)
```

**After (Lines 58-81):**
```python
_guidelines_store = None
_publications_store = None
_legacy_store = None

if os.path.exists("faiss_index_guidelines"):
    _guidelines_store = FAISS.load_local("faiss_index_guidelines", ...)

if os.path.exists("faiss_index_publications"):
    _publications_store = FAISS.load_local("faiss_index_publications", ...)

# Legacy fallback
if _guidelines_store is None and _publications_store is None and os.path.exists("faiss_index"):
    _legacy_store = FAISS.load_local("faiss_index", ...)
    _guidelines_store = _legacy_store
```

**Why:** Each source type gets its own FAISS vector store. The retriever for guidelines only searches guideline chunks, and vice versa. The legacy fallback (Lines 71-74) means if someone hasn't run the new `Build_Index.py` yet, the old single `faiss_index/` still works.

### 3.2 — `AVAILABLE_SOURCES` List (Exported to UI)

**Lines 77-81 (new):**
```python
AVAILABLE_SOURCES = []
if _guidelines_store is not None:
    AVAILABLE_SOURCES.append("guidelines")
if _publications_store is not None:
    AVAILABLE_SOURCES.append("publications")
```

**Why:** The UI (app.py) needs to know which source types are actually available (i.e., which indexes were built). If no publications are indexed, the UI shouldn't show "📄 Publications" as an option — it would just confuse the doctor.

### 3.3 — Reusable `_build_retriever()` Function

**Lines 106-114 (new):**
```python
def _build_retriever(vector_store):
    """Build a multi-query + reranked retriever from a FAISS vector store."""
    base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 25})
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=gwdg_llm, prompt=query_gen_prompt
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multi_query_retriever
    )
```

**Why:** The same retrieval pipeline (MMR → Multi-Query → FlashRank Rerank) applies to both guidelines and publications. Instead of duplicating this logic, we build a factory function.

### 3.4 — `format_docs_with_citations()` Now Tags Source Type

**Before (old):**
```python
formatted.append(f"Source: {source}, p.{page}\n{doc.page_content}")
```

**After (Lines 120-128):**
```python
source_type = doc.metadata.get("source_type", "unknown")
tag = "[GUIDELINE]" if source_type == "guideline" else "[PUBLICATION]"
formatted.append(f"{tag} Source: {source}, p.{page}\n{doc.page_content}")
```

**Why:** When the LLM sees the context, each chunk is prefixed with `[GUIDELINE]` or `[PUBLICATION]`. This is critical for the "Both" mode prompt — the LLM can then say "According to the KDIGO guideline (p.159)..." vs "A 2020 RCT by Bagshaw et al. found...". Without this tag, the LLM can't distinguish evidence types.

### 3.5 — Three Source-Specific LLM Prompts

This is the most important change in the pipeline. Each source type gets a tailored prompt that instructs the LLM HOW to present the evidence.

#### Guidelines Prompt (`_PROMPT_GUIDELINES`, Lines 134-151)
**Same as before** — emphasizes KDIGO recommendation grades, guideline-specific citations.

#### Publications Prompt (`_PROMPT_PUBLICATIONS`, Lines 153-171) — NEW
Key rules:
- "Mention the **study design** (RCT, cohort, meta-analysis), **sample size**, and **key statistical findings** (OR, HR, CI, p-value)"
- "Clearly distinguish between **primary endpoints** and **secondary/exploratory findings**"

**Why this matters clinically:** A doctor reading publication evidence needs to know: Was this an RCT or an observational study? How many patients? Was the finding statistically significant? A guideline prompt doesn't need this because guidelines already synthesize evidence into graded recommendations.

#### Both Prompt (`_PROMPT_BOTH`, Lines 173-192) — NEW
Key rules:
- "The context contains excerpts tagged as `[GUIDELINE]` or `[PUBLICATION]`. **Clearly distinguish** between guideline recommendations and research findings"
- "Structure your answer to present **guideline recommendations first**, then supporting or contrasting research evidence"

**Why this structure:** Clinicians think in terms of "What does the standard of care say?" (guidelines first) → "What's the latest evidence?" (publications second). This mirrors how UpToDate structures its content.

### 3.6 — Combined Retriever for "Both" Mode

**Lines 208-215 (new):**
```python
def _build_combined_retriever(retriever_a, retriever_b):
    def _merge_retrieve(question):
        docs_a = retriever_a.invoke(question) if retriever_a else []
        docs_b = retriever_b.invoke(question) if retriever_b else []
        return docs_a + docs_b
    return RunnableLambda(_merge_retrieve)
```

**Why:** When the doctor selects "Both", we need to search BOTH indexes and merge the results. We retrieve top-5 reranked chunks from each index (5 guideline + 5 publication = 10 total), concatenate them, and send the combined context to the LLM with the "Both" prompt.

**Why not re-rerank the merged set?** The chunks from each index are already independently reranked by FlashRank. Re-reranking the combined set would bias toward one source type over the other (guidelines are typically more structured and would score higher). By keeping them separate and concatenating, we guarantee the doctor gets balanced evidence from both sources.

### 3.7 — `answer_question()` Now Accepts `source_type`

**Before (old):**
```python
def answer_question(question: str) -> str:
```

**After (Lines 242-284):**
```python
def answer_question(question: str, source_type: str = "guidelines") -> str:
```

**Why:** The UI passes the doctor's source selection to the pipeline. Defaults to `"guidelines"` for backward compatibility — if any code calls `answer_question(q)` without specifying `source_type`, it still works exactly as before.

**The selection logic (Lines 253-263):**
```python
if source_type == "publications" and publications_chain:
    chain = publications_chain
elif source_type == "both" and combined_chain:
    chain = combined_chain
elif guidelines_chain:
    chain = guidelines_chain      # ← fallback: always try guidelines
else:
    return "⚠️ No FAISS index is available..."
```

**Intuition behind the fallback chain:** If a doctor selects "Publications" but no publications are indexed, we fall back to guidelines rather than showing an error. Better to give a useful answer from a different source than to show nothing. The source badge in the UI makes it clear which source was actually used.

---

## 📁 Change 4: app.py — Source Selector UI

**File:** [`app.py`](file:///c:/Users/sunil/OneDrive/Desktop/Lang_Chain/Clinical_Guideline_Assistant_RAG_Project/app.py)  
**Lines changed:** Entire file rewritten (556 → 758 lines)

### 4.1 — Import `AVAILABLE_SOURCES` from Pipeline

**Line 9 (new):**
```python
from Rag_Pipeline import answer_question, AVAILABLE_SOURCES
```

**Why:** The UI needs to know which source types have built indexes. If only guidelines are indexed, we don't show the "Publications" or "Both" radio options — that would confuse the doctor.

### 4.2 — Page Title Updated

**Before:** `"KDIGO AKI/AKD Guideline Assistant"`  
**After:** `"Nephrology Clinical Evidence Assistant"`

**Why:** The app is no longer just a guideline assistant — it searches multiple evidence sources. The broader name reflects its expanded scope.

### 4.3 — Source Badge CSS (Lines 133-157) — NEW

```css
.source-badge { ... }
.source-badge-guidelines { background: #E0F2F1; color: #094E4C; }   /* teal */
.source-badge-publications { background: #EDE9FE; color: #5B21B6; } /* purple */
.source-badge-both { background: #FEF3C7; color: #92400E; }         /* amber */
```

**Why:** Every assistant answer now shows a small colored badge above it indicating which source was searched:
- 📋 GUIDELINES (teal badge)
- 📄 PUBLICATIONS (purple badge)
- 📚 BOTH (amber badge)

**Clinical importance:** If a doctor scrolls through their chat history, they need to instantly see which evidence type each answer came from. A teal badge = guideline recommendation. A purple badge = research finding. This prevents confusion.

### 4.4 — Three Sets of Sample Questions (Lines 365-392) — NEW

**Before:** One list of 10 guideline-focused sample questions.

**After:** Three lists:
- `SAMPLE_QUESTIONS_GUIDELINES` (10 questions) — same as before
- `SAMPLE_QUESTIONS_PUBLICATIONS` (5 questions) — research-focused: trial results, meta-analyses, latest evidence
- `SAMPLE_QUESTIONS_BOTH` (5 questions) — comparative: "What does KDIGO recommend AND what do the trials show?"

**Why:** If a doctor selects "Publications" mode, showing guideline-focused sample questions like "What are the KDIGO 2026 criteria..." doesn't make sense. The sample questions should match the selected source type to guide the doctor toward meaningful queries.

### 4.5 — `SOURCE_CONFIG` Dictionary (Lines 409-431) — NEW

```python
SOURCE_CONFIG = {
    "guidelines": { "icon": "📋", "label": "Guidelines", "spinner": "Searching KDIGO 2026 guideline...", ... },
    "publications": { "icon": "📄", "label": "Publications", "spinner": "Searching research publications...", ... },
    "both": { "icon": "📚", "label": "Both", "spinner": "Searching all clinical evidence...", ... },
}
```

**Why:** Instead of scattering `if/elif/else` blocks throughout the code for icons, labels, spinner text, and CSS classes, we centralize all source-type display properties in one dictionary. Cleaner code, easier to modify.

### 4.6 — Evidence Source Radio Selector in Sidebar (Lines 454-496) — NEW

```python
selected_label = st.radio(
    "Select evidence source",
    source_options,    # ["📋 Guidelines", "📄 Publications", "📚 Both"]
    index=0,
    label_visibility="collapsed",
    key="source_selector",
)
```

**Why this design:**
- **Radio buttons** (not a dropdown) — fewer clicks, immediately visible, large tap targets for ward tablets
- **Only shows available options** — if no publications index exists, only "📋 Guidelines" appears
- **"📚 Both" only appears when both indexes exist** (Line 467: `if len(AVAILABLE_SOURCES) >= 2`)
- **Shows a helper message** when publications aren't indexed yet (Lines 490-496)

### 4.7 — Dynamic About Section (Lines 500-523) — NEW

The "About" text in the sidebar changes based on the selected source:
- **Guidelines**: "Answers retrieved exclusively from the KDIGO 2026 guideline..."
- **Publications**: "Answers retrieved from indexed research publications..."
- **Both**: "Answers synthesize evidence from both..."

**Why:** The doctor should always know exactly what knowledge base they're querying.

### 4.8 — Conditional KDIGO Chapter Navigator (Lines 527-539) — NEW

**Before:** Chapter navigator always visible.  
**After:** Only shows when source is "guidelines" or "both".

```python
if selected_source in ("guidelines", "both"):
    # Show chapter navigator
```

**Why:** KDIGO chapters are irrelevant when the doctor is searching only publications. Hiding them reduces sidebar clutter.

### 4.9 — Dynamic Header Title & Subtitle (Lines 608-626) — NEW

The main area header changes based on source:
- **Guidelines**: "🫘 KDIGO 2026 AKI / AKD Guideline Assistant"
- **Publications**: "📄 Nephrology Research Publications Assistant"
- **Both**: "📚 Nephrology Evidence Assistant"

**Why:** The header should immediately tell the doctor what mode they're in.

### 4.10 — Source Badge on Every Assistant Message (Lines 670-678, 731-737) — NEW

```python
st.markdown(
    f'<span class="source-badge {src_cfg["badge_class"]}">'
    f'{src_cfg["icon"]} {src_cfg["label"]}</span>',
    unsafe_allow_html=True,
)
```

**Why:** Each answer in the chat history shows which source was used. If the doctor switches between modes mid-conversation, they can still see which answers came from guidelines vs publications.

### 4.11 — `source_type` Stored in Message History (Lines 753-757) — NEW

```python
st.session_state.messages.append({
    "role": "assistant",
    "content": answer,
    "source_type": selected_source,    # ← NEW: remembers which source was used
})
```

**Why:** When the chat history is replayed (e.g., page refresh), we need to know which source badge to show for each historical answer.

### 4.12 — Dynamic Spinner Text (Line 739) — NEW

```python
with st.spinner(src_cfg["spinner"]):
```

- Guidelines: "Searching KDIGO 2026 guideline..."
- Publications: "Searching research publications..."
- Both: "Searching all clinical evidence..."

**Why:** The doctor should know what's happening during the loading state.

### 4.13 — Fixed Double `answer_question()` Call Bug

**Before (old, Lines 540-543):**
```python
with st.spinner("Searching KDIGO 2026 guideline..."):
    answer = answer_question(question)     # ← Called ONCE here
    try:
        answer = answer_question(question) # ← Called AGAIN here (bug!)
```

**After (Lines 739-741):**
```python
with st.spinner(src_cfg["spinner"]):
    try:
        answer = answer_question(question, source_type=selected_source)  # ← Called only ONCE
```

**Why:** The old code called `answer_question()` twice — once before the try block and once inside it. This wasted API calls and doubled response time. Fixed.

### 4.14 — Sidebar Branding Updated (Lines 447-448) — NEW

**Before:** "🫘 KDIGO 2026" / "AKI / AKD Guideline Assistant"  
**After:** "🫘 Nephrology Evidence Assistant" / "Guidelines & Publications"

---

## 📁 Change 5: README.md — Documentation Updates

**File:** [`README.md`](file:///c:/Users/sunil/OneDrive/Desktop/Lang_Chain/Clinical_Guideline_Assistant_RAG_Project/README.md)

### What Changed:
1. **Title**: "KDIGO 2026 Nephrology Guideline Assistant" → "Nephrology Clinical Evidence Assistant"
2. **Description**: Updated to mention dual evidence sources (guidelines + publications)
3. **Key Features**: Added "Dual Evidence Sources" as the first feature bullet
4. **Architecture Diagram**: Updated Mermaid diagram to show two input paths (Guidelines PDFs + Publication PDFs), two FAISS indexes, and source-aware prompts
5. **Getting Started**: Updated step 4 to explain the new folder structure and dual indexing
6. **Guardrails**: Added "Source Type Tagging" as a new guardrail — in "Both" mode, each claim is labeled as guideline or research
7. **Limitations**: "Guideline Scope" → "Document Scope" (broader, accurate)

---

## 📁 Change 6: documents/publications/README.md — NEW File

A helper file inside the publications folder with:
- Instructions for adding PDFs
- Supported format (PDF only)
- Naming tips
- Note that reference pages are auto-filtered

---

## 🔗 How All Changes Connect (End-to-End Flow)

```
Doctor selects "📄 Publications" in sidebar (app.py)
    │
    ▼
app.py calls answer_question(question, source_type="publications")
    │
    ▼
Rag_Pipeline.py selects publications_chain
    │
    ▼
publications_chain uses _publications_retriever (searches faiss_index_publications/)
    │
    ▼
Retrieved chunks tagged [PUBLICATION] by format_docs_with_citations()
    │
    ▼
_PROMPT_PUBLICATIONS tells LLM: "Report study design, sample size, statistics..."
    │
    ▼
LLM generates answer with publication-style citations
    │
    ▼
app.py shows answer with purple "📄 PUBLICATIONS" badge
```

---

## ✅ Summary: Files Changed

| File | Status | Lines Before | Lines After | Key Change |
|------|--------|-------------|-------------|------------|
| `Build_Index.py` | MODIFIED | 156 | 219 | Builds 2 separate FAISS indexes with source_type metadata |
| `Rag_Pipeline.py` | MODIFIED | 154 | 291 | Source-aware retrievers, 3 tailored prompts, combined retriever |
| `app.py` | MODIFIED | 556 | 758 | Source selector UI, dynamic header/samples, source badges, bug fix |
| `README.md` | MODIFIED | 154 | 160 | Updated title, architecture diagram, docs |
| `documents/guidelines/` | NEW | — | — | New folder for guideline PDFs |
| `documents/publications/` | NEW | — | — | New folder for publication PDFs |
| `documents/publications/README.md` | NEW | — | 15 | Instructions for adding papers |

**Total new/changed lines of code: ~1,268 lines across 4 Python/Markdown files**  
**Zero breaking changes — full backward compatibility with the old folder structure and index**

