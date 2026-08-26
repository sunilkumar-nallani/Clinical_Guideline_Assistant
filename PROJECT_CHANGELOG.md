## Summary
This project is an end-to-end RAG (Retrieval-Augmented Generation) clinical decision support assistant designed to answer medical questions grounded strictly in the *KDIGO 2026 Clinical Practice Guideline for Acute Kidney Injury (AKI) and Acute Kidney Disease (AKD)* (Public Review Draft, March 2026). It processes a dense, 500-page medical document, preserving markdown table structures and dynamically injecting manually annotated visual flowcharts to eliminate AI visual blind spots. The retrieval pipeline leverages FAISS similarity search combined with a local, zero-latency FlashRank cross-encoder reranker, and generates strictly cited answers using the Qwen-30B LLM. The frontend is built with Streamlit, styled with a modern Apple/Gemini-inspired "MedTech" interface tailored for clinical ward workflows.

---

## What Changed (file by file)

### 1. `Build_Index.py`
- **Status**: Modified
- **Why**: The original pipeline used naive PDF text extraction and semantic chunking, which broke multi-row clinical tables and misidentified printed page numbers on a 500-page PDF.
- **Intuition**: Clinical guidelines rely heavily on criteria grids (e.g., AKI staging by SCr vs. Urine Output). If table boundaries are split or formatting is lost, the LLM hallucinates numbers or fails to stage patients accurately.
- **What Changed**:
  - Replaced `PyPDFLoader` with `pdfplumber` to extract tables as structured Markdown blocks wrapped in `[TABLE START] ... [TABLE END]`.
  - Integrated `PyMuPDF` (`fitz`) to extract true printed page labels.
  - Replaced `SemanticChunker` with `RecursiveCharacterTextSplitter` using `\n\n` as the primary delimiter so markdown tables stay intact as single atomic units.
  - Added a citation-density-based reference filter (`is_reference_page`) to remove pure bibliographies while preserving inline clinical evidence.
  - Added `load_figure_annotations()` to merge manual flowchart annotations into the indexing pipeline before vector store generation.
- **Key Code Snippet**:
  ```python
  def load_figure_annotations(annotations_path="figure_annotations.json"):
      with open(annotations_path, "r", encoding="utf-8") as f:
          annotations = json.load(f)
      docs = []
      for ann in annotations:
          content = f"[FIGURE ANNOTATION]\nFigure: {ann.get('title')}\nSource: {ann.get('source')}, p.{ann.get('page')}\n\n{ann.get('description')}"
          docs.append(Document(page_content=content, metadata={"source": ann.get("source"), "page_label": ann.get("page"), "chunk_type": "figure_annotation"}))
      return docs
  ```

---

### 2. `figure_annotations.json`
- **Status**: Created
- **Why**: Traditional PDF parsers only extract raw text streams, completely ignoring embedded images, decision trees, and clinical flowcharts.
- **Intuition**: Key treatment algorithms (e.g., Figure 9: *Multicomponent Kidney Protection Strategy*, Figure 12: *Contrast-Associated AKI Risk Flowchart*, Figure 22: *Post-Discharge Follow-up Framework*) only exist as diagrams in the guideline. Converting them into semantic JSON allows the RAG vector store to retrieve and reason over visual clinical algorithms.
- **What Changed**: Authored 12 structured JSON objects containing clinical summaries, step-by-step decision rules, and evidence citations for the primary KDIGO figures.

---

### 3. `Rag_Pipeline.py`
- **Status**: Modified
- **Why**: Medical queries from clinicians frequently use abbreviations, acronyms, or non-English terms (e.g., German terms like *OAK*, *ANV*). A single similarity search can miss crucial guideline sections that express the same clinical concept using formal English phrasing.
- **Intuition**: By pairing **Multi-Query expansion** (which translates and generates 3 medical synonym variations) with **Maximal Marginal Relevance (MMR)** and a local **FlashRank cross-encoder reranker**, the pipeline achieves high recall across the 500-page document while keeping candidate diversity balanced.
- **What Changed**:
  - Implemented `MultiQueryRetriever` with a specialized clinical synonym prompt (`query_gen_prompt`) that explicitly handles medical acronym expansion and German-to-English terminology mapping.
  - Configured the base retriever with MMR (`search_type="mmr"`, `k=10`, `fetch_k=25`) to prevent repetitive chunk retrieval.
  - Piped all multi-query candidate chunks into `FlashrankRerank(top_n=5)` for local cross-encoder re-ranking.
  - Added strict timeout (`30s`) and retry (`max_retries=2`) boundaries to protect against remote university server lag.
  - Updated the system prompt to KDIGO 2026 Nephrology with citation requirements (`[KDIGO 2026, p.X]`).
- **Key Code Snippet**:
  ```python
  base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 25})
  multi_query_retriever = MultiQueryRetriever.from_llm(
      retriever=base_retriever, llm=gwdg_llm, prompt=query_gen_prompt
  )
  compressor = FlashrankRerank(top_n=5)
  retriever = ContextualCompressionRetriever(
      base_compressor=compressor, base_retriever=multi_query_retriever
  )
  ```

---

### 4. `app.py`
- **Status**: Modified
- **Why**: 
  1. The UI was branded for Bacteremia with irrelevant sample questions.
  2. The CSS had `#MainMenu, footer, header { visibility: hidden; }`, which hid Streamlit's top header and caused the sidebar toggle chevron (`>`) to disappear, making the sidebar permanently invisible on smaller screens or tablets.
- **Intuition**: Clinicians need an uncluttered, responsive interface with easy chapter navigation, one-tap sample questions, and a quick reset button for patient handoffs.
- **What Changed**:
  - Replaced the global `header` hiding rule with targeted rules that keep `[data-testid="stSidebarCollapseButton"]` and `[data-testid="collapsedControl"]` visible and functional.
  - Rebranded the interface completely to KDIGO 2026 AKI/AKD.
  - Added a KDIGO Chapter Navigator (Chapters 1–6) to the sidebar.
  - Added a `🗑️ Clear conversation` session reset button.
  - Added an evidence-based clinical disclaimer banner.
  - Updated empty-state prompts and sample question chips to target AKI staging, RRT, drug stewardship, and follow-up care.

---

### 5. `requirements.txt`
- **Status**: Modified
- **Why**: Added required libraries for table parsing, PDF metadata extraction, and text splitting.
- **What Changed**: Added `pdfplumber`, `PyMuPDF`, and `langchain-text-splitters`.

---

## Architecture After These Changes

```
┌────────────────────────────────────────────────────────┐
│                   Data Ingestion                       │
│  KDIGO 2026 PDF (500 pages) + figure_annotations.json  │
└──────────────────────────┬─────────────────────────────┘
                           │ pdfplumber (Markdown Tables) + PyMuPDF
                           ▼
┌────────────────────────────────────────────────────────┐
│                   Index Building                       │
│  RecursiveCharacterTextSplitter (\n\n atomic tables)   │
│  → e5-mistral-7b-instruct Embeddings → FAISS Index     │
└──────────────────────────┬─────────────────────────────┘
                           │
┌────────────────────────────────────────────────────────┐
│                 Clinical Query Flow                    │
│  1. User asks question in Streamlit UI (app.py)        │
│  2. MultiQueryRetriever expands query (synonyms/transl)│
│  3. FAISS MMR retrieves diverse candidate chunks (k=10)│
│  4. FlashRank Cross-Encoder reranks locally to top 5   │
│  5. Context formatted with [Source, p.X] tags          │
│  6. Qwen-30B LLM generates cited, evidence-based reply │
└────────────────────────────────────────────────────────┘
```

---

## Commands That Were Run
- `pip install -r requirements.txt`: Installed `pdfplumber`, `PyMuPDF`, and `langchain-text-splitters`.
- `python Build_Index.py`: Extracted 488 content pages + 12 figure annotations, creating 1,250 structured chunks saved to `faiss_index/`.
- `python -m py_compile app.py`: Validated syntax across UI components.
- `git status` / `git diff`: Verified exact line-by-line modifications across all files.

---

## Known Issues / Things Not Yet Verified
- **Nephrologist Review**: The 12 descriptions in `figure_annotations.json` are currently in `DRAFT_NEEDS_NEPHROLOGIST_REVIEW` status and require formal clinical sign-off prior to production deployment in a hospital.
- **AcademicCloud Server Latency**: The remote embedding endpoint (`chat-ai.academiccloud.de/v1/embeddings`) occasionally experiences high queuing times. While single-query retrieval is significantly faster than multi-query, cold requests may still take several seconds depending on server load.

---

## How to Verify This Yourself
1. **Start the App**:
   ```powershell
   .\venv\Scripts\activate
   streamlit run app.py
   ```
2. **Verify Sidebar Visibility**: Check that the sidebar appears on the left with KDIGO Chapters 1–6 and sample questions (and that the toggle chevron expands/collapses it cleanly).
3. **Verify Flowchart Retrieval**: Ask *"What is the multicomponent kidney protection strategy?"* — Confirm that it cites **Figure 9 (p. 159)**.
4. **Verify Table Retrieval**: Ask *"What are the criteria for staging AKI?"* — Confirm that it displays the SCr and Urine Output staging criteria with intact units (`mg/dl`, `ml/kg/h`).
5. **Verify Reset**: Tap `🗑️ Clear conversation` to confirm that conversation state clears instantly.
