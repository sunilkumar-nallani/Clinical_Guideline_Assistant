# 🫘 KDIGO 2026 Nephrology Guideline Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=for-the-badge)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)

A clinical decision support assistant built to query the **KDIGO 2026 Clinical Practice Guideline for Acute Kidney Injury (AKI) and Acute Kidney Disease (AKD)** (Public Review Draft, March 2026). 

Every answer is strictly grounded in the guideline text, clinical tables, and treatment flowcharts, with exact source document and printed page citations attached to every single claim.

---

## 🎯 Why This Exists

Medical clinical practice guidelines are often 500+ pages long, packed with complex staging grids, diagnostic criteria, and decision trees. When clinicians need to make time-sensitive decisions on the ward—such as determining AKI staging, selecting baseline creatinine, reviewing nephrotoxic medication rules, or planning post-discharge follow-up—flipping through a massive PDF takes valuable time.

This project delivers a **grounded, citation-backed conversational assistant** that allows clinicians to ask direct questions in natural language (in English or German) and receive evidence-based, page-cited answers in seconds.

---

## 💡 Key Clinical Features & Upgrades

- **Preserved Clinical Tables:** Unlike standard PDF text extractors that break multi-row criteria, the ingestion engine (`pdfplumber`) extracts tables as intact Markdown grids (`[TABLE START] ... [TABLE END]`). Complex staging grids (e.g., AKI staging by SCr and Urine Output) remain structured and fully aligned.
- **Flowchart & Algorithm Ingestion:** Standard RAG pipelines are blind to embedded diagrams and decision trees. This assistant injects manually verified semantic descriptions of critical KDIGO algorithms (e.g., *Figure 9: Multicomponent Kidney Protection Strategy*, *Figure 12: Contrast-Associated AKI Risk Assessment*, *Figure 22: Post-Discharge Follow-up Framework*).
- **True Printed Page Numbers:** Uses PyMuPDF (`fitz`) to extract logical page numbers matching the document's printed pages, rather than offset PDF index numbers.
- **Multi-Query Synonym Expansion:** Uses a clinical query generator that expands acronyms and non-English terminology (e.g., German terms like *OAK* $\rightarrow$ *Oral Anticoagulation*, *ANV* $\rightarrow$ *Acute Kidney Injury*, *RRT* $\rightarrow$ *Renal Replacement Therapy*) into targeted medical search queries.
- **Diverse Retrieval (MMR) & FlashRank Reranking:** Merges candidate chunks across queries using Maximal Marginal Relevance (MMR) for breadth across 500 pages, followed by ultra-fast local cross-encoder reranking (`FlashRank`) to deliver high-precision context to the LLM.
- **MedTech UI Design:** A modern, clean Streamlit frontend inspired by Apple and Gemini design principles, complete with a KDIGO chapter navigator, clinical disclaimer, and a one-click session reset for ward device handoffs.

---

## 🔄 End-to-End Architecture

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
┌──────────────────────────▼─────────────────────────────┐
│                 Clinical Query Flow                    │
│  1. User asks question in Streamlit UI (app.py)        │
│  2. MultiQueryRetriever expands query & acronyms       │
│  3. FAISS MMR retrieves diverse candidates (k=10)      │
│  4. FlashRank Cross-Encoder reranks locally to top 5   │
│  5. Context formatted with [Source: filename, p.X]     │
│  6. Qwen-30B LLM generates cited, evidence-based reply │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/sunilkumar-nallani/Bacteremia_Clinical_Guideline_Assistant.git
cd Bacteremia_Clinical_Guideline_Assistant
```

### 2. Set up virtual environment and install dependencies
```bash
python -m venv venv
.\venv\Scripts\activate      # On Windows
# source venv/bin/activate   # On Linux/macOS

pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=https://chat-ai.academiccloud.de/v1   # Optional custom endpoint
```

### 4. Build the Vector Store Index
Ensure the guideline PDF is placed inside `Nephrology_Guidelines/`, then run:
```bash
python Build_Index.py
```
*This extracts text, tables, and flowchart annotations from the 500-page guideline and creates your local `faiss_index/`.*

### 5. Launch the Clinical Assistant
```bash
streamlit run app.py
```

---

## 🩺 Guardrails & Clinical Verification

1. **Strict Source Grounding:** The assistant is explicitly instructed to answer *only* from retrieved excerpts. If the guideline does not address a clinical scenario (for instance, general outpatient oral anticoagulation regimens), it truthfully states that the information is not in the guideline rather than inventing recommendations.
2. **Mandatory Citations:** Every single clinical statement includes a direct citation tag: `[KDIGO 2026, p.X]`.
3. **Recommendation Grading:** Whenever available in the context, specific KDIGO recommendation grades (e.g., *Recommendation 3.2.3, Grade 1B*) are cited.
4. **Draft Notice:** The system displays a prominent clinical disclaimer indicating that answers are sourced from the *Public Review Draft* and require verification against the final published text and individual clinical judgement.

---

## 🛠️ Tech Stack

- **Frameworks:** [LangChain](https://www.langchain.com/) · [LangChain Classic](https://pypi.org/project/langchain-classic/)
- **Document Parsing:** [pdfplumber](https://github.com/jsvine/pdfplumber) (Table extraction) · [PyMuPDF](https://pymupdf.readthedocs.io/) (Page labels & text)
- **Vector Database:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **Reranker:** [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) (Local ultra-fast cross-encoder)
- **LLM & Embeddings:** Qwen-30B / e5-mistral-7b-instruct via OpenAI-compatible API
- **Frontend UI:** [Streamlit](https://streamlit.io/) with custom clinical CSS

---

## ⚠️ Limitations & Ethical Disclaimer

- **Research & Prototyping Only:** This application is designed for research, education, and decision support—it is **not** an autonomous medical device and does not replace professional nephrology consultation or clinical diagnosis.
- **Human-in-the-Loop:** Clinicians must always verify cited pages and data points before acting on any output.
- **Guideline Scope:** The assistant's knowledge boundary is strictly limited to the loaded guideline documents.

---

## 👤 Author & Contact

**Sunil Kumar Nallani**  
- Email: sunilkumar.nallani@stud.uni-goettingen.de  
- LinkedIn: [linkedin.com/in/nallani-sunil-kumar-67227a243](https://www.linkedin.com/in/nallani-sunil-kumar-67227a243/)  
- GitHub: [github.com/sunilkumar-nallani](https://github.com/sunilkumar-nallani)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
