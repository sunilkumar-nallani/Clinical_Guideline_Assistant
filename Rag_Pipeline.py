# Module file (a file meant to be imported elsewhere)

# rag_pipeline.py  (runs every time someone asks a question)
#     │
#     │  Loads separate FAISS indexes for guidelines and publications
#     │  Selects the appropriate retriever based on source_type
#     ▼
# main_chain ready to answer questions instantly, no re-embedding needed

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiQueryRetriever

# Specialized model to filterout which chunks (in total 10) are highly releveant to question
from langchain_classic.retrievers import ContextualCompressionRetriever

# Third-party integrations like Flashrank are imported from langchain_community
from langchain_community.document_compressors import FlashrankRerank

from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ---------------------------------------------------------------------------
# Embeddings (shared across all indexes)
# ---------------------------------------------------------------------------
embeddings = OpenAIEmbeddings(
    model="e5-mistral-7b-instruct",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    tiktoken_enabled=False,
    check_embedding_ctx_length=False,
    max_retries=2,
    request_timeout=20.0
)

# ---------------------------------------------------------------------------
# LLM for Query Expansion & Answer Generation
# ---------------------------------------------------------------------------
gwdg_llm = ChatOpenAI(
    model="qwen3-30b-a3b-instruct-2507",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    temperature=0,
    max_retries=2,
    request_timeout=30.0
)

# ---------------------------------------------------------------------------
# Load FAISS indexes — separate for guidelines and publications
# Falls back to the legacy single index if the new indexes don't exist yet
# ---------------------------------------------------------------------------
_guidelines_store = None
_publications_store = None
_legacy_store = None

if os.path.exists("faiss_index_guidelines"):
    _guidelines_store = FAISS.load_local("faiss_index_guidelines", embeddings, allow_dangerous_deserialization=True)
    print("[RAG] ✅ Loaded guidelines FAISS index")

if os.path.exists("faiss_index_publications"):
    _publications_store = FAISS.load_local("faiss_index_publications", embeddings, allow_dangerous_deserialization=True)
    print("[RAG] ✅ Loaded publications FAISS index")

# Legacy fallback: if neither new index exists, try the old single index
if _guidelines_store is None and _publications_store is None and os.path.exists("faiss_index"):
    _legacy_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    _guidelines_store = _legacy_store
    print("[RAG] ⚠️  Using legacy single FAISS index (run Build_Index.py to create separate indexes)")

# Track which sources are available for the UI to display
AVAILABLE_SOURCES = []
if _guidelines_store is not None:
    AVAILABLE_SOURCES.append("guidelines")
if _publications_store is not None:
    AVAILABLE_SOURCES.append("publications")


# ---------------------------------------------------------------------------
# Multi-Query Prompt Template with Clinical & Multi-lingual Synonym Expansion
# ---------------------------------------------------------------------------
query_gen_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant specialized in clinical medical information retrieval for Nephrology guidelines and research publications.
Generate exactly 3 alternative search queries in English for the given clinical question.
If the question is in German or contains medical acronyms (e.g. OAK -> Oral Anticoagulation, ANV -> Acute Kidney Injury, RRT -> Renal Replacement Therapy), translate and expand them into standard English nephrology terminology.
Keep each search query concise, direct, and focused on guideline recommendations or research evidence.

Question: {question}

Alternative search queries:"""
)


# ---------------------------------------------------------------------------
# Reranker (shared)
# ---------------------------------------------------------------------------
compressor = FlashrankRerank(top_n=5)


def _build_retriever(vector_store):
    """Build a multi-query + reranked retriever from a FAISS vector store."""
    base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 25})
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=gwdg_llm, prompt=query_gen_prompt
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multi_query_retriever
    )


# ---------------------------------------------------------------------------
# Format retrieved docs with citations
# ---------------------------------------------------------------------------
def format_docs_with_citations(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_label", "?")
        source_type = doc.metadata.get("source_type", "unknown")
        tag = "[GUIDELINE]" if source_type == "guideline" else "[PUBLICATION]"
        formatted.append(f"{tag} Source: {source}, p.{page}\n{doc.page_content}")
    return "\n\n".join(formatted)


# ---------------------------------------------------------------------------
# Source-specific answer prompts
# ---------------------------------------------------------------------------
_PROMPT_GUIDELINES = PromptTemplate(
    template="""You are a clinical decision support assistant specialized in the KDIGO 2026 Clinical Practice Guideline for Acute Kidney Injury (AKI) and Acute Kidney Disease (AKD). This tool is for research and clinical decision support only.

Rules you must follow:
1. Answer ONLY using the provided reference documents below. Do not make up facts or use outside unverified claims.
2. If the context contains tabular data ([TABLE START]...[TABLE END]) or figure annotations ([FIGURE ANNOTATION]), incorporate their structured information clearly into your answer.
3. If the answer is not explicitly stated in the reference documents, state clearly that the KDIGO guideline excerpts provided do not contain this information.
4. For every clinical claim or recommendation, cite the exact source document and page number as [Source: filename, p.X] or [KDIGO 2026, p.X].
5. Present recommendations with their specific KDIGO grade (e.g., Recommendation 3.2.3, Grade 1B) whenever present in the context.

Reference documents:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

_PROMPT_PUBLICATIONS = PromptTemplate(
    template="""You are a clinical research assistant specialized in analyzing peer-reviewed medical publications, clinical trials, and meta-analyses in Nephrology. This tool is for research and clinical decision support only.

Rules you must follow:
1. Answer ONLY using the provided research publication excerpts below. Do not make up facts or use outside unverified claims.
2. If the context contains tabular data ([TABLE START]...[TABLE END]), incorporate their structured information clearly into your answer.
3. If the answer is not explicitly stated in the provided excerpts, state clearly that the available publication excerpts do not contain this information.
4. For every clinical finding or claim, cite the exact source document and page number as [Source: filename, p.X].
5. When reporting study results, mention the study design (RCT, cohort, meta-analysis), sample size, and key statistical findings (e.g., OR, HR, CI, p-value) when available in the context.
6. Clearly distinguish between primary endpoints and secondary/exploratory findings.

Research publication excerpts:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

_PROMPT_BOTH = PromptTemplate(
    template="""You are a comprehensive clinical decision support assistant that synthesizes evidence from both clinical practice guidelines and peer-reviewed research publications in Nephrology. This tool is for research and clinical decision support only.

Rules you must follow:
1. Answer ONLY using the provided reference documents below. Do not make up facts or use outside unverified claims.
2. The context contains excerpts tagged as [GUIDELINE] or [PUBLICATION]. Clearly distinguish between guideline recommendations and research findings in your answer.
3. If the context contains tabular data ([TABLE START]...[TABLE END]) or figure annotations ([FIGURE ANNOTATION]), incorporate their structured information clearly into your answer.
4. If the answer is not found in the provided excerpts, state clearly that the available evidence excerpts do not contain this information.
5. For guideline claims: cite as [KDIGO 2026, p.X] with the recommendation grade (e.g., Grade 1B) when available.
6. For research claims: cite as [Source: filename, p.X] with study design and key statistics when available.
7. Structure your answer to present guideline recommendations first, then supporting or contrasting research evidence.

Reference documents:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)


def _build_chain(retriever_obj, prompt):
    """Build a LCEL chain from a retriever and a prompt template."""
    return (
        RunnableParallel({
            "context": retriever_obj | RunnableLambda(format_docs_with_citations),
            "question": RunnablePassthrough()
        })
        | prompt
        | gwdg_llm
        | StrOutputParser()
    )


def _build_combined_retriever(retriever_a, retriever_b):
    """Retrieve from two sources and merge results before reranking."""
    def _merge_retrieve(question):
        docs_a = retriever_a.invoke(question) if retriever_a else []
        docs_b = retriever_b.invoke(question) if retriever_b else []
        # Interleave: guidelines first, then publications
        return docs_a + docs_b
    return RunnableLambda(_merge_retrieve)


# ---------------------------------------------------------------------------
# Build chains for each source type
# ---------------------------------------------------------------------------
_guidelines_retriever = _build_retriever(_guidelines_store) if _guidelines_store else None
_publications_retriever = _build_retriever(_publications_store) if _publications_store else None

guidelines_chain = _build_chain(_guidelines_retriever, _PROMPT_GUIDELINES) if _guidelines_retriever else None
publications_chain = _build_chain(_publications_retriever, _PROMPT_PUBLICATIONS) if _publications_retriever else None

# Combined chain: merges results from both sources
if _guidelines_retriever and _publications_retriever:
    _combined_retriever = _build_combined_retriever(_guidelines_retriever, _publications_retriever)
    combined_chain = _build_chain(_combined_retriever, _PROMPT_BOTH)
elif _guidelines_retriever:
    combined_chain = guidelines_chain
elif _publications_retriever:
    combined_chain = publications_chain
else:
    combined_chain = None


# ---------------------------------------------------------------------------
# Public API — called by app.py
# ---------------------------------------------------------------------------
def answer_question(question: str, source_type: str = "guidelines") -> str:
    """
    Runs the full RAG pipeline end-to-end and returns a plain string answer.

    Args:
        question: The clinical question to answer.
        source_type: One of "guidelines", "publications", or "both".
    """
    print(f"\n[RAG] Received question: '{question}' (source: {source_type})", flush=True)

    # Select the appropriate chain
    if source_type == "publications" and publications_chain:
        chain = publications_chain
    elif source_type == "both" and combined_chain:
        chain = combined_chain
    elif guidelines_chain:
        chain = guidelines_chain
    else:
        return (
            "⚠️ No FAISS index is available for the selected source type. "
            "Run `python Build_Index.py` to build the indexes, then restart the app."
        )

    try:
        print(f"[RAG] Retrieving relevant {source_type} chunks...", flush=True)
        ans = chain.invoke(question)
        print("[RAG] Generated answer successfully.", flush=True)
        return ans
    except Exception as e:
        err_msg = str(e)
        print(f"[RAG ERROR] {err_msg}", flush=True)
        if "500" in err_msg or "InternalServerError" in err_msg:
            return (
                "⚠️ **Remote API Server Error (HTTP 500):**  \n"
                "The AcademicCloud server (`chat-ai.academiccloud.de`) is currently experiencing an internal server issue. "
                "Please wait a few moments and try again."
            )
        elif "timeout" in err_msg.lower() or "ReadTimeout" in err_msg:
            return (
                "⚠️ **Request Timed Out:**  \n"
                "The remote API server took too long to respond. Please retry in a few moments."
            )
        return f"⚠️ **Error generating response:** `{err_msg}`"


# runs only when this file is executed directly, not when imported as a module
if __name__ == "__main__":
    question = "therapie Lymphozytäre Myokarditis (Virus-negativ)?"
    print(answer_question(question, source_type="guidelines"))
