# Module file (a file meant to be imported elsewhere)

# rag_pipeline.py  (runs every time someone asks a question)
#     │
#     │  FAISS.load_local("faiss_index", ...) → reads both files back into memory
#     ▼
# main_chain ready to answer questions instantly, no re-embedding needed

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiQueryRetriever

# Specialized model to filterout which chunks (in total 10) are highly releveant to question
# Fixed: Moved from langchain.retrievers to langchain_classic.retrievers
from langchain_classic.retrievers import ContextualCompressionRetriever

# Third-party integrations like Flashrank are imported from langchain_community
from langchain_community.document_compressors import FlashrankRerank

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="e5-mistral-7b-instruct",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    tiktoken_enabled=False,
    check_embedding_ctx_length=False,
    max_retries=2,
    request_timeout=20.0
)

# retreiveing embeddings from vector store and creating retriever for similarity search

# it opens that same faiss_index/ folder, reads both files back into memory, and rebuilds an identical FAISS vector store object, without needing to re-embed any documents or call the embedding API again.
# Saves Cost and Time
# allow_dangerous_deserialization=True is required because index.pkl uses Python's pickle format, which can technically execute arbitrary code if the file came from an untrusted source. Since you created this file yourself with your own trusted documents, this is safe — but never load a faiss_index/ folder you downloaded from someone else without inspecting it first.

# Base vector store retriever using MMR for diversity across the 500-page document
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 25})

# LLM for Query Expansion & Answer Generation
gwdg_llm = ChatOpenAI(
    model="qwen3-30b-a3b-instruct-2507",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    temperature=0,
    max_retries=2,
    request_timeout=30.0
)

# Multi-Query Prompt Template with Clinical & Multi-lingual Synonym Expansion
query_gen_prompt = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI assistant specialized in clinical medical information retrieval for Nephrology guidelines (KDIGO 2026).
Generate exactly 3 alternative search queries in English for the given clinical question.
If the question is in German or contains medical acronyms (e.g. OAK -> Oral Anticoagulation, ANV -> Acute Kidney Injury, RRT -> Renal Replacement Therapy), translate and expand them into standard English nephrology terminology.
Keep each search query concise, direct, and focused on guideline recommendations.

Question: {question}

Alternative search queries:"""
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever, llm=gwdg_llm, prompt=query_gen_prompt
)

# FlashRank Reranker: Re-scores all candidate chunks (up to 40) from multi-query and selects the top 5 most relevant
compressor = FlashrankRerank(top_n=5)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=multi_query_retriever
)

# Format retrieved docs with citations before they reach the prompt
def format_docs_with_citations(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page_label", "?")
        formatted.append(f"Source: {source}, p.{page}\n{doc.page_content}")
    return "\n\n".join(formatted)

# Prompt template for KDIGO Nephrology clinical guidance
answer_prompt = PromptTemplate(
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

main_chain = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs_with_citations),
        "question": RunnablePassthrough()
    })
    | answer_prompt
    | gwdg_llm
    | StrOutputParser()
)

# --- Convenience wrapper with explicit console output and error handling ---
def answer_question(question: str) -> str:
    """Runs the full RAG pipeline end-to-end and returns a plain string answer."""
    print(f"\n[RAG] Received question: '{question}'", flush=True)
    try:
        print("[RAG] Retrieving relevant guideline chunks...", flush=True)
        ans = main_chain.invoke(question)
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
    print(answer_question(question))
