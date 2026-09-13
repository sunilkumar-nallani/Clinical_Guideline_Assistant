import os
from dotenv import load_dotenv
import pdfplumber
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="e5-mistral-7b-instruct",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_base=os.environ["OPENAI_API_BASE"],
    tiktoken_enabled=False,
    check_embedding_ctx_length=False
)

def is_reference_page(page_text):
    """Density-based reference filter to avoid dropping inline citation pages."""
    lines = [line.strip() for line in page_text.split('\n') if line.strip()]
    if not lines:
        return False
        
    # Check for dedicated reference section headers at the top
    first_lines_upper = " ".join(lines[:3]).upper()
    if "REFERENCES" in first_lines_upper or "BIBLIOGRAPHY" in first_lines_upper:
        return True
        
    citation_markers = 0
    for line in lines:
        # Matches lines starting with number+period or brackets, common in bibliographies
        if (len(line) > 1 and line[0].isdigit() and (line[1:3] == '. ' or line[1:2] == '.')) or \
           line.startswith('[') or \
           "doi:10." in line or \
           "et al." in line:
            citation_markers += 1
            
    # If more than 60% of the lines appear to be pure citations, drop the page
    if (citation_markers / len(lines)) > 0.6:
        return True
        
    return False

def extract_pdf_with_tables(pdf_path, source_type="guideline"):
    """Extract text and tables from a PDF, tagging each page with source_type metadata."""
    documents = []
    
    # Use PyMuPDF to extract logical page labels (handles Roman numerals)
    try:
        doc_fitz = fitz.open(pdf_path)
        page_labels = doc_fitz.get_page_labels()
    except Exception:
        page_labels = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_label = page_labels[i] if i < len(page_labels) else str(i + 1)
            
            # Extract standard text
            text = page.extract_text() or ""
            
            # Extract tables as markdown grids
            tables = page.extract_tables()
            for table in tables:
                if not table: continue
                md_table = "\n\n[TABLE START]\n"
                for row_idx, row in enumerate(table):
                    clean_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                    md_table += "| " + " | ".join(clean_row) + " |\n"
                    # Add markdown table separator after header
                    if row_idx == 0:
                        md_table += "|" + "|".join(["---"] * len(clean_row)) + "|\n"
                md_table += "[TABLE END]\n\n"
                text += md_table
                
            if not is_reference_page(text):
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page_label": page_label,
                        "source_type": source_type,
                    }
                ))
    return documents

def load_figure_annotations(annotations_path="figure_annotations.json"):
    """
    Loads manually written figure/flowchart annotations from a JSON file
    and converts each one into a LangChain Document. These are injected into
    the FAISS index so that flowchart content is retrievable.
    """
    import json
    if not os.path.exists(annotations_path):
        print(f"[INFO] No figure_annotations.json found at '{annotations_path}'. Skipping figure injection.")
        return []

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    docs = []
    for ann in annotations:
        content = (
            f"[FIGURE ANNOTATION]\n"
            f"Figure: {ann.get('title', '')}\n"
            f"Source: {ann.get('source', '')}, p.{ann.get('page', '?')}\n"
            f"Review Status: {ann.get('review_status', 'UNKNOWN')}\n\n"
            f"{ann.get('description', '')}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source": ann.get("source", "figure_annotations.json"),
                "page_label": ann.get("page", "?"),
                "figure_id": ann.get("figure_id", "unknown"),
                "chunk_type": "figure_annotation",
                "source_type": "guideline",
            }
        ))
    print(f"Loaded {len(docs)} figure annotations from '{annotations_path}'")
    return docs


def _build_single_index(pdf_folder, source_type, save_path, figure_annotations_path=None):
    """Build a FAISS index from all PDFs in a folder with the given source_type tag."""
    all_documents = []

    if not os.path.exists(pdf_folder):
        print(f"[WARN] Directory '{pdf_folder}' not found. Skipping {source_type} index.")
        return False

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"[INFO] No PDFs found in '{pdf_folder}'. Skipping {source_type} index.")
        return False

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_folder, filename)
        docs = extract_pdf_with_tables(pdf_path, source_type=source_type)
        all_documents.extend(docs)

    print(f"[{source_type.upper()}] Loaded {len(all_documents)} content pages from {pdf_folder}")

    # Inject figure/flowchart annotations only for guidelines
    if figure_annotations_path and source_type == "guideline":
        figure_docs = load_figure_annotations(figure_annotations_path)
        all_documents.extend(figure_docs)
        print(f"[{source_type.upper()}] Total documents before chunking (pages + figure annotations): {len(all_documents)}")

    # Structural Splitting (preserving markdown tables)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"[{source_type.upper()}] Created {len(chunks)} structural chunks")

    # Creating Embeddings and store in FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    print(f"[{source_type.upper()}] Saved FAISS index to {save_path}/")
    return True


def build_and_save_index(
    guidelines_folder="documents/guidelines/",
    publications_folder="documents/publications/",
    guidelines_index_path="faiss_index_guidelines",
    publications_index_path="faiss_index_publications",
    # Legacy support: also check the old folder location
    legacy_guidelines_folder="Nephrology_Guidelines/",
):
    """
    Build separate FAISS indexes for guidelines and publications.

    - Guidelines: clinical practice guideline PDFs + figure annotations
    - Publications: research papers, clinical trials, meta-analyses
    """
    print("=" * 60)
    print("Building FAISS indexes for Guidelines and Publications")
    print("=" * 60)

    # --- Guidelines Index ---
    # Try new folder first, fall back to legacy folder
    gf = guidelines_folder if os.path.exists(guidelines_folder) else legacy_guidelines_folder
    g_built = _build_single_index(
        pdf_folder=gf,
        source_type="guideline",
        save_path=guidelines_index_path,
        figure_annotations_path="figure_annotations.json",
    )
    if g_built:
        print(f"\n✅ Guidelines index built successfully → {guidelines_index_path}/")
    else:
        print(f"\n⚠️  No guidelines found. Add PDFs to '{guidelines_folder}' and re-run.")

    print()

    # --- Publications Index ---
    p_built = _build_single_index(
        pdf_folder=publications_folder,
        source_type="publication",
        save_path=publications_index_path,
    )
    if p_built:
        print(f"\n✅ Publications index built successfully → {publications_index_path}/")
    else:
        print(f"\n⚠️  No publications found. Add PDFs to '{publications_folder}' and re-run.")

    print("\n" + "=" * 60)
    print("Index building complete.")
    print("=" * 60)


if __name__ == "__main__":
    build_and_save_index()