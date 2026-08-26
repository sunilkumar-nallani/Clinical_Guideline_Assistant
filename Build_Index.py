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

def extract_pdf_with_tables(pdf_path):
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
                    metadata={"source": os.path.basename(pdf_path), "page_label": page_label}
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
                "chunk_type": "figure_annotation"
            }
        ))
    print(f"Loaded {len(docs)} figure annotations from '{annotations_path}'")
    return docs


def build_and_save_index(pdf_folder="Nephrology_Guidelines/", save_path="faiss_index"):
    all_documents = []
    if not os.path.exists(pdf_folder):
        print(f"Directory '{pdf_folder}' not found. Please create it and add your PDFs.")
        return

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            docs = extract_pdf_with_tables(pdf_path)
            all_documents.extend(docs)

    print(f"Loaded {len(all_documents)} content pages from {pdf_folder}")

    # Inject figure/flowchart annotations to fill visual content gaps
    figure_docs = load_figure_annotations()
    all_documents.extend(figure_docs)
    print(f"Total documents before chunking (pages + figure annotations): {len(all_documents)}")


    # Structural Splitting (preserving markdown tables)
    # Using "\n\n" as the primary separator ensures we avoid breaking inside a [TABLE START]...[TABLE END] block
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Created {len(chunks)} structural chunks")

    # Creating Embeddings and store in Faiss vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    print(f"Saved FAISS index to {save_path}/")

if __name__ == "__main__":
    build_and_save_index()