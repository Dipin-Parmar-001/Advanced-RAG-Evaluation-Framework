import re
import fitz
from langchain_core.documents import Document

def load_pdf(file_path: str) -> list[Document]:
    """
    Extracts text page-by-page from a PDF using PyMuPDF (fitz) and returns Document objects.
    """
    documents = []
    try:
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                metadata = {
                    "source": file_path,
                    "page_number": page_num
                }
                documents.append(Document(page_content=text, metadata=metadata))
        doc.close()
    except Exception as e:
        print(f"[Critical Error] PDF parsing failed entirely: {e}")
        raise RuntimeError(f"Could not extract text from this specific PDF file configuration: {str(e)}")
    return documents

def clean_text(documents: list[Document]) -> list[Document]:
    """
    Cleans raw document text by merging hyphenated words, removing page numbers,
    handling single newlines, and removing control characters.
    """
    cleaned_doc = []
    for doc in documents:
        text = doc.page_content

        # Merge hyphenated words broken across lines
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # Convert single newlines to spaces, leaving double newlines
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        # Normalize multiple newlines
        text = re.sub(r'\n+', '\n', text)
        # Remove ASCII control/non-printable characters
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        # Normalize spacing
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove page numbers and headers/footers
        text = re.sub(r'(?i)\bpage\s*\d+(\s*of\s*\d+)?\b', '', text)
        text = re.sub(r'\b\d+\s*\|\s*.*', '', text)
        text = text.strip()

        if text:
            cleaned_doc.append(Document(page_content=text, metadata=doc.metadata))
    return cleaned_doc

