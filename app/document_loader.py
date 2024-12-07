from typing import Any, List, Optional
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

def load_pdf_documents(file_paths: Optional[List[str]] = None) -> List[Document]:
    """
    Load documents from PDF files.
    Args:
        file_paths: List of paths to PDF files
    Returns:
        List of loaded documents
    """
    if not file_paths:
        return []
    
    documents = []
    for path in file_paths:
        try:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
            print(f"Successfully loaded PDF: {path}")
        except Exception as e:
            print(f"Error loading PDF {path}: {str(e)}")
    
    return documents

def load_web_documents(urls: Optional[List[str]] = None) -> List[Document]:
    """
    Load documents from web URLs.
    Args:
        urls: List of URLs to load
    Returns:
        List of loaded documents
    """
    if not urls:
        return []
    
    documents = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
            print(f"Successfully loaded URL: {url}")
        except Exception as e:
            print(f"Error loading URL {url}: {str(e)}")
    
    return documents

def load_documents(file_paths: Optional[List[str]] = None, 
                  urls: Optional[List[str]] = None) -> List[Document]:
    """
    Load documents from both PDF files and web URLs.
    Args:
        file_paths: List of paths to PDF files
        urls: List of URLs to load
    Returns:
        Combined list of loaded documents
    """
    documents = []
    
    # Load PDFs if provided
    if file_paths:
        pdf_docs = load_pdf_documents(file_paths)
        documents.extend(pdf_docs)
        print(f"Loaded {len(file_paths)} PDF documents")
    
    # Load web documents if provided
    if urls:
        web_docs = load_web_documents(urls)
        documents.extend(web_docs)
        print(f"Loaded {len(urls)} web documents")
    
    if not documents:
        print("Warning: No documents were loaded. Check your file paths and URLs.")
    
    return documents
