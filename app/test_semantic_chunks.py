from document_loader import load_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from transformers import AutoTokenizer
import os
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import csv
from datetime import datetime
from pathlib import Path

# Test different chunking methods and retrieval methods
# Load environment variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


FILE_PATHS = [
    "../data/ih2o-ishares-global-water-ucits-etf-fund-fact-sheet-en-ch.pdf",
]


URLS = [
    "https://www.investopedia.com/water-etfs-how-they-work-8426968",
    "https://www.investopedia.com/precious-metals-etfs-8549823",
    "https://www.investopedia.com/treasury-exchange-traded-funds-8536147",
]

# Initialize models
login(token=HUGGINGFACE_TOKEN)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", token=HUGGINGFACE_TOKEN)
embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
llm = ChatMistralAI(model="open-mistral-7b", mistral_api_key=MISTRAL_API_KEY)

PROMPT_TEMPLATE = """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""

def token_length(text: str) -> int:
    """Calculate token length of text."""
    return len(tokenizer.encode(text))

def get_chunks(docs, method="regular"):
    """Split documents using specified method."""
    if method == "regular":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=40,
            length_function=token_length,
            is_separator_regex=False,
        )
    else:  # semantic
        splitter = SemanticChunker(
            embeddings=embeddings,
        )
    
    return splitter.split_documents(docs)

def get_retrievers(chunks):
    """Create different types of retrievers."""
    # Vector store retriever
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # BM25 (keyword) retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10
    
    # Hybrid retriever (combining both)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]  # 30% keyword, 70% semantic
    )
    
    return {
        "vector": vector_retriever,
        "keyword": bm25_retriever,
        "hybrid": hybrid_retriever
    }

RESULTS_DIR = Path("./test_results")
RESULTS_FILE = RESULTS_DIR / "retrieval_comparison.csv"

def init_results_file():
    """Initialize results CSV file if it doesn't exist."""
    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'query',
                'chunk_type',
                'retriever_type',
                'response',
                'num_docs_retrieved',
                'processing_time_ms'
            ])

def log_result(query: str, chunk_type: str, retriever_type: str, 
               response: str, num_docs: int, time_ms: float):
    """Log a single retrieval result to CSV."""
    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            query,
            chunk_type,
            retriever_type,
            response,
            num_docs,
            time_ms
        ])

def compare_chunks_and_generate(query: str):
    """Compare different chunking and retrieval methods."""
    init_results_file()
    
    # Load documents
    docs = load_documents(file_paths=FILE_PATHS, urls=URLS)
    
    # Get chunks using both methods
    regular_chunks = get_chunks(docs, "regular")
    semantic_chunks = get_chunks(docs, "semantic")
    
    print(f"\nQuery: {query}")
    print("=" * 100)
    
    # Create retrievers for both chunking methods
    regular_retrievers = get_retrievers(regular_chunks)
    semantic_retrievers = get_retrievers(semantic_chunks)
    
    # Create chain for answer generation
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context"
    )
    
    print("\nAnswer Comparison:")
    print("=" * 100)
    
    # Test each retrieval method
    for chunk_type, retrievers in [("Regular", regular_retrievers), ("Semantic", semantic_retrievers)]:
        print(f"\n{chunk_type} Chunking:")
        print("-" * 80)
        
        for retriever_type, retriever in retrievers.items():
            start_time = datetime.now()
            
            # Get relevant documents
            relevant_docs = retriever.get_relevant_documents(query)
            
            # Generate answer
            response = document_chain.invoke({
                "input": query,
                "context": relevant_docs
            })
            
            # Calculate processing time
            time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Log results
            log_result(
                query=query,
                chunk_type=chunk_type,
                retriever_type=retriever_type,
                response=response,
                num_docs=len(relevant_docs),
                time_ms=time_ms
            )
            
            print(f"\n{retriever_type.upper()} Search Answer:")
            print("-" * 40)
            print(response)

if __name__ == "__main__":
    test_queries = [
        "What is the investment objective of the ETF?",
        "What are the top holdings in the fund?",
        "What is the fund's expense ratio?",
    ]
    
    for query in test_queries:
        compare_chunks_and_generate(query) 