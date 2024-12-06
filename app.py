import os
import warnings
from typing import List, Any

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI

import chainlit as cl

# Configuration
URLS = [
    "https://www.investopedia.com/water-etfs-how-they-work-8426968",
    "https://www.investopedia.com/precious-metals-etfs-8549823",
    "https://www.investopedia.com/treasury-exchange-traded-funds-8536147",
]

PROMPT_TEMPLATE = """Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}"""

class RAGPipeline:
    def __init__(self):
        # Disable warnings
        warnings.filterwarnings('ignore')
        
        # Load environment variables
        load_dotenv()
        self.mistral_api_key = os.getenv("MISTRAL_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Initialize core components
        self._init_models()
        
    def _init_models(self) -> None:
        """Initialize LLM, embeddings, and tokenizer."""
        login(token=self.huggingface_token)
        
        self.llm = ChatMistralAI(
            model="open-mistral-7b", 
            mistral_api_key=self.mistral_api_key
        )
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed", 
            mistral_api_key=self.mistral_api_key
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1", 
            token=self.huggingface_token
        )

    def _token_length(self, text: str) -> int:
        """Calculate token length of text."""
        return len(self.tokenizer.encode(text))

    def _split_documents(self, urls: List[str]) -> List[Any]:
        """Load and split documents into chunks."""
        # Load documents
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=self._token_length,
            is_separator_regex=False,
        )

        return splitter.split_documents(docs_list)

    def create_retrieval_chain(self, urls: List[str]) -> Any:
        """Create the full RAG retrieval chain."""
        # Process documents
        chunked_docs = self._split_documents(urls)
        
        # Create vector store and retriever
        vector_store = FAISS.from_documents(chunked_docs, self.embeddings)
        base_retriever = vector_store.as_retriever(
            search_kwargs={"k": 3, "fetch_k": 10}
        )
        
        # Create compression retriever
        compressor = LLMChainExtractor.from_llm(self.llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )

        # Create chain
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        return create_retrieval_chain(retriever, document_chain)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

@cl.on_chat_start
async def start():
    """Initialize chat session."""
    retrieval_chain = rag_pipeline.create_retrieval_chain(URLS)
    cl.user_session.set("retrieval_chain", retrieval_chain)
    await cl.Message(content="Welcome! Ask me anything about ETFs!").send()

@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    retrieval_chain = cl.user_session.get("retrieval_chain")
    response = await retrieval_chain.ainvoke({"input": message.content})
    await cl.Message(content=response["answer"]).send()
