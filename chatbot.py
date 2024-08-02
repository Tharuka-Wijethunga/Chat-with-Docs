import os
import logging
from typing import List, Tuple
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class cbfs:
    def __init__(self):
        self.chat_history: List[Tuple[str, str]] = []
        self.loaded_file = ""
        self.vectorstore = None
        
        # Initialize Mistral AI client
        self.mistral_api_key = os.environ.get("MISTRAL_API_KEY")
        if not self.mistral_api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        self.client = MistralClient(api_key=self.mistral_api_key)
        
        # Load the embedding model (using a free alternative)
        self.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def load_db(self, file: str, k: int = 4):
        try:
            # Load documents
            loader = PyPDFLoader(file)
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = text_splitter.split_documents(documents)
            
            # Create vector database from data
            self.vectorstore = FAISS.from_documents(docs, self.embed_model)
            
            self.loaded_file = file
            logger.info(f"Successfully loaded file: {file}")
            return f"Loaded file: {file}"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"

    def __call__(self, query: str):
        if self.vectorstore is None:
            return {"answer": "Please load a PDF file first.", "chat_history": self.chat_history}
        
        try:
            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search(query, k=2)
            context = " ".join([doc.page_content for doc in docs])
            
            # Prepare the messages for Mistral AI
            messages = [
                ChatMessage(role="system", content="You are a helpful assistant. Use the following context to answer the user's question."),
                ChatMessage(role="user", content=f"Context: {context}\n\nQuestion: {query}")
            ]
            
            # Get answer using Mistral AI
            chat_response = self.client.chat(
                model="mistral-tiny",  # or "mistral-small", "mistral-medium" depending on your needs
                messages=messages
            )
            
            answer = chat_response.choices[0].message.content
            
            self.chat_history.append((query, answer))
            
            return {
                "answer": answer,
                "chat_history": self.chat_history,
                "source_documents": docs
            }
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"answer": f"An unexpected error occurred: {str(e)}", "chat_history": self.chat_history}

    def clr_history(self):
        self.chat_history = []

# Test the Mistral AI connection
try:
    test_model = cbfs()
    logger.info("Mistral AI client initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Mistral AI client: {str(e)}")