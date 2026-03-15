import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any, Tuple
from app.rag.vector_db import FaissVectorStore
from app.rag.embeddings import EmbeddingManager


load_dotenv()

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store: FaissVectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5,score_threshold: float = None):

        print(f"Retrieving documents for query: {query}")

        # BGE embedding models ke liye recommended prefix
        formatted_query = "query: " + query
        # generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([formatted_query]).astype('float32')

        results = self.vector_store.search(query_embedding, top_k)

        retrieved_docs = []

        for rank, r in enumerate(results):
            if score_threshold is None or r["distance"] <= score_threshold:

                retrieved_docs.append({
                "rank": rank + 1,
                "distance": float(r["distance"]),
                "content": r["metadata"].get("text", ""),
                "metadata": r["metadata"]
            })

        print(f"Retrieved {len(retrieved_docs)} documents")

        return retrieved_docs
            
class GroqLLM:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", api_key: str =None):
        """
        Initialize Groq LLM
        
        Args:
            model_name: Groq model name (qwen2-72b-instruct, llama3-70b-8192, etc.)
            api_key: Groq API key (or set GROQ_API_KEY environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )
        
        print(f"Initialized Groq LLM with model: {self.model_name}")

    def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
        """
        Generate response using retrieved context
        
        Args:
            query: User question
            context: Retrieved document context
            max_length: Maximum response length
            
        Returns:
            Generated response string
        """
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

                    Context:
                    {context}

                    Question: {question}

                    Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say no."""
                            )
        
        # Format the prompt
        formatted_prompt = prompt_template.format(context=context, question=query)
        
        try:
            # Generate response
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
        
    def generate_response_simple(self, query: str, context: str) -> str:
        """
        Simple response generation without complex prompting
        
        Args:
            query: User question
            context: Retrieved context
            
        Returns:
            Generated response
        """
        simple_prompt = f"""Based on this context: {context}

                    Question: {query}

                    Answer:"""
                            
        try:
            messages = [HumanMessage(content=simple_prompt)]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"
        
    def invoke(self, messages):
        return self.llm.invoke(messages)
'''   
    if __name__ == "__main__":
        vector_store = FaissVectorStore()
        vector_store.load()  
        embedding_manager = EmbeddingManager()
        
        rag_search = RAGRetriever(vector_store,embedding_manager)
        
        query = "What is attention mechanism?"
        
        summary = RAGRetriever.retrieve(rag_search,query,top_k=3)
        print("Summary:", summary)

'''