from app.rag.pipeline import AdvancedRAGPipeline
from app.rag.retriever import RAGRetriever
from app.rag.retriever import GroqLLM
from app.rag.vector_db import FaissVectorStore
from app.rag.embeddings import EmbeddingManager

vector_store = FaissVectorStore()
vector_store.load()
embedding_manager = EmbeddingManager()
retriever = RAGRetriever(vector_store, embedding_manager)
llm = GroqLLM() 

pipeline = AdvancedRAGPipeline(retriever, llm)

result = pipeline.query("What is machine learning?")

print(result)

# ✅ Define question
question = "What is Vitamin D?"
print("Query:", question)

results = retriever.retrieve(question)

print("Retriever results:", results)

context = "\n\n".join([doc['content'] for doc in results])

print("Context:", context)