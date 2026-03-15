from app.rag.retriever import RAGRetriever
from app.rag.vector_db import FaissVectorStore
from app.rag.embeddings import EmbeddingManager

vector_store = FaissVectorStore()
vector_store.load()

embedding_manager = EmbeddingManager()

retriever = RAGRetriever(vector_store, embedding_manager)

results = retriever.retrieve("What Redcliffe lab?")

print(results)