from app.rag.vector_db import FaissVectorStore 
from app.rag.embeddings import EmbeddingManager
from app.rag.retriever import RAGRetriever, GroqLLM
from app.rag.pipeline import AdvancedRAGPipeline

if __name__ == "__main__":

    # 1️⃣ Load vector store
    vector_store = FaissVectorStore()
    vector_store.load()

    # 2️⃣ Initialize embedding manager
    embedding_manager = EmbeddingManager()

    # 3️⃣ Create retriever
    retriever = RAGRetriever(vector_store, embedding_manager)

    # 4️⃣ Initialize LLM
    llm = GroqLLM()

    # 5️⃣ Create RAG pipeline
    rag_pipeline = AdvancedRAGPipeline(retriever, llm)

    # 6️⃣ Ask question
    question = "why Vitamin D is important"

    result = rag_pipeline.query(
        question,
        top_k=3,
        min_score=0.1,
        stream=False,
        summarize=True
    )

    print("\n===== QUESTION =====")
    print(result["question"])

    print("\n===== ANSWER =====")
    print(result["answer"])

    print("\n===== SUMMARY =====")
    print(result["summary"])

    print("\n===== SOURCES =====")
    for src in result["sources"]:
        print(src)