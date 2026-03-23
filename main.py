
from app.rag.embeddings import EmbeddingManager
from app.rag.vector_db import FaissVectorStore
from app.rag.retriever import RAGRetriever
from app.rag.pipeline import AdvancedRAGPipeline, Reranker, LLMModel, PromptBuilder


def main():
    print("🚀 Starting RAG System...")

    # 1️⃣ Load vector store
    vector_store = FaissVectorStore()

    try:
        vector_store.load()
        print("✅ Vector store loaded")
    except:
        print("⚠️ No vector DB found. Building new one...")

    # 2️⃣ Initialize embedding manager
    embedding_manager = EmbeddingManager()

    # 3️⃣ Create retriever
    retriever = RAGRetriever(vector_store, embedding_manager)

    # 4️⃣ Initialize reranker
    reranker = Reranker()

    # 5️⃣ Initialize LLM
    llm = LLMModel()

    # 6️⃣ Prompt builder
    prompt_builder = PromptBuilder()

    # 7️⃣ Create RAG pipeline
    rag_pipeline = AdvancedRAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        prompt_builder=prompt_builder
    )

    print("✅ System Ready! Type 'exit' to quit.\n")

    # 🔥 Interactive CLI loop
    '''while True:
        question = input("\n💬 Ask a question: ")

        if question.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break
        '''

    question = "What is Due Diligence?"


    result = rag_pipeline.query(
        question,
        top_k=3,
        summarize=True
    )

    print("\n===== ANSWER =====")
    print(result["answer"])

    print("\n===== SUMMARY =====")
    print(result.get("summary"))

    print("\n===== SOURCES =====")
    for src in result.get("sources", []):
        print(src)


if __name__ == "__main__":
    main()