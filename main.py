from app.rag.embeddings import EmbeddingManager
from app.rag.vector_db import FaissVectorStore
from app.rag.retriever import RAGRetriever
from app.rag.pipeline import AdvancedRAGPipeline, Reranker, LLMModel, PromptBuilder

# 🔥 RAGAS
from app.evaluation.ragas_eval import RAGASEvaluator


def main():
    print("🚀 Starting RAG System...")

    # -----------------------------
    # 1️⃣ Load vector store
    # -----------------------------
    vector_store = FaissVectorStore()

    try:
        vector_store.load()
        print("✅ Vector store loaded")
    except:
        print("⚠️ No vector DB found. Please build it first.")
        return

    # -----------------------------
    # 2️⃣ Initialize components
    # -----------------------------
    embedding_manager = EmbeddingManager()
    retriever = RAGRetriever(vector_store, embedding_manager)
    reranker = Reranker()
    llm = LLMModel()
    prompt_builder = PromptBuilder()

    rag_pipeline = AdvancedRAGPipeline(
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        prompt_builder=prompt_builder
    )

    evaluator = RAGASEvaluator()

    print("✅ System Ready!\n")

    # -----------------------------
    # 🔥 Test Question
    # -----------------------------
    question = "What is Due Diligence?"

    result = rag_pipeline.query(
        question,
        top_k=3,
        summarize=True
    )

    # -----------------------------
    # Output
    # -----------------------------
    print("\n===== ANSWER =====")
    print(result.get("answer"))

    print("\n===== SUMMARY =====")
    print(result.get("summary"))

    print("\n===== SOURCES =====")
    for src in result.get("sources", []):
        print(src)

    # -----------------------------
    # 🔥 DEBUG (IMPORTANT)
    # -----------------------------
    contexts = [
        src.get("content", "")
        for src in result.get("sources", [])
        if src.get("content")
    ]

    print("\n===== DEBUG =====")
    print("CONTEXT COUNT:", len(contexts))
    print("FIRST CONTEXT:", contexts[:1])

    # -----------------------------
    # 🔥 RAGAS EVALUATION
    # -----------------------------
    if contexts:
        clean_answer = result.get("answer", "").split("Citations")[0]

        eval_result = evaluator.evaluate_response(
            question=result.get("question"),
            answer=clean_answer,
            contexts=contexts
        )

        print("\n===== EVALUATION =====")
        print(eval_result)
    else:
        print("\n⚠️ No context found → RAGAS cannot run")


if __name__ == "__main__":
    main()