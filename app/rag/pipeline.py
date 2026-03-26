# --- Advanced RAG Pipeline: Streaming, Citations, History, Summarization ---
from sentence_transformers import CrossEncoder
from typing import List, Dict
import os
from langchain_groq import ChatGroq
from app.rag.retriever import RAGRetriever

class Reranker:

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[INFO] Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list, top_k: int = 5):

        # 🔹 query-doc pairs
        pairs = [(query, d["content"]) for d in docs]

        # 🔹 scores
        scores = self.model.predict(pairs)

        # 🔹 attach score
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        # 🔹 sort
        reranked = sorted(
            docs,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]


class LLMModel:

    def __init__(self, model_name="llama-3.3-70b-versatile", temperature=0.3):
        
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("❌ API_KEY not found in environment variables")

        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=api_key   # 👈 explicitly pass
        )

    def generate(self, prompt: str):
        response = self.llm.invoke(prompt)
        return response.content
    
class PromptBuilder:

    def build(self, query: str, context: str):
        return f"""
        You are a compliance assistant.
        Answer ONLY from context.
        If not found → say "Not available".

        Context:
        {context}

        Question:
        {query}
        """
    
class AdvancedRAGPipeline:

    def __init__(self, retriever, reranker, llm, prompt_builder):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.history = []

    # 🔹 Context Builder
    def build_context(self, docs: List[Dict], max_chars: int = 2000):
        context = ""
        sources = []

        for d in docs:
            chunk = d["content"]

            if len(context) + len(chunk) > max_chars:
                break

            context += chunk + "\n\n"

            # source tracking
            sources.append({
                "source": d.get("metadata", {}).get("source_file", "unknown"),
                "page": d.get("metadata", {}).get("page_number", "N/A")
            })

        return context, sources

    # 🔹 Main Query Pipeline
    def query(
        self,
        question: str,
        top_k: int = 5,
        summarize: bool = True
    ):

        print(f"[INFO] Query: {question}")

        # 1️⃣ Retrieve
        docs = self.retriever.retrieve(question, top_k=top_k)

        if not docs:
            return {
                "answer": "No relevant context found.",
                "sources": [],
                "summary": None
            }

        # 2️⃣ Rerank
        docs = self.reranker.rerank(question, docs, top_k=top_k)

        # 3️⃣ Context
        context, sources = self.build_context(docs)

        # 4️⃣ Prompt
        prompt = self.prompt_builder.build(question, context)

        # 5️⃣ LLM Answer
        response = self.llm.generate(prompt)
        answer = response

        # 6️⃣ Citations
        citations_list = [
            f"[{i+1}] {src['source']} (page {src['page']})"
            for i, src in enumerate(sources)
        ]

        citations_str = "\n".join(citations_list)

        answer_with_citations = (
            f"{answer}\n\nCitations:\n{citations_str}"
            if citations_list else answer
        )

        # 7️⃣ Summarization
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"

            summary_resp = self.llm.generate(summary_prompt)
            summary = summary_resp

        # 8️⃣ History store
        self.history.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "summary": summary
        })

        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history
        }

'''       
if __name__ == "__main__":
    from vector_db import FaissVectorStore
    from embeddings import EmbeddingManager
    from retriever import RAGRetriever

    store = FaissVectorStore()
    store.load()

    emb = EmbeddingManager()
    retriever = RAGRetriever(store, emb)

    reranker = Reranker()
    llm = LLMModel()
    prompt = PromptBuilder()

    pipeline = AdvancedRAGPipeline(retriever, reranker, llm, prompt)

    result = pipeline.query("What is Due Diligence?", top_k=3)

    print("\nANSWER:\n", result["answer"])
'''