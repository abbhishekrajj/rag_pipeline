# --- Advanced RAG Pipeline: Streaming, Citations, History, Summarization ---
from typing import List, Dict, Any
import time
from app.rag.retriever import RAGRetriever,GroqLLM

class AdvancedRAGPipeline:

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []

    def query(self, question, top_k=5,min_score: float = 0.2):

        # Retrieve docs
        results = self.retriever.retrieve(question, top_k=top_k)

        print("Retrieved docs:", len(results))

        if not results:
            return {"answer": "No relevant context found.", "sources": []}

        # Build context
        context = "\n\n".join([
            doc.get("content") or getattr(doc, "page_content", "")
            for doc in results
        ])

        # Sources
        sources = [{
            "source": doc["metadata"].get("source_file", "unknown"),
            "page": doc["metadata"].get("page", "unknown"),
            'score': doc.get('similarity_score') or doc.get('distance'),
            'preview': (doc.get('content') or getattr(doc, 'page_content', ''))[:120] + '...'

        } for doc in results]
        

        # Call Groq LLM wrapper
        answer = self.llm.generate_response(
            query=question,
            context=context
        )
        # Add citations to answer
        citations_list = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        citations_str = "\n".join(citations_list)
        answer_with_citations = f"{answer}\n\nCitations:\n{citations_str}" if citations_list else answer
        
        
        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }

        

