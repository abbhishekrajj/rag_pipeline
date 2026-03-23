from dotenv import load_dotenv
import faiss
from rank_bm25 import BM25Okapi
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

        # 🔥 IMPORTANT: normalize
        faiss.normalize_L2(query_embedding)
        
        results = self.vector_store.search(query_embedding, top_k)

        retrieved_docs = []

        for rank, r in enumerate(results):
            if score_threshold is None or r["score"] >= score_threshold:

                retrieved_docs.append({
                "rank": rank + 1,
                "score": float(r["score"]),
                "content": r["content"],
                "metadata": r["metadata"]
            })

        print(f"Retrieved {len(retrieved_docs)} documents")

        return retrieved_docs
    
    @staticmethod
    def merge_results(bm25_results, faiss_results):
        combined = []

        for r in bm25_results:
            combined.append({
                "content": r["content"],
                "score": r["score"],
                "source": "bm25"
            })

        for r in faiss_results:
            combined.append({
                "content": r["content"],
                "score": r["score"],
                "source": "faiss"
            })

        return combined
    
class BM25Retriever:
    def __init__(self, documents):
        import re
        self.texts = [doc["content"] for doc in documents]
        tokenized = [re.findall(r"\w+", text.lower()) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=5):
        import re
        tokenized_query = re.findall(r"\w+", query.lower())

        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        return [
            {
                "content": self.texts[idx],
                "score": float(score),
                "type": "bm25"
            }
            for idx, score in ranked[:top_k]
        ]


class HybridScore:
    def __init__(self, faiss_store, bm25_retriever, reranker):
        self.faiss = faiss_store
        self.bm25 = bm25_retriever
        self.reranker = reranker

    def query(self, question):
        bm25_results = self.bm25.search(question, top_k=5)
        faiss_results = self.faiss.query(question, top_k=5)

        combined = RAGRetriever.merge_results(bm25_results, faiss_results)

        final = self.reranker.rerank(question, combined, top_k=5)

        return final
'''
if __name__ == "__main__":
    #from app.rag.vector_db import FaissVectorStore
    #from app.rag.embeddings import EmbeddingManager

    store = FaissVectorStore()
    store.load()

    emb = EmbeddingManager()

    retriever = RAGRetriever(store, emb)

    results = retriever.retrieve("What is Due Diligence?", top_k=3)

    print("\nRetrieved Docs:")
    for r in results:
        print(r["score"], r["content"][:100])
'''