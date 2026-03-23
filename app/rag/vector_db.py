import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from app.rag.embeddings import EmbeddingManager

BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str = os.path.join(BASE_DIR, "vector_store"),
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model
        self.model = None #SentenceTransformer(embedding_model)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
         # 🔥 Embedding pipeline use karo
        emb_pipe = EmbeddingManager(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )        
        # 1️⃣ Chunking
        chunks = emb_pipe.chunk_documents(documents)
        print(f"[INFO] Total chunks: {len(chunks)}")

        # 2️⃣ Embedding
        embeddings = emb_pipe.embed_chunks(chunks)

        # 3️⃣ Metadata
        metadatas = [
            {
                "content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]


        # 4️⃣ Add to FAISS
        self.add_embeddings(np.array(embeddings).astype("float32"), metadatas)

        # 5️⃣ Save
        self.save()

        print(f"[INFO] Vector store built successfully!")

# ================= ADD =================
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]

        # 🔥 Normalize (cosine similarity)
        faiss.normalize_L2(embeddings)

        # Create index if not exists
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors")

    # ================= SAVE =================
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        print("[INFO] Saving FAISS to:", os.path.abspath(faiss_path))

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print("[INFO] Save complete")

    # ================= LOAD =================
    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_path}")

        self.index = faiss.read_index(faiss_path)

        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)

        print("[INFO] Loaded FAISS index")

    # ================= SEARCH =================
    def search(self, query_embedding: np.ndarray, top_k: int = 5, min_score: float = 0.5):
        D, I = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            if score < min_score:
                continue

            meta = self.metadata[idx] if idx < len(self.metadata) else None

            results.append({
                "index": int(idx),
                "score": float(score),
                "content": meta["content"] if meta else None,
                "metadata": meta["metadata"] if meta else None,
                "type": "faiss"
            })

        return results

    # ================= QUERY =================
    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Query: {query_text}")

        # 1️⃣ Query embedding
        #query_emb = self.model.encode([query_text]).astype("float32")

        self.emb_pipe = EmbeddingManager(model_name=self.embedding_model)
        query_emb = self.emb_pipe.model.encode([query_text]).astype("float32")

        # 2️⃣ Normalize
        faiss.normalize_L2(query_emb)

        # 3️⃣ Search
        return self.search(query_emb, top_k=top_k)

    # ================= OPTIONAL =================
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query]).astype("float32")
    
'''
if __name__ == "__main__":
    from document_processor import DocumentProcess

    loader = DocumentProcess(data_dir="data")
    docs = loader.load_all_data()

    store = FaissVectorStore()
    store.build_from_documents(docs)

    store.load()

    results = store.query("What is Vitamin D?", top_k=3)

    print("\nResults:")
    for r in results:
        print(r["score"], r["content"][:100])

'''