from typing import List, Any,Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import numpy as np


class EmbeddingManager:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    '''def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings'''
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts,batch_size=32, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _ensure_documents(self, documents):
        # Agar already Document hai → return as it is
        if len(documents) > 0 and hasattr(documents[0], "page_content"):
            return documents

        # Agar dict hai → convert karo
        converted = []
        for d in documents:
            converted.append(
                Document(
                    page_content=d.get("content", ""),
                    metadata=d.get("metadata", {})
                )
            )
        return converted
    
    def process_documents(self, documents: List[Any]) -> Tuple[List[Any], np.ndarray]:

        documents = self._ensure_documents(documents)

        # chunk documents
        chunks = self.chunk_documents(documents)

        # generate embeddings
        embeddings = self.embed_chunks(chunks)

        return chunks, embeddings
    

'''
# Example usage
if __name__ == "__main__":
    from document_processor import DocumentProcess

    loader = DocumentProcess(data_dir="data")
    docs = loader.load_all_data()

    emb = EmbeddingManager()
    chunks, embeddings = emb.process_documents(docs)

    print("Chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)

'''