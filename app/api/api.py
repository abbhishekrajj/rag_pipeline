from fastapi import FastAPI
from pydantic import BaseModel

from app.rag.vector_db import FaissVectorStore
from app.rag.embeddings import EmbeddingManager
from app.rag.retriever import RAGRetriever
from app.rag.pipeline import AdvancedRAGPipeline, Reranker, LLMModel, PromptBuilder

app = FastAPI()

 # 1️⃣ Load vector store
vector_store = FaissVectorStore()
try:
    vector_store.load()
    print("✅ Vector DB loaded")
except:
    print("⚠️ Building vector DB...")
    from app.rag.document_processor import DocumentProcessor
    loader = DocumentProcessor(data_dir="data")
    docs = loader.load_all_data()
    vector_store.build_from_documents(docs)

# 2️⃣ Initialize embedding manager
embedding_manager = EmbeddingManager()

# 3️⃣ Create retriever
retriever = RAGRetriever(vector_store, embedding_manager)

# 4️⃣ Initialize LLM
reranker = Reranker()
llm = LLMModel()
prompt_builder = PromptBuilder()

# 5️⃣ Create RAG pipeline
rag_pipeline = AdvancedRAGPipeline(
    retriever,
    reranker,
    llm,
    prompt_builder
)

# request structure
class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "This is Response from API"}

@app.post("/ask")
def ask_question(req: QueryRequest):
    result = rag_pipeline.query(req.question, top_k=3, summarize=True)
    return result