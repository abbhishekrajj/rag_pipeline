from fastapi import FastAPI
from pydantic import BaseModel
from app.rag.vector_db import FaissVectorStore 
from app.rag.embeddings import EmbeddingManager
from app.rag.retriever import RAGRetriever, GroqLLM
from app.rag.pipeline import AdvancedRAGPipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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

# request structure
class QueryRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "This is Response from API"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask")
def ask_question(request: QueryRequest):

    question = request.question

    answer = rag_pipeline.query(question)

    return {
        "question": question,
        "answer": answer
    }