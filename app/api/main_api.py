from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# -----------------------------
# ENV Setup
# -----------------------------
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG_APP"

app = FastAPI()

# -----------------------------
# RAG Imports
# -----------------------------
from app.rag.vector_db import FaissVectorStore
from app.rag.embeddings import EmbeddingManager
from app.rag.retriever import RAGRetriever
from app.rag.pipeline import AdvancedRAGPipeline, Reranker, LLMModel, PromptBuilder

from app.evaluation.ragas_eval import RAGASEvaluator

# -----------------------------
# Load / Build Vector DB
# -----------------------------
vector_store = FaissVectorStore()

try:
    vector_store.load()
    print("✅ Vector DB loaded")
except Exception as e:
    print("⚠️ Building vector DB...")

    from app.rag.document_processor import DocumentProcessor
    loader = DocumentProcessor(data_dir="data")
    docs = loader.load_all_data()

    vector_store.build_from_documents(docs)
    print("✅ Vector DB built successfully")

# -----------------------------
# Initialize Components
# -----------------------------
embedding_manager = EmbeddingManager()
retriever = RAGRetriever(vector_store, embedding_manager)

reranker = Reranker()
llm = LLMModel()
prompt_builder = PromptBuilder()

rag_pipeline = AdvancedRAGPipeline(
    retriever,
    reranker,
    llm,
    prompt_builder
)

evaluator = RAGASEvaluator()

# -----------------------------
# Request Schema
# -----------------------------
class QueryRequest(BaseModel):
    question: str
    evaluate: bool = False


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "This is Response from API"}


@app.post("/ask")
def query_api(req: QueryRequest):
    try:
        # 🔹 Run pipeline
        result = rag_pipeline.query(
            req.question,
            top_k=3,
            summarize=True
        )

        # 🔴 Safety check
        if result is None:
            return {"error": "RAG pipeline returned None"}

        # -----------------------------
        # Debug Logs
        # -----------------------------
        print("QUESTION:", result.get("question"))
        print("ANSWER:", result.get("answer", "")[:200])

        sources = result.get("sources", []) or []

        # 🔹 Extract contexts
        contexts = [
            src.get("content", "")
            for src in sources
            if src.get("content")
        ]

        print("CONTEXT COUNT:", len(contexts))
 
        # -----------------------------
        # RAGAS Evaluation
        # -----------------------------
        if req.evaluate:
            if not contexts:
                result["evaluation"] = {"error": "No context"}
            else:
                clean_answer = result.get("answer", "").split("Citations")[0]

                eval_result = evaluator.evaluate_response(
                    question=result.get("question"),
                    answer=clean_answer,
                    contexts=contexts
                )

                result["evaluation"] = eval_result

        return result

    except Exception as e:
        return {
            "error": str(e),
            "message": "Something went wrong"
        }