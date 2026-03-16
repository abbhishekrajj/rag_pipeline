# RAG-Based AI Chatbot using FastAPI, Vector Database and LLM

This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer user queries using custom documents.  
It combines vector search with LLMs to provide accurate, context-aware responses.

Data → Loader → Chunking → Embedding → Vector DB

User Query
   ↓
Query Embedding
   ↓
Retriever
   ↓
Re-Ranker
   ↓
Prompt Engineering
   ↓
Context Window
   ↓
LLM
   ↓
Generated Answer

# Upcomoing Version

Devployment

## Future Improvements

- Hybrid search (BM25 + Vector search)
- Advanced re-ranking models
- Multi-document summarization
- UI interface

## [Architecture](doc/rag_architecture.png)

## Author

Abhishek Raj