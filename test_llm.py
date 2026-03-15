from app.rag.retriever import GroqLLM

llm = GroqLLM()

response = llm.generate_response_simple(
    "What is machine learning?",
    "Machine learning is a subset of AI."
)

print(response)