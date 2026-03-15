from app.rag.embeddings import EmbeddingManager

embed = EmbeddingManager()

text = ["Machine learning is a part of AI"]

vec = embed.generate_embeddings(text)

print(vec.shape)