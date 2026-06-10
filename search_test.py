from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

texts = [
    "Boolean Algebra is used in digital circuits",
    "Logic Gates perform logical operations",
    "Java is a programming language"
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)

index = faiss.IndexFlatL2(384)

index.add(np.array(embeddings).astype("float32"))

question = "What is Boolean Algebra?"

query_embedding = model.encode([question])

D, I = index.search(
    np.array(query_embedding).astype("float32"),
    1
)

print("Best Match:")
print(texts[I[0][0]])