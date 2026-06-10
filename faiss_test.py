from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [
    "Boolean Algebra",
    "Digital Logic Design",
    "Java Programming"
]

embeddings = model.encode(texts)

index = faiss.IndexFlatL2(384)

index.add(
    np.array(embeddings).astype("float32")
)

print("Vectors stored:", index.ntotal)