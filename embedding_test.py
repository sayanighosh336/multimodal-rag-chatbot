from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

texts = [
    "Boolean Algebra",
    "Digital Logic Design"
]

embeddings = model.encode(texts)

print(embeddings.shape)