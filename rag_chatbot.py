from sentence_transformers import SentenceTransformer
import os
import faiss
import numpy as np
import google.generativeai as genai

from pdf_reader import extract_pdf_text


# Read PDF
pdf_text = extract_pdf_text("notes.pdf")


# Chunk PDF
def chunk_text(text, size=500):
    chunks = []

    for i in range(0, len(text), size):
        chunks.append(text[i:i + size])

    return chunks


texts = chunk_text(pdf_text)

print("Total chunks:", len(texts))


# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)


# Store embeddings in FAISS
index = faiss.IndexFlatL2(384)

index.add(
    np.array(embeddings).astype("float32")
)


# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini = genai.GenerativeModel(
    "gemini-2.5-flash"
)


# Chat loop
while True:

    question = input("\nAsk Question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Convert question to embedding
    query_embedding = model.encode([question])

    # Retrieve top 3 chunks
    D, I = index.search(
        np.array(query_embedding).astype("float32"),
        3
    )

    # Combine retrieved chunks
    context = "\n".join(
        [texts[i] for i in I[0]]
    )

    # Prompt for Gemini
    prompt = f"""
Context:
{context}

Question:
{question}

Answer based only on the context provided.
"""

    # Generate answer
    response = gemini.generate_content(prompt)

    print("\nAnswer:")
    print(response.text)