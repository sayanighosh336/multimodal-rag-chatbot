import streamlit as st
import os
from sentence_transformers import SentenceTransformer
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
        chunks.append(text[i:i+size])

    return chunks

texts = chunk_text(pdf_text)

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)

# FAISS
index = faiss.IndexFlatL2(384)

index.add(
    np.array(embeddings).astype("float32")
)

# Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# Streamlit UI
st.title("Multimodal RAG Chatbot")

question = st.text_input("Ask a Question")

if st.button("Submit"):

    query_embedding = model.encode([question])

    D, I = index.search(
        np.array(query_embedding).astype("float32"),
        3
    )

    context = "\n".join(
        [texts[i] for i in I[0]]
    )

    prompt = f"""
    Context:
    {context}

    Question:
    {question}
    """

    response = gemini.generate_content(prompt)

    st.write(response.text)