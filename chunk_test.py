from pdf_reader import extract_pdf_text

def chunk_text(text, size=500):
    chunks = []

    for i in range(0, len(text), size):
        chunks.append(text[i:i+size])

    return chunks

pdf_text = extract_pdf_text("notes.pdf")

chunks = chunk_text(pdf_text)

print("Number of chunks:", len(chunks))
print()
print(chunks[0])