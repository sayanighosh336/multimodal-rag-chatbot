from pdf_reader import extract_pdf_text

pdf_text = extract_pdf_text("notes.pdf")

all_text = pdf_text

print(all_text[:1000])