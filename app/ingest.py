from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import json
import os


# Step 1: Extract text from PDF
def extract_text(pdf_path):
    if not os.path.exists(pdf_path):
        print("Error: PDF file not found!")
        print("Expected location:", os.path.abspath(pdf_path))
        return ""

    if os.path.getsize(pdf_path) == 0:
        print("Error: PDF file is empty!")
        return ""

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# Step 2: Create chunks
def create_chunks(text, chunk_size=500, overlap=100):
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# Step 3: Read PDF
pdf_path = "sample.pdf"

text = extract_text(pdf_path)

if not text:
    print("No text extracted from PDF.")
    exit()

print("Text extracted successfully.")


# Step 4: Create chunks
chunks = create_chunks(text)

print("Number of chunks:", len(chunks))

if not chunks:
    print("No chunks created.")
    exit()


# Step 5: Create embeddings
print("Creating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    chunks,
    convert_to_numpy=True
)

print("Embeddings created.")


# Step 6: Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS index created.")


# Step 7: Save vector database
os.makedirs("vectorstore", exist_ok=True)

faiss.write_index(
    index,
    "vectorstore/index.faiss"
)


# Save chunks
with open(
    "vectorstore/chunks.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        chunks,
        f,
        ensure_ascii=False,
        indent=2
    )


print("Vector database created successfully!")