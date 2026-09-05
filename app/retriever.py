import faiss
import json
from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "vectorstore/index.faiss"
)

with open(
    "vectorstore/chunks.json",
    "r",
    encoding="utf-8"
) as f:
    chunks = json.load(f)


def retrieve(question, k=3):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for idx in indices[0]:

        if idx != -1:
            results.append(chunks[idx])

    return results