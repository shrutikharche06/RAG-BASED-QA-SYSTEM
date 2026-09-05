from transformers import pipeline

# Free local model
generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    device=-1
)


def generate_answer(question, context):

    prompt = f"""You are a document question-answering assistant.

Answer the question using ONLY the information provided in the context.

If the answer is not present in the context, say:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        return_full_text=False
    )

    return result[0]["generated_text"].strip()