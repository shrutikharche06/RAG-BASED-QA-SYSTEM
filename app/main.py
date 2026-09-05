from retriever import retrieve
from generator import generate_answer


def main():

    print("================================")
    print("     RAG DOCUMENT ASSISTANT")
    print("================================")

    question = input("\nAsk a question: ")

    results = retrieve(
        question,
        k=3
    )

    context = "\n\n".join(results)

    answer = generate_answer(
        question,
        context
    )

    print("\nAnswer:")
    print("--------------------------------")
    print(answer)
    print("--------------------------------")

    print("\nRetrieved Sources:")

    for i, result in enumerate(results, 1):
        print(f"\nSource {i}:")
        print(result[:500])


if __name__ == "__main__":
    main()