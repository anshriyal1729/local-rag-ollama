"""Run this to see the local/on-prem RAG pipeline in action.

Works with or without Ollama running:
- If `ollama serve` is running with a pulled model, answers are generated
  by the local LLM.
- Otherwise, it automatically falls back to an extractive answer so the
  demo still runs end-to-end with zero setup.
"""
from data.sample_docs import SAMPLE_DOCS
from src.rag import LocalRAGPipeline


def main() -> None:
    pipeline = LocalRAGPipeline(ollama_model="llama3", chunk_size=60, overlap=10)
    pipeline.ingest(SAMPLE_DOCS)

    queries = [
        "Why deploy AI models on-premises instead of using a cloud API?",
        "What port does Ollama expose its API on?",
        "How does combining RAG with a local model help with privacy?",
    ]

    for query in queries:
        result = pipeline.answer(query)
        print("=" * 80)
        print(f"QUERY: {query}")
        print(f"ANSWER ({result['generated_by']}): {result['answer']}")


if __name__ == "__main__":
    main()
