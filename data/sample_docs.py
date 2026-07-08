SAMPLE_DOCS = {
    "on_prem_ai": (
        "On-premises AI deployment means running models entirely within an "
        "organization's own infrastructure, with no data sent to external "
        "APIs. This matters for telecom and financial workloads where "
        "customer data cannot leave internal systems for regulatory or "
        "contractual reasons. Ollama and locally-hosted Hugging Face models "
        "make this practical by letting teams run capable open-source LLMs "
        "on their own servers or even laptops."
    ),
    "ollama_basics": (
        "Ollama runs open-source large language models such as Llama, "
        "Mistral, and Phi locally, exposing a simple REST API on port 11434. "
        "You pull a model once with `ollama pull llama3`, then call "
        "/api/generate to get completions without any internet dependency "
        "at inference time."
    ),
    "rag_privacy": (
        "Combining RAG with a local model preserves both grounding and "
        "privacy: retrieval finds relevant internal documents using local "
        "embeddings or TF-IDF, and generation happens on a model running on "
        "the same machine, so neither the documents nor the user's query "
        "are ever sent to a third-party API."
    ),
}
