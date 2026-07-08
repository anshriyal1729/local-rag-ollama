import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.sample_docs import SAMPLE_DOCS
from src.ollama_client import OllamaClient
from src.rag import LocalRAGPipeline


def build_pipeline():
    pipeline = LocalRAGPipeline(chunk_size=60, overlap=10)
    pipeline.ingest(SAMPLE_DOCS)
    return pipeline


def test_ollama_reports_unavailable_when_not_running():
    # No Ollama server is expected to be running in CI/test environments.
    client = OllamaClient(host="http://localhost:11434")
    assert client.is_available() is False


def test_falls_back_to_extractive_answer_without_ollama():
    pipeline = build_pipeline()
    result = pipeline.answer("What port does Ollama use?")
    assert result["generated_by"] == "extractive_fallback"
    assert "11434" in result["answer"]


def test_answer_includes_sources():
    pipeline = build_pipeline()
    result = pipeline.answer("Why deploy models on-premises?")
    assert len(result["sources"]) > 0


def test_out_of_scope_query_has_no_sources():
    pipeline = build_pipeline()
    result = pipeline.answer("banana smoothie recipe ingredients")
    assert result["sources"] == []
