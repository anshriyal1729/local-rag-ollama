"""Thin client for a locally-running Ollama server.

Ollama exposes a simple REST API on localhost:11434 by default, letting you
run open-source LLMs (Llama, Mistral, Phi, etc.) fully offline -- no data
ever leaves the machine, which matters for privacy-sensitive deployments.
"""
from __future__ import annotations

import requests


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server can't be reached."""


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3", timeout: float = 30.0):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Calls Ollama's /api/generate endpoint (non-streaming) and returns
        the generated text. Raises OllamaUnavailableError if the server
        can't be reached, so callers can fall back gracefully."""
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system

        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` running "
                f"and have you pulled the '{self.model}' model? (`ollama pull {self.model}`)"
            ) from exc

        return resp.json().get("response", "")
