"""LLM access behind one tiny interface: complete(system, user, timeout_s) -> str.

Groq and Gemini both expose an OpenAI-compatible /chat/completions endpoint, so ONE small client
(plain httpx, no SDK) covers both. `fake` is an offline extractive stand-in for tests and demos
without a key. Nothing here knows about companies or policies.
"""
import re
import time

import httpx


class LLMError(Exception):
    """Any failure to get text back from the model (network, quota, timeout, empty body)."""


PROVIDERS = {
    # qwen/qwen3.8-27b is the model the official evaluation ran on; override with LLM_MODEL (openai/gpt-oss-20b was used in earlier runs)
    "groq": ("https://api.groq.com/openai/v1", "qwen/qwen3.8-27b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash"),
}


class OpenAICompatLLM:
    def __init__(self, provider: str, api_key: str, model: str = "", transport=None):
        base, default_model = PROVIDERS[provider]
        if not api_key:
            raise ValueError(f"LLM_API_KEY is required for provider {provider!r}")
        self._model = model or default_model
        self._client = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {api_key}"}, transport=transport)

    def complete(self, system: str, user: str, timeout_s: float, json_mode: bool = True) -> str:
        """json_mode=True is for the answer step (strict JSON). Query rewriting / expansion pass False (free text)."""
        body = {"model": self._model, "temperature": 0, "max_tokens": 800,  # room for hidden reasoning tokens
                **({"reasoning_effort": "low"} if "gpt-oss" in self._model else {}),
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        start = time.monotonic()
        try:
            try:
                r = self._client.post("/chat/completions", json=body, timeout=max(timeout_s, 0.05))
            except httpx.ConnectError:
                # The connection never opened, so nothing reached the model and no tokens were used: safe to retry ONCE,
                # and only if there is time left. Timeouts and HTTP errors are NOT retried (the time or quota is already spent).
                left = timeout_s - (time.monotonic() - start)
                if left < 0.5:
                    raise
                time.sleep(0.15)
                r = self._client.post("/chat/completions", json=body, timeout=max(left - 0.15, 0.05))
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        if not isinstance(text, str) or not text.strip():
            raise LLMError("empty completion")
        return text


class ExtractiveLLM:
    """Offline fake: answers with the first excerpt if it shares enough words with the question.
    Deterministic, free, never hits a quota. It parses the same prompt the real model receives."""

    def complete(self, system: str, user: str, timeout_s: float, json_mode: bool = True) -> str:
        import json

        from .retrieval import tokenize

        q = re.search(r"<question>\n(.*?)\n</question>", user, re.S)
        ex = re.search(r"\[1\] .*?\n(.*?)(?=\n\[2\] |\n</policy_excerpts>)", user, re.S)
        if not q or not ex:
            return json.dumps({"status": "not_covered", "answer": "", "citations": []})
        overlap = set(tokenize(q.group(1))) & set(tokenize(ex.group(1)))
        if len(overlap) < 2:
            return json.dumps({"status": "not_covered", "answer": "", "citations": []})
        return json.dumps({"status": "answered", "answer": ex.group(1).strip(), "citations": [1]})


def make_llm(provider: str, api_key: str, model: str):
    if provider == "fake":
        return ExtractiveLLM()
    if provider in PROVIDERS:
        return OpenAICompatLLM(provider, api_key, model)
    raise ValueError(f"unknown LLM provider {provider!r}")
