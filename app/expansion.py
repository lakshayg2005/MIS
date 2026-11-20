"""Document expansion ("doc2query"): at INGESTION, ask an LLM for the questions employees might type that this
chunk answers, and embed them together with the chunk. Query time costs nothing extra (no LLM call per question).
Questions are cached by chunk text, so an unchanged chunk never calls the LLM again and its hash stays stable
(which is what lets incremental ingestion keep reusing vectors)."""
import hashlib
import json
import re
from pathlib import Path

from .llm import LLMError

SYSTEM = ("You help search a corporate transport policy. Given one policy section, write 5 short questions an employee might "
          "type that this section answers. Use everyday wording and slang an employee would use (not the policy's own words), "
          "include one with a typo-free short keyword style. Reply ONLY with a JSON list of 5 strings.")


class Doc2Query:
    def __init__(self, llm, cache_path: str, sleep_s: float = 0.0):
        self._llm, self._path, self._sleep = llm, Path(cache_path), sleep_s
        self._cache = json.loads(self._path.read_text()) if self._path.exists() else {}

    def questions(self, title: str, text: str) -> list:
        key = hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()
        if key not in self._cache:
            try:
                raw = self._llm.complete(SYSTEM, f"Section: {title}\n{text}", 20, json_mode=False)
                qs = [str(q) for q in json.loads(re.search(r"\[.*\]", raw, re.S).group(0))][:6]
            except (LLMError, ValueError, AttributeError, json.JSONDecodeError):
                return []  # no expansion for this chunk (not cached, so it is retried next time)
            self._cache[key] = qs
            self._path.write_text(json.dumps(self._cache, indent=1))
            if self._sleep:
                import time
                time.sleep(self._sleep)
        return self._cache[key]
