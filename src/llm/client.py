"""Thin LLM wrapper. Anthropic API if ANTHROPIC_API_KEY is set, otherwise 'offline' mode (rule-based fallbacks everywhere).
Swap provider by editing this file only; nothing else in the project imports an LLM SDK."""
import os

from src.utils.config import COPILOT_MODEL


class LLM:
    def __init__(self, provider: str | None = None):
        self.provider = provider or ("anthropic" if os.getenv("ANTHROPIC_API_KEY") else "offline")
        self._client = None

    @property
    def available(self) -> bool:
        return self.provider != "offline"

    def complete(self, system: str, user: str, max_tokens: int = 1200) -> str:
        if self.provider == "offline":
            raise RuntimeError("LLM not configured (set ANTHROPIC_API_KEY).")
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        r = self._client.messages.create(model=COPILOT_MODEL, max_tokens=max_tokens, temperature=0, system=system,
                                         messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()


class MockLLM(LLM):
    """Scripted LLM for tests: responses is a list (popped in order) or a callable(system, user)->str."""
    def __init__(self, responses):
        super().__init__(provider="mock")
        self.responses, self.calls = responses, []

    def complete(self, system, user, max_tokens=1200):
        self.calls.append((system, user))
        return self.responses(system, user) if callable(self.responses) else self.responses.pop(0)
