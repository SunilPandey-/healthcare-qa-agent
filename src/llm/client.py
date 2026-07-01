"""Provider-agnostic LLM client with retries, timeouts, and JSON validation.

Design goals
------------
* **One interface, many providers.** The agent code only ever talks to
  :class:`LLMClient`; swapping Anthropic for OpenAI is a config change.
* **Graceful failure.** Transient API errors are retried with exponential
  backoff (tenacity). If the primary provider is exhausted and a secondary is
  configured, we fall back to it before giving up.
* **Trustworthy structured output.** ``complete_json`` validates the model's
  reply against a Pydantic schema and, on malformed JSON, re-prompts the model
  once with the parser error ("self-repair") before raising.
* **Offline mode.** A deterministic ``mock`` provider lets the whole agent loop
  run with no network and no API key (used by tests / CI / graders).
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings
from ..logging_config import get_logger

logger = get_logger("agent.llm")

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM call cannot be completed after all retries/fallbacks."""


class _TransientLLMError(RuntimeError):
    """Internal marker for errors that are worth retrying."""


# --------------------------------------------------------------------------- #
# Base client
# --------------------------------------------------------------------------- #
class LLMClient:
    """Base class defining the provider-agnostic interface."""

    provider: str = "base"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.active_model

    # -- provider-specific hook ------------------------------------------- #
    def _raw_complete(self, system: str, messages: List[dict]) -> str:
        raise NotImplementedError

    # -- public API ------------------------------------------------------- #
    def complete_text(self, system: str, messages: List[dict]) -> str:
        """Return a plain-text completion, with retries + backoff."""
        return self._complete_with_retry(system, messages)

    def complete_json(
        self,
        system: str,
        messages: List[dict],
        response_model: Type[T],
    ) -> T:
        """Return a completion parsed + validated into ``response_model``.

        On malformed output the model is re-prompted once with the parser error
        (a lightweight self-repair loop) before we give up.
        """
        schema_hint = (
            "\n\nRespond with a SINGLE valid JSON object only — no prose, no "
            "markdown fences. It MUST match this JSON schema:\n"
            f"{json.dumps(response_model.model_json_schema())}"
        )
        convo = list(messages)
        last_error = ""
        for attempt in range(2):
            raw = self._complete_with_retry(system + schema_hint, convo)
            try:
                return response_model.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                logger.warning(
                    "JSON validation failed (attempt %d/2): %s", attempt + 1, last_error
                )
                convo = convo + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid JSON for the schema. "
                            f"Error: {last_error}. Reply again with ONLY the corrected JSON."
                        ),
                    },
                ]
        raise LLMError(f"Could not obtain valid {response_model.__name__} JSON: {last_error}")

    # -- retry wrapper ---------------------------------------------------- #
    def _complete_with_retry(self, system: str, messages: List[dict]) -> str:
        retrying = retry(
            reraise=True,
            stop=stop_after_attempt(max(1, self.settings.llm_max_retries)),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(_TransientLLMError),
        )
        try:
            return retrying(self._raw_complete)(system, messages)
        except _TransientLLMError as exc:
            raise LLMError(str(exc)) from exc


# --------------------------------------------------------------------------- #
# Anthropic (default)
# --------------------------------------------------------------------------- #
class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        try:
            import anthropic  # noqa: WPS433 (import inside to keep dep optional)
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'anthropic' package is not installed.") from exc
        self._sdk = anthropic
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,  # we own the retry policy
        )

    def _raw_complete(self, system: str, messages: List[dict]) -> str:
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                system=system,
                messages=messages,
            )
            return "".join(block.text for block in resp.content if block.type == "text")
        except (self._sdk.APITimeoutError, self._sdk.APIConnectionError, self._sdk.RateLimitError) as exc:
            raise _TransientLLMError(f"Anthropic transient error: {exc}") from exc
        except self._sdk.APIStatusError as exc:
            if getattr(exc, "status_code", 0) >= 500:
                raise _TransientLLMError(f"Anthropic 5xx: {exc}") from exc
            raise LLMError(f"Anthropic API error: {exc}") from exc


# --------------------------------------------------------------------------- #
# OpenAI (fallback)
# --------------------------------------------------------------------------- #
class OpenAIClient(LLMClient):
    provider = "openai"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.openai_api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        try:
            import openai  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover
            raise LLMError("The 'openai' package is not installed.") from exc
        self._sdk = openai
        self._client = openai.OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )

    def _raw_complete(self, system: str, messages: List[dict]) -> str:
        oai_messages = [{"role": "system", "content": system}] + messages
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.settings.llm_max_tokens,
                temperature=self.settings.llm_temperature,
                messages=oai_messages,
            )
            return resp.choices[0].message.content or ""
        except (self._sdk.APITimeoutError, self._sdk.APIConnectionError, self._sdk.RateLimitError) as exc:
            raise _TransientLLMError(f"OpenAI transient error: {exc}") from exc
        except self._sdk.APIStatusError as exc:
            if getattr(exc, "status_code", 0) >= 500:
                raise _TransientLLMError(f"OpenAI 5xx: {exc}") from exc
            raise LLMError(f"OpenAI API error: {exc}") from exc


# --------------------------------------------------------------------------- #
# Mock (offline / deterministic)
# --------------------------------------------------------------------------- #
class MockClient(LLMClient):
    """Deterministic provider that drives the full loop with no network.

    It inspects the conversation to decide, per stage, what a well-behaved model
    would return — enough to exercise planning, a tool call, synthesis, and
    self-critique end-to-end.
    """

    provider = "mock"

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.model = "mock-model"

    def _raw_complete(self, system: str, messages: List[dict]) -> str:
        blob = "\n".join(m["content"] for m in messages if isinstance(m.get("content"), str))
        user_query = _extract_after(blob, "USER QUESTION:") or "the question"
        pmids = re.findall(r"PMID:\s*(\d+)", blob)
        have_evidence = "OBSERVATION" in blob

        if "PlannerDecision" in system:
            if have_evidence:
                return json.dumps({
                    "thought": "Sufficient evidence gathered; ready to answer.",
                    "action": "answer",
                    "search_queries": [],
                    "rationale": "Retrieved abstracts cover the question.",
                })
            return json.dumps({
                "thought": "No evidence yet; I should search PubMed first.",
                "action": "search",
                "search_queries": [user_query[:120]],
                "rationale": "Need grounded, citable sources before answering.",
            })

        if "DraftAnswer" in system:
            cite = pmids[0] if pmids else "00000000"
            used = pmids[:3] or [cite]
            markers = " ".join(f"[{p}]" for p in used)
            return json.dumps({
                "answer": (
                    f"Based on the retrieved literature {markers}, here is an "
                    f"evidence-grounded summary addressing: {user_query}. "
                    "This is a deterministic mock answer used for offline testing."
                ),
                "citations_used": used,
                "confidence": "medium",
            })

        if "Critique" in system:
            return json.dumps({
                "is_grounded": True,
                "is_safe": True,
                "issues": [],
                "revision_needed": False,
                "suggested_fix": "",
            })

        return json.dumps({"note": "mock", "echo": user_query})


# --------------------------------------------------------------------------- #
# Factory with fallback
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "mock": MockClient,
}


class FallbackClient(LLMClient):
    """Wraps a primary client and, on hard failure, delegates to a secondary."""

    def __init__(self, primary: LLMClient, secondary: LLMClient):
        self.settings = primary.settings
        self.model = primary.model
        self.provider = primary.provider
        self._primary = primary
        self._secondary = secondary

    def _complete_with_retry(self, system: str, messages: List[dict]) -> str:
        try:
            return self._primary._complete_with_retry(system, messages)
        except LLMError as exc:
            logger.warning(
                "Primary provider '%s' failed (%s); falling back to '%s'.",
                self._primary.provider, exc, self._secondary.provider,
            )
            self.provider = self._secondary.provider
            self.model = self._secondary.model
            return self._secondary._complete_with_retry(system, messages)


def build_client(settings: Settings) -> LLMClient:
    """Construct the configured client, wiring an automatic fallback if possible.

    Fallback rule: if the primary is Anthropic/OpenAI and the *other* one has a
    key configured, wrap them so a total outage of one provider degrades to the
    other instead of failing the run.
    """
    primary_name = settings.llm_provider
    if primary_name not in _REGISTRY:
        raise LLMError(f"Unknown LLM_PROVIDER '{primary_name}'.")

    primary = _REGISTRY[primary_name](settings)

    secondary_name = {"anthropic": "openai", "openai": "anthropic"}.get(primary_name)
    if secondary_name:
        try:
            secondary = _REGISTRY[secondary_name](settings)
            logger.info(
                "LLM ready: primary=%s (%s), fallback=%s (%s)",
                primary.provider, primary.model, secondary.provider, secondary.model,
            )
            return FallbackClient(primary, secondary)
        except LLMError:
            logger.info("No fallback provider configured; using %s only.", primary.provider)

    logger.info("LLM ready: provider=%s, model=%s", primary.provider, primary.model)
    return primary


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> str:
    """Pull the first balanced JSON object out of a possibly-noisy reply."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unbalanced JSON object in model output.")


def _extract_after(text: str, marker: str) -> Optional[str]:
    idx = text.find(marker)
    if idx == -1:
        return None
    return text[idx + len(marker):].splitlines()[0].strip()
