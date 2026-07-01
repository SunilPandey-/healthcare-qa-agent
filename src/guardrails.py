"""Input validation and safety guard-rails.

These run *before* any LLM/tool call. They cheaply reject malformed or unsafe
input and keep us within token/cost limits, so the agent never wastes an API
call on something it should refuse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# Rough char budget for a user query (well under model context; keeps costs sane).
MAX_QUERY_CHARS = 1200
MIN_QUERY_CHARS = 3

# Requests we should refuse: individualized medical decision-making and clearly
# harmful intent. This is a lightweight lexical filter — a first line of defence,
# complemented by the model-side safety prompt and the critic agent.
_REFUSAL_PATTERNS = [
    # Individualized dosing (words may sit between "dose" and "should I").
    r"\b(what|which|how much)\b.{0,40}\b(dose|dosage)\b.{0,40}\b(should|do|can|must) i\b",
    r"\b(dose|dosage)\b.{0,40}\b(should|do|can|must) i (take|use)\b",
    r"\bhow (much|many)\b.{0,40}\bshould i (take|use)\b",
    r"\bshould i (take|stop|start|increase|decrease|switch)\b",
    # Self-diagnosis / personal medical judgement.
    r"\bdiagnose me\b",
    r"\bwhat('?s| is) wrong with me\b",
    r"\bam i (having|going to have)\b",
    # Clearly harmful intent.
    r"\bhow (to|do i|can i) (make|synthesize|produce)\b.{0,40}\b(bioweapon|nerve agent|ricin|anthrax|poison)\b",
    r"\bhow to (poison|kill)\b",
]

_REFUSAL_MESSAGE = (
    "I can't help with individualized medical decisions (such as personal dosing, "
    "starting/stopping a medication, or self-diagnosis) or unsafe requests. I can, "
    "however, summarize what the published literature says about a condition, "
    "treatment, or guideline in general terms."
)


@dataclass
class GuardResult:
    ok: bool
    cleaned_query: str = ""
    reason: str = ""
    refusal_message: str = ""


def validate_query(query: str) -> GuardResult:
    """Validate + normalize a user query; return a refusal if it violates policy."""
    if query is None:
        return GuardResult(ok=False, reason="empty", refusal_message="Please provide a question.")

    cleaned = " ".join(query.strip().split())

    if len(cleaned) < MIN_QUERY_CHARS:
        return GuardResult(ok=False, reason="too_short", refusal_message="Please provide a more specific question.")

    if len(cleaned) > MAX_QUERY_CHARS:
        # Truncate rather than reject outright — token-limit safety.
        cleaned = cleaned[:MAX_QUERY_CHARS]

    lowered = cleaned.lower()
    for pattern in _REFUSAL_PATTERNS:
        if re.search(pattern, lowered):
            return GuardResult(ok=False, reason="policy", refusal_message=_REFUSAL_MESSAGE)

    return GuardResult(ok=True, cleaned_query=cleaned)


def estimate_tokens(text: str) -> int:
    """Very rough token estimate (~4 chars/token) for pre-flight budget checks."""
    return max(1, len(text) // 4)


def enforce_token_budget(texts: List[str], budget: int) -> bool:
    """Return True if the combined estimate stays within ``budget`` tokens."""
    return sum(estimate_tokens(t) for t in texts) <= budget
