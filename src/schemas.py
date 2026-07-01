"""Pydantic models that define every structured contract in the system.

These schemas are the backbone of structured output parsing: the LLM is asked
to return JSON that conforms to ``PlannerDecision`` / ``FinalAnswer`` etc., and
we validate it before trusting it. Validation turns "the model said something
weird" into a catchable, recoverable error instead of a crash.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Tool contracts
# --------------------------------------------------------------------------- #
class PubMedSearchArgs(BaseModel):
    """Arguments for the PubMed search tool."""

    query: str = Field(..., description="A focused PubMed search query.")
    max_results: int = Field(default=4, ge=1, le=10)


class Citation(BaseModel):
    """A single retrieved evidence item (one PubMed article)."""

    pmid: str
    title: str
    journal: Optional[str] = None
    year: Optional[str] = None
    abstract: str = ""
    url: str = ""


class ToolResult(BaseModel):
    """Normalized result returned by any tool."""

    tool: str
    ok: bool
    citations: List[Citation] = Field(default_factory=list)
    error: Optional[str] = None
    summary: str = ""


# --------------------------------------------------------------------------- #
# Agent reasoning contracts (what the LLM must return as JSON)
# --------------------------------------------------------------------------- #
class PlannerDecision(BaseModel):
    """The planner agent's decision for the next step of the loop."""

    thought: str = Field(..., description="Brief chain-of-thought about what to do next.")
    action: Literal["search", "answer"] = Field(
        ..., description="'search' to gather more evidence, 'answer' to respond now."
    )
    search_queries: List[str] = Field(
        default_factory=list,
        description="If action=='search', 1-3 focused PubMed queries to run.",
    )
    rationale: str = Field(default="", description="Why this action was chosen.")


class DraftAnswer(BaseModel):
    """The synthesizer's grounded answer, with inline citation markers."""

    answer: str = Field(..., description="Answer grounded ONLY in provided abstracts, with [PMID] markers.")
    citations_used: List[str] = Field(
        default_factory=list, description="PMIDs actually cited in the answer."
    )
    confidence: Literal["low", "medium", "high"] = "medium"


class Critique(BaseModel):
    """The self-critique agent's verdict on a draft answer."""

    is_grounded: bool = Field(..., description="True if every claim is supported by the cited abstracts.")
    is_safe: bool = Field(..., description="True if the answer avoids unsafe/prescriptive medical advice.")
    issues: List[str] = Field(default_factory=list)
    revision_needed: bool = False
    suggested_fix: str = ""


class AgentResult(BaseModel):
    """The final object returned by a complete agent run."""

    query: str
    answer: str
    confidence: str = "medium"
    citations: List[Citation] = Field(default_factory=list)
    steps_taken: int = 0
    provider: str = ""
    model: str = ""
    trace: List[dict] = Field(default_factory=list)
    disclaimer: str = ""

    def to_display(self) -> str:
        """Human-readable rendering for the CLI."""
        lines = [self.answer.strip(), ""]
        if self.citations:
            lines.append("Sources:")
            for c in self.citations:
                meta = " · ".join(x for x in [c.journal, c.year] if x)
                suffix = f" ({meta})" if meta else ""
                lines.append(f"  [{c.pmid}] {c.title}{suffix}\n        {c.url}")
            lines.append("")
        if self.disclaimer:
            lines.append(self.disclaimer)
        return "\n".join(lines)
