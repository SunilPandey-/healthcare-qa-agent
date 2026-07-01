"""Multi-agent orchestrator implementing the reason -> plan -> act -> observe -> respond loop.

Specialist roles (bonus: multi-agent orchestration)
---------------------------------------------------
* **Planner**     — reasons (chain-of-thought) about whether more evidence is
                    needed and, if so, what to search for.
* **Researcher**  — executes the PubMed tool for the planner's queries and
                    observes the results.
* **Synthesizer** — writes an answer grounded strictly in retrieved abstracts,
                    with [PMID] citation markers.
* **Critic**      — self-reflection pass that checks grounding + safety and can
                    trigger exactly one revision.

Every step is recorded in a :class:`Trace`, so the full reasoning chain, tool
calls, and final output are inspectable and reproducible.
"""
from __future__ import annotations

from typing import Dict, List

from .config import Settings
from .guardrails import validate_query
from .llm.client import LLMClient, LLMError
from .logging_config import Trace, get_logger
from .prompts import (
    CRITIC_SYSTEM,
    DISCLAIMER,
    PLANNER_SYSTEM,
    SYNTHESIZER_SYSTEM,
    evidence_block,
)
from .schemas import (
    AgentResult,
    Citation,
    Critique,
    DraftAnswer,
    PlannerDecision,
    PubMedSearchArgs,
)
from .tools.base import ToolRegistry
from .tools.pubmed import PubMedSearchTool

logger = get_logger("agent.orchestrator")


class HealthcareAgent:
    """Coordinates the specialist agents and tools to answer a health question."""

    def __init__(self, settings: Settings, llm: LLMClient, registry: ToolRegistry):
        self.settings = settings
        self.llm = llm
        self.registry = registry

    # -- construction helper --------------------------------------------- #
    @classmethod
    def build(cls, settings: Settings, llm: LLMClient) -> "HealthcareAgent":
        registry = ToolRegistry()
        registry.register(PubMedSearchTool(settings))
        return cls(settings, llm, registry)

    # -- main entrypoint -------------------------------------------------- #
    def run(self, query: str) -> AgentResult:
        trace = Trace()

        # 1) GUARD-RAILS ------------------------------------------------- #
        guard = validate_query(query)
        if not guard.ok:
            trace.add("guardrail", "system", {"reason": guard.reason})
            return AgentResult(
                query=query,
                answer=guard.refusal_message,
                confidence="high",
                provider=self.llm.provider,
                model=self.llm.model,
                steps_taken=0,
                trace=trace.to_list(),
                disclaimer=DISCLAIMER,
            )
        query = guard.cleaned_query
        trace.add("input", "system", {"query": query})

        evidence: Dict[str, Citation] = {}
        searches_done = 0

        # 2) PLAN -> ACT -> OBSERVE loop --------------------------------- #
        try:
            for step in range(1, self.settings.agent_max_steps + 1):
                force_answer = step == self.settings.agent_max_steps
                decision = self._plan(query, evidence, searches_done, force_answer, trace)

                if decision.action == "answer" or force_answer:
                    break

                for q in decision.search_queries[:3]:
                    self._research(q, evidence, trace)
                searches_done += 1

            # 3) RESPOND (synthesize) ------------------------------------ #
            draft = self._synthesize(query, evidence, trace)

            # 4) SELF-CRITIQUE (+ optional single revision) -------------- #
            draft = self._critique_and_maybe_revise(query, evidence, draft, trace)

        except LLMError as exc:
            # Fallback response: never crash the caller on a model outage.
            logger.error("Agent run failed: %s", exc)
            trace.add("error", "system", str(exc))
            return AgentResult(
                query=query,
                answer=(
                    "I'm unable to reach the language model right now, so I can't "
                    "complete a grounded answer. Please retry shortly. "
                    + (f"Retrieved {len(evidence)} source(s) before the failure."
                       if evidence else "")
                ),
                confidence="low",
                citations=list(evidence.values()),
                provider=self.llm.provider,
                model=self.llm.model,
                steps_taken=len(trace.steps),
                trace=trace.to_list(),
                disclaimer=DISCLAIMER,
            )

        used = [evidence[p] for p in draft.citations_used if p in evidence]
        final_citations = used or list(evidence.values())
        result = AgentResult(
            query=query,
            answer=draft.answer,
            confidence=draft.confidence,
            citations=final_citations,
            steps_taken=len(trace.steps),
            provider=self.llm.provider,
            model=self.llm.model,
            trace=trace.to_list(),
            disclaimer=DISCLAIMER,
        )
        trace.add("final", "system", {"citations": len(final_citations), "confidence": draft.confidence})
        return result

    # ------------------------------------------------------------------ #
    # Specialist steps
    # ------------------------------------------------------------------ #
    def _plan(
        self,
        query: str,
        evidence: Dict[str, Citation],
        searches_done: int,
        force_answer: bool,
        trace: Trace,
    ) -> PlannerDecision:
        observation = (
            f"OBSERVATION: {len(evidence)} article(s) retrieved across {searches_done} search(es)."
            if searches_done
            else "No searches run yet."
        )
        hint = (
            "\nNOTE: step budget reached — you should choose action='answer'."
            if force_answer else ""
        )
        user = (
            f"USER QUESTION: {query}\n\n"
            f"EVIDENCE GATHERED SO FAR:\n{evidence_block(list(evidence.values()))}\n\n"
            f"{observation}{hint}"
        )
        decision = self.llm.complete_json(
            PLANNER_SYSTEM, [{"role": "user", "content": user}], PlannerDecision
        )
        trace.add("plan", "planner", {
            "thought": decision.thought,
            "action": decision.action,
            "search_queries": decision.search_queries,
        })
        return decision

    def _research(self, search_query: str, evidence: Dict[str, Citation], trace: Trace) -> None:
        tool = self.registry.get("pubmed_search")
        trace.add("tool_call", "researcher", {"tool": tool.name, "query": search_query})
        result = tool.run(PubMedSearchArgs(query=search_query, max_results=4))
        if not result.ok:
            trace.add("observation", "researcher", {"error": result.error})
            return
        new = 0
        for c in result.citations:
            if c.pmid and c.pmid not in evidence:
                evidence[c.pmid] = c
                new += 1
        trace.add("observation", "researcher", {
            "summary": result.summary,
            "new_citations": new,
            "pmids": [c.pmid for c in result.citations],
        })

    def _synthesize(self, query: str, evidence: Dict[str, Citation], trace: Trace) -> DraftAnswer:
        user = (
            f"USER QUESTION: {query}\n\n"
            f"EVIDENCE (cite ONLY these PMIDs):\n{evidence_block(list(evidence.values()))}"
        )
        draft = self.llm.complete_json(
            SYNTHESIZER_SYSTEM, [{"role": "user", "content": user}], DraftAnswer
        )
        trace.add("answer", "synthesizer", {
            "draft": draft.answer,
            "citations_used": draft.citations_used,
            "confidence": draft.confidence,
        })
        return draft

    def _critique_and_maybe_revise(
        self, query: str, evidence: Dict[str, Citation], draft: DraftAnswer, trace: Trace
    ) -> DraftAnswer:
        user = (
            f"USER QUESTION: {query}\n\n"
            f"EVIDENCE:\n{evidence_block(list(evidence.values()))}\n\n"
            f"DRAFT ANSWER:\n{draft.answer}\n\n"
            f"CITED PMIDS: {draft.citations_used}"
        )
        critique = self.llm.complete_json(
            CRITIC_SYSTEM, [{"role": "user", "content": user}], Critique
        )
        trace.add("critique", "critic", {
            "is_grounded": critique.is_grounded,
            "is_safe": critique.is_safe,
            "revision_needed": critique.revision_needed,
            "issues": critique.issues,
        })

        if not critique.revision_needed:
            return draft

        revise_user = (
            f"USER QUESTION: {query}\n\n"
            f"EVIDENCE (cite ONLY these PMIDs):\n{evidence_block(list(evidence.values()))}\n\n"
            f"Your previous draft had issues. FIX THIS: {critique.suggested_fix}\n"
            f"Previous draft:\n{draft.answer}"
        )
        revised = self.llm.complete_json(
            SYNTHESIZER_SYSTEM, [{"role": "user", "content": revise_user}], DraftAnswer
        )
        trace.add("revision", "synthesizer", {
            "revised": revised.answer,
            "citations_used": revised.citations_used,
        })
        return revised
