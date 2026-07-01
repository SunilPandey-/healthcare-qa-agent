"""Lightweight evaluation harness.

Runs the agent against a JSON scenario file and scores each run on objective,
inspectable checks:

* **answered / refused** — did the agent do the right high-level thing?
* **grounding** — is every [PMID] cited in the answer one we actually retrieved?
  (This is a concrete hallucination check.)
* **has_citations** — are sources attached when they should be?
* **keyword_coverage** — a soft, qualitative signal that the answer is on-topic.
* **safety_disclaimer** — is the educational disclaimer present?

The harness prints a per-scenario table + an aggregate pass rate and can emit
JSON for CI. It runs offline with `--provider mock` (structural checks only) or
against a real provider for full keyword coverage.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Dict, List

if __package__ in (None, ""):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.agent import run_query
    from src.llm.client import LLMError
    from src.schemas import AgentResult
else:  # pragma: no cover
    from .agent import run_query
    from .llm.client import LLMError
    from .schemas import AgentResult

_PMID_MARKER = re.compile(r"\[(\d{5,9})\]")


def _retrieved_pmids(result: AgentResult) -> set:
    pmids = {c.pmid for c in result.citations}
    for step in result.trace:
        content = step.get("content", {})
        if isinstance(content, dict):
            pmids.update(str(p) for p in content.get("pmids", []))
    return pmids


def evaluate_answer(scenario: Dict, result: AgentResult) -> Dict:
    """Score a single 'answer' scenario."""
    answer_lc = result.answer.lower()

    cited = set(_PMID_MARKER.findall(result.answer))
    retrieved = _retrieved_pmids(result)
    grounding = all(p in retrieved for p in cited) if cited else False

    has_citations = len(result.citations) > 0
    kw_hits = [k for k in scenario.get("expect_keywords_any", []) if k.lower() in answer_lc]
    keyword_ok = len(kw_hits) >= scenario.get("min_keyword_hits", 1)
    disclaimer_ok = bool(result.disclaimer)

    checks = {
        "answered": result.steps_taken > 0,
        "has_citations": has_citations if scenario.get("must_cite") else True,
        "grounding_no_hallucinated_pmids": grounding if scenario.get("must_cite") else True,
        "keyword_coverage": keyword_ok,
        "safety_disclaimer": disclaimer_ok,
    }
    # Keyword coverage is a soft signal — report it, but don't fail the run on it.
    hard_checks = {k: v for k, v in checks.items() if k != "keyword_coverage"}
    return {
        "id": scenario["id"],
        "type": "answer",
        "passed": all(hard_checks.values()),
        "checks": checks,
        "keyword_hits": kw_hits,
        "citations": len(result.citations),
    }


def evaluate_refusal(scenario: Dict, result: AgentResult) -> Dict:
    """Score a single 'refusal' scenario."""
    refused = result.steps_taken == 0 and len(result.citations) == 0
    checks = {
        "refused_before_tools": refused,
        "no_citations": len(result.citations) == 0,
    }
    return {
        "id": scenario["id"],
        "type": "refusal",
        "passed": all(checks.values()),
        "checks": checks,
    }


def run_evaluation(scenarios_path: str, provider: str | None) -> Dict:
    with open(scenarios_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    scenarios: List[Dict] = data["scenarios"]

    results = []
    for sc in scenarios:
        print(f"\n>>> Running scenario '{sc['id']}' ({sc['type']}): {sc['query']}", file=sys.stderr)
        try:
            result = run_query(sc["query"], provider=provider)
        except LLMError as exc:
            results.append({"id": sc["id"], "type": sc["type"], "passed": False, "error": str(exc)})
            continue
        if sc["type"] == "refusal":
            results.append(evaluate_refusal(sc, result))
        else:
            results.append(evaluate_answer(sc, result))

    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    return {
        "provider": provider or "(from .env)",
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "results": results,
    }


def _print_report(report: Dict) -> None:
    print("\n" + "=" * 78)
    print(f"EVALUATION REPORT  —  provider={report['provider']}")
    print("=" * 78)
    for r in report["results"]:
        status = "PASS" if r.get("passed") else "FAIL"
        print(f"\n[{status}] {r['id']} ({r['type']})")
        if "error" in r:
            print(f"    error: {r['error']}")
            continue
        for check, ok in r.get("checks", {}).items():
            print(f"    {'✓' if ok else '✗'} {check}: {ok}")
        if r.get("keyword_hits") is not None:
            print(f"    · keyword hits: {r['keyword_hits']}  · citations: {r.get('citations')}")
    print("\n" + "-" * 78)
    print(f"OVERALL: {report['passed']}/{report['total']} passed  (pass rate {report['pass_rate']})")
    print("-" * 78)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate the healthcare agent against scenarios.")
    p.add_argument("--scenarios", default="tests/scenarios.json", help="Path to scenarios JSON.")
    p.add_argument("--provider", default=None, choices=["anthropic", "openai", "mock"],
                   help="Override LLM provider for the evaluation run.")
    p.add_argument("--json", action="store_true", dest="as_json", help="Emit the report as JSON.")
    args = p.parse_args(argv)

    from src.logging_config import enable_utf8_output
    enable_utf8_output()

    report = run_evaluation(args.scenarios, args.provider)
    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    # Non-zero exit if any scenario failed — useful for CI gating.
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
