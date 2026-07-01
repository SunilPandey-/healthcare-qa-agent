"""CLI entrypoint for the Healthcare Q&A agent.

Usage
-----
    python src/agent.py --domain healthcare --query "What are the latest treatment options for Type 2 diabetes?"
    python src/agent.py --query "..." --provider mock          # offline, no API key
    python src/agent.py --query "..." --show-trace             # print the full reasoning trace
    python src/agent.py --query "..." --json                   # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import sys

# Allow running both as `python src/agent.py` and `python -m src.agent`.
if __package__ in (None, ""):
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.config import get_settings, Settings
    from src.llm.client import build_client, LLMError
    from src.logging_config import configure_logging
    from src.orchestrator import HealthcareAgent
    from src.schemas import AgentResult
else:  # pragma: no cover
    from .config import get_settings, Settings
    from .llm.client import build_client, LLMError
    from .logging_config import configure_logging
    from .orchestrator import HealthcareAgent
    from .schemas import AgentResult


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Healthcare Q&A agent — grounded answers from PubMed literature.",
    )
    p.add_argument("--query", "-q", required=True, help="The health question to answer.")
    p.add_argument(
        "--domain", default="healthcare",
        choices=["healthcare"],
        help="Agent domain (only 'healthcare' is implemented).",
    )
    p.add_argument(
        "--provider", default=None,
        choices=["anthropic", "openai", "mock"],
        help="Override LLM_PROVIDER for this run.",
    )
    p.add_argument("--show-trace", action="store_true", help="Print the full reasoning trace.")
    p.add_argument("--json", action="store_true", dest="as_json", help="Emit the result as JSON.")
    return p.parse_args(argv)


def run_query(query: str, provider: str | None = None) -> AgentResult:
    """Programmatic entrypoint (also used by the evaluation harness)."""
    settings = get_settings()
    if provider:
        # Build a fresh settings object so overrides don't mutate the cached one.
        settings = Settings(**{**settings.model_dump(), "llm_provider": provider})
    configure_logging(settings.log_level)
    llm = build_client(settings)
    agent = HealthcareAgent.build(settings, llm)
    return agent.run(query)


def main(argv=None) -> int:
    from src.logging_config import enable_utf8_output
    enable_utf8_output()
    args = parse_args(argv)
    try:
        result = run_query(args.query, provider=args.provider)
    except LLMError as exc:
        print(f"[configuration error] {exc}", file=sys.stderr)
        print(
            "Hint: set a provider key in .env, or run offline with `--provider mock`.",
            file=sys.stderr,
        )
        return 2

    if args.as_json:
        print(json.dumps(result.model_dump(), indent=2))
        return 0

    print("\n" + "=" * 78)
    print(f"Q: {result.query}")
    print(f"(provider={result.provider}, model={result.model}, steps={result.steps_taken})")
    print("=" * 78 + "\n")
    print(result.to_display())

    if args.show_trace:
        print("\n" + "-" * 78)
        print("REASONING TRACE")
        print("-" * 78)
        for s in result.trace:
            print(f"[{s['index']:>2}] {s['kind']:<11} ({s['actor']}) @{s['elapsed_s']}s")
            print(f"     {json.dumps(s['content'], default=str)[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
