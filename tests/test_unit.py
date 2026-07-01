"""Deterministic, offline unit tests (no network, no API key).

Covers the parts that must be correct regardless of the LLM: guard-rails, the
PubMed XML parser, JSON extraction from noisy model output, and a full agent
run driven by the deterministic mock provider.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Settings
from src.guardrails import validate_query
from src.llm.client import MockClient, _extract_json
from src.orchestrator import HealthcareAgent
from src.tools.pubmed import PubMedSearchTool


# --- Guard-rails ----------------------------------------------------------- #
def test_guardrail_accepts_normal_query():
    r = validate_query("What are treatment options for type 2 diabetes?")
    assert r.ok and r.cleaned_query


def test_guardrail_rejects_personal_dosing():
    r = validate_query("What dose should I take of metformin?")
    assert not r.ok and r.reason == "policy" and r.refusal_message


def test_guardrail_rejects_too_short():
    assert not validate_query("a").ok


def test_guardrail_truncates_overlong_query():
    r = validate_query("diabetes " * 500)
    assert r.ok and len(r.cleaned_query) <= 1200


# --- JSON extraction ------------------------------------------------------- #
def test_extract_json_from_fenced_block():
    raw = 'Sure!\n```json\n{"action": "answer"}\n```\nHope that helps.'
    assert _extract_json(raw) == '{"action": "answer"}'


def test_extract_json_nested_braces():
    raw = 'noise {"a": {"b": 1}} trailing'
    assert _extract_json(raw) == '{"a": {"b": 1}}'


# --- PubMed XML parser (no network) --------------------------------------- #
SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <Journal><Title>Journal of Testing</Title>
          <JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>A study of metformin</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Diabetes is common.</AbstractText>
          <AbstractText Label="RESULTS">Metformin helped.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_pubmed_parser_extracts_fields():
    cites = PubMedSearchTool._parse_articles(SAMPLE_XML)
    assert len(cites) == 1
    c = cites[0]
    assert c.pmid == "12345678"
    assert c.title == "A study of metformin"
    assert c.journal == "Journal of Testing"
    assert c.year == "2023"
    assert "BACKGROUND" in c.abstract and "Metformin helped" in c.abstract
    assert c.url.endswith("/12345678/")


# --- Full agent loop with the mock provider (offline) --------------------- #
@pytest.fixture
def mock_settings():
    return Settings(_env_file=None, llm_provider="mock")


def test_agent_runs_end_to_end_with_mock(mock_settings, monkeypatch):
    # Avoid any real network by stubbing the PubMed tool.
    from src.schemas import Citation, ToolResult

    def fake_run(self, args):
        return ToolResult(
            tool="pubmed_search", ok=True,
            citations=[Citation(pmid="11111111", title="Mock paper", year="2024",
                                abstract="Evidence about the topic.",
                                url="https://pubmed.ncbi.nlm.nih.gov/11111111/")],
            summary="Retrieved 1 article.",
        )

    monkeypatch.setattr(PubMedSearchTool, "run", fake_run)
    agent = HealthcareAgent.build(mock_settings, MockClient(mock_settings))
    result = agent.run("What are treatment options for type 2 diabetes?")

    assert result.answer
    assert result.steps_taken > 0
    assert result.citations and result.citations[0].pmid == "11111111"
    kinds = {s["kind"] for s in result.trace}
    assert {"plan", "tool_call", "observation", "answer", "critique"}.issubset(kinds)


def test_agent_refuses_unsafe_query_before_tools(mock_settings):
    agent = HealthcareAgent.build(mock_settings, MockClient(mock_settings))
    result = agent.run("What dose of metformin should I take?")
    assert result.steps_taken == 0
    assert not result.citations
    assert "can't help" in result.answer.lower() or "cannot" in result.answer.lower()
