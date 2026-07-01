"""PubMed retrieval tool built on the free NCBI E-utilities API.

Two-step retrieval:
1. ``esearch`` — turn a natural-language query into a list of PMIDs.
2. ``efetch``  — pull article metadata + abstracts for those PMIDs (XML).

No API key is required (an optional ``NCBI_API_KEY`` raises the rate limit).
Network calls are wrapped with tenacity retries + timeouts so a flaky request
does not crash the agent — the tool returns a ``ToolResult(ok=False, ...)`` that
the agent can reason about instead.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import Settings
from ..logging_config import get_logger
from ..schemas import Citation, PubMedSearchArgs, ToolResult
from .base import Tool

logger = get_logger("agent.tool.pubmed")

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedSearchTool(Tool):
    name = "pubmed_search"
    description = (
        "Search PubMed for peer-reviewed biomedical literature and return article "
        "titles, journals, years, and abstracts. Use for any clinical/medical question "
        "that should be grounded in published evidence."
    )
    args_model = PubMedSearchArgs

    def __init__(self, settings: Settings):
        self.settings = settings
        self._session = requests.Session()
        self._timeout = settings.llm_timeout_seconds

    # -- public entrypoint ------------------------------------------------ #
    def run(self, args: PubMedSearchArgs) -> ToolResult:
        try:
            pmids = self._esearch(args.query, args.max_results)
            if not pmids:
                # Robustness: verbose natural-language questions often return 0
                # hits. Retry once with a keyword-only simplification.
                simplified = _simplify_query(args.query)
                if simplified and simplified != args.query.lower():
                    logger.info("No hits for full query; retrying with '%s'.", simplified)
                    pmids = self._esearch(simplified, args.max_results)
            if not pmids:
                return ToolResult(
                    tool=self.name, ok=True, citations=[],
                    summary=f"No PubMed results for '{args.query}'.",
                )
            citations = self._efetch(pmids)
            return ToolResult(
                tool=self.name, ok=True, citations=citations,
                summary=f"Retrieved {len(citations)} article(s) for '{args.query}'.",
            )
        except requests.RequestException as exc:
            logger.error("PubMed network error: %s", exc)
            return ToolResult(tool=self.name, ok=False, error=f"PubMed request failed: {exc}")
        except ET.ParseError as exc:
            logger.error("PubMed XML parse error: %s", exc)
            return ToolResult(tool=self.name, ok=False, error=f"Could not parse PubMed response: {exc}")

    # -- E-utilities steps ------------------------------------------------ #
    def _base_params(self) -> dict:
        params = {"tool": self.settings.ncbi_tool, "email": self.settings.ncbi_email}
        if self.settings.ncbi_api_key:
            params["api_key"] = self.settings.ncbi_api_key
        return params

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def _esearch(self, query: str, max_results: int) -> List[str]:
        params = {
            **self._base_params(),
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        }
        resp = self._session.get(f"{_EUTILS}/esearch.fcgi", params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(requests.RequestException),
    )
    def _efetch(self, pmids: List[str]) -> List[Citation]:
        params = {
            **self._base_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        resp = self._session.get(f"{_EUTILS}/efetch.fcgi", params=params, timeout=self._timeout)
        resp.raise_for_status()
        return self._parse_articles(resp.text)

    # -- XML parsing ------------------------------------------------------ #
    @staticmethod
    def _parse_articles(xml_text: str) -> List[Citation]:
        root = ET.fromstring(xml_text)
        citations: List[Citation] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = _text(art.find(".//PMID"))
            title = _text(art.find(".//ArticleTitle")) or "(no title)"
            journal = _text(art.find(".//Journal/Title"))
            year = _text(art.find(".//JournalIssue/PubDate/Year")) or _text(
                art.find(".//JournalIssue/PubDate/MedlineDate")
            )
            abstract = _join_abstract(art)
            citations.append(
                Citation(
                    pmid=pmid,
                    title=title,
                    journal=journal or None,
                    year=year or None,
                    abstract=abstract,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                )
            )
        return citations


# Question framing / stopwords that hurt PubMed keyword matching.
_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "being", "been", "the", "a", "an",
    "of", "for", "to", "in", "on", "at", "and", "or", "as", "with", "by",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "there", "evidence", "latest", "recent", "best", "options", "option",
    "treatment", "treatments", "about", "into", "that", "this", "these", "those",
}


def _simplify_query(query: str) -> str:
    """Reduce a verbose question to salient keywords for a fallback search."""
    tokens = re.findall(r"[A-Za-z0-9\-]+", query.lower())
    kept = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return " ".join(kept) if kept else query.lower()


def _text(el: Optional[ET.Element]) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def _join_abstract(art: ET.Element, limit: int = 2000) -> str:
    """Concatenate (possibly multi-section) abstract text, truncated for prompts."""
    parts = []
    for node in art.findall(".//Abstract/AbstractText"):
        label = node.get("Label")
        body = "".join(node.itertext()).strip()
        if not body:
            continue
        parts.append(f"{label}: {body}" if label else body)
    text = " ".join(parts)
    return text[:limit] + "…" if len(text) > limit else text
