"""Structured logging + an in-memory reasoning trace.

Two complementary things live here:

1. ``configure_logging`` — a standard console logger so every agent step is
   visible on stderr while the agent runs.
2. ``Trace`` — an ordered, structured record of every step (thoughts, tool
   calls, observations, sub-agent hand-offs). The trace is what we serialize
   into the Agent Run Report and what the evaluation harness inspects. Keeping
   it as data (not just log lines) makes the full reasoning chain inspectable
   and testable.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

_CONFIGURED = False


def enable_utf8_output() -> None:
    """Force stdout/stderr to UTF-8 so non-ASCII output (emoji, ✓, …) works.

    Windows consoles default to a legacy code page (cp1252) that raises
    UnicodeEncodeError on such characters. ``reconfigure`` is a no-op on
    platforms that already use UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):  # pragma: no cover - very old/odd streams
            pass


def configure_logging(level: str = "INFO") -> None:
    """Configure a single, idempotent console handler."""
    global _CONFIGURED
    enable_utf8_output()
    if _CONFIGURED:
        logging.getLogger().setLevel(level.upper())
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@dataclass
class TraceStep:
    """A single, structured entry in the reasoning trace."""

    index: int
    kind: str  # e.g. "plan", "thought", "tool_call", "observation", "answer", "critique"
    actor: str  # which (sub-)agent produced it, e.g. "planner", "researcher"
    content: Any
    elapsed_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "actor": self.actor,
            "content": self.content,
            "elapsed_s": round(self.elapsed_s, 3),
        }


@dataclass
class Trace:
    """Ordered collection of reasoning steps for one agent run."""

    steps: List[TraceStep] = field(default_factory=list)
    _start: float = field(default_factory=time.monotonic)
    _logger: logging.Logger = field(default_factory=lambda: get_logger("agent.trace"))

    def add(self, kind: str, actor: str, content: Any) -> TraceStep:
        step = TraceStep(
            index=len(self.steps) + 1,
            kind=kind,
            actor=actor,
            content=content,
            elapsed_s=time.monotonic() - self._start,
        )
        self.steps.append(step)
        preview = self._preview(content)
        self._logger.info("[step %d] %-11s (%s) %s", step.index, kind, actor, preview)
        return step

    @staticmethod
    def _preview(content: Any, limit: int = 220) -> str:
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        text = " ".join(text.split())
        return text if len(text) <= limit else text[:limit] + "…"

    def to_list(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.steps]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_list(), indent=indent, default=str)
