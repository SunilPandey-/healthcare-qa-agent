"""Tool abstraction + a tiny registry.

Every tool exposes a name, a human description, and a Pydantic args schema. The
registry lets the agent look tools up by name and keeps their JSON schemas in
one place for prompting. Adding a new capability (e.g. a clinical-guidelines
retriever) is a matter of implementing ``Tool.run`` and registering it.
"""
from __future__ import annotations

from typing import Dict, Type

from pydantic import BaseModel

from ..schemas import ToolResult


class Tool:
    """Base class for a callable tool."""

    name: str = "tool"
    description: str = ""
    args_model: Type[BaseModel] = BaseModel

    def run(self, args: BaseModel) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError

    def json_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_model.model_json_schema(),
        }


class ToolRegistry:
    """Name -> Tool lookup with schema export for prompts."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'. Available: {list(self._tools)}")
        return self._tools[name]

    def describe(self) -> str:
        """Compact, prompt-friendly description of all tools."""
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())
