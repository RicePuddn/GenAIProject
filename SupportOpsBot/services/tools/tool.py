"""Model-facing tool contract shared by SupportOpsBot tool implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from SupportOpsBot.services.structure.structure import ToolArguments


@dataclass(frozen=True)
class Tool:
    """A validated tool definition and its implementation."""

    name: str
    arguments: type[ToolArguments]
    description: str
    run: Callable[[dict[str, Any]], dict[str, Any]]
    data: bool = True
    output_model: type[BaseModel] | None = None

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": " ".join(self.description.split()),
                "parameters": _untitled(self.arguments.model_json_schema()),
            },
        }


def _untitled(schema: Any) -> Any:
    """Remove generated Pydantic titles while retaining a user property called `title`."""
    if isinstance(schema, dict):
        return {
            key: _untitled(value)
            for key, value in schema.items()
            if not (
                key == "title"
                and isinstance(value, str)
                and any(keyword in schema for keyword in ("type", "properties", "anyOf", "$ref"))
            )
        }
    if isinstance(schema, list):
        return [_untitled(value) for value in schema]
    return schema
