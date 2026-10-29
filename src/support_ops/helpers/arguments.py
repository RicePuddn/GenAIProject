"""Strict base model and field types for tool arguments the LLM supplies."""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonEmpty = Annotated[str, Field(min_length=1)]
Amount = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
