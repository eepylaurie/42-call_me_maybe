"""Pydantic data models for the function-calling project."""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class ParameterSpec(BaseModel):
    """Type specification for one parameter or return value."""

    type: str


class FunctionDefinition(BaseModel):
    """One callable function the model may choose to invoke."""

    name: str
    description: str = ""
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ParameterSpec | None = None


class TestPrompt(BaseModel):
    """A single natural-language request to process."""

    prompt: str


class FunctionCall(BaseModel):
    """The structured result produced for one prompt."""

    prompt: str
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
