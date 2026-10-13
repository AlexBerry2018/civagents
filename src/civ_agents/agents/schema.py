"""Структурированный ответ агента: JSON по схеме, проверка, понятные ошибки для переспроса."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError


class Decision(BaseModel):
    type: str  # unit / city / tech / gov
    id: str | int
    action: str


class Plan(BaseModel):
    actions: list[Decision] = Field(default_factory=list)
    strategy: str | None = None  # только у правителя


class PlanError(ValueError):
    """Ответ модели не разобрать. Текст ошибки уходит модели при переспросе."""


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

FORMAT_HINT = (
    'Reply with JSON only: {"actions": [{"type": "unit", "id": 101, "action": "build city"}]}. '
    "Use exact action names from the lists. Empty list if nothing to do."
)


def parse_plan(text: str) -> Plan:
    """Текст модели → Plan. Терпит обёртку ```json``` и текст вокруг JSON."""
    if not text or not text.strip():
        raise PlanError("Empty answer. " + FORMAT_HINT)
    if match := _FENCE.search(text):
        text = match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise PlanError("No JSON object found. " + FORMAT_HINT)
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise PlanError(f"Invalid JSON: {error.msg}. " + FORMAT_HINT) from error
    if isinstance(data, dict) and "actions" not in data and {"type", "id", "action"} <= data.keys():
        data = {"actions": [data]}  # одно действие без обёртки — частая ошибка маленьких моделей
    try:
        return Plan.model_validate(data)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise PlanError(f"Wrong JSON structure at '{where}': {first['msg']}. " + FORMAT_HINT) from error
