"""Чистые функции очистки ответа модели. Без сети — легко тестировать."""

from __future__ import annotations

import json
import re
from typing import Any

from civ_agents.llm.errors import LLMInvalidToolArguments
from civ_agents.llm.types import FinishReason

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

# Замер на данных Freeciv: ~2,2 символа на токен (см. docs/benchmarks.md).
CHARS_PER_TOKEN = 2.2


def strip_thinking(text: str | None) -> str:
    """Убрать размышления модели из текста ответа.

    Обрабатывает три случая:
    - полный блок <think>...</think>;
    - только закрывающий </think> (thinking-модели, у которых открывающий тег
      подставляет шаблон чата) — отбрасывается всё до него;
    - незакрытый <think> (ответ обрезан посреди размышлений) — отбрасывается всё после.
    """
    if not text:
        return ""
    text = _THINK_BLOCK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.strip()


def parse_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    """Аргументы вызова инструмента: строка JSON → словарь."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LLMInvalidToolArguments(f"Аргументы инструмента — не JSON: {error}: {raw[:200]!r}") from error
    if not isinstance(value, dict):
        raise LLMInvalidToolArguments(f"Аргументы инструмента должны быть объектом, получено: {type(value).__name__}")
    return value


def normalize_finish_reason(raw: str | None, has_tool_calls: bool) -> FinishReason:
    """Причина остановки сервера → наше перечисление.

    Обрезка важнее всего: обрезанный ответ негоден, даже если в нём есть вызов.
    Вызов инструмента определяется по наличию tool_calls, а не только по finish_reason.
    """
    if raw == "length":
        return FinishReason.LENGTH
    if has_tool_calls:
        return FinishReason.TOOL_CALLS
    return FinishReason.STOP


def estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов по длине текста."""
    return int(len(text) / CHARS_PER_TOKEN)