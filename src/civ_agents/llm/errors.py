"""Исключения слоя модели.

Два семейства:
- транспортные (LLMTransportError) — проблемы связи, клиент повторяет их сам;
- содержательные (LLMContentError) — сервер ответил, но ответ негоден;
  клиент их не повторяет, решение принимает цикл агента.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from civ_agents.llm.types import LLMResponse


class LLMError(Exception):
    """Базовое исключение слоя модели."""


# --- Транспорт -------------------------------------------------------------


class LLMTransportError(LLMError):
    """Временная проблема связи с сервером модели (повторяется)."""


class LLMTimeout(LLMTransportError):
    """Сервер не ответил вовремя."""


class LLMUnavailable(LLMTransportError):
    """Нет соединения или сервер вернул 5xx / 429."""


class LLMBadRequest(LLMError):
    """Постоянная ошибка запроса (4xx): неверная модель, параметры. Не повторяется."""


# --- Содержание ------------------------------------------------------------


class LLMContentError(LLMError):
    """Сервер ответил, но ответ не годится. Не повторяется клиентом."""

    def __init__(self, message: str, response: LLMResponse | None = None) -> None:
        super().__init__(message)
        self.response = response


class LLMTruncated(LLMContentError):
    """Ответ обрезан лимитом max_tokens."""


class LLMInvalidToolArguments(LLMContentError):
    """Аргументы вызова инструмента — не JSON-объект."""


class LLMNoToolCall(LLMContentError):
    """Ожидался вызов инструмента, а модель ответила текстом."""