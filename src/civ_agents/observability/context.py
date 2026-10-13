"""Контекст вызова модели: какой ход и какой агент сейчас работают.

Агент оборачивает свою работу в `call_context(turn=5, agent="economist")`,
и все вызовы модели внутри автоматически получают эти поля в журнале.
Клиенту модели при этом ничего не нужно знать об агентах и ходах.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_context: ContextVar[dict[str, Any]] = ContextVar("llm_call_context", default={})


@contextmanager
def call_context(**fields: Any) -> Iterator[None]:
    """Добавить поля к контексту на время блока. Вложенные блоки дополняют внешние."""
    token = _context.set({**_context.get(), **fields})
    try:
        yield
    finally:
        _context.reset(token)


def current_context() -> dict[str, Any]:
    """Копия текущего контекста."""
    return dict(_context.get())