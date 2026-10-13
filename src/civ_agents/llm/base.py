"""Интерфейс клиента модели и запись о вызове для наблюдателя."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from civ_agents.llm.errors import LLMError
from civ_agents.llm.types import LLMResponse, Message, ToolSpec


class LLMClient(Protocol):
    """То, что нужно агентам от модели. Реализации: OpenAICompatClient, FakeLLM."""

    @property
    def model(self) -> str: ...

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        require_tool: bool = False,
        deadline: float | None = None,
    ) -> LLMResponse:
        """Отправить диалог модели.

        require_tool — ответ без вызова инструмента считается ошибкой (LLMNoToolCall).
        deadline — момент time.monotonic(), после которого повторять запрос нельзя.
        """
        ...


@dataclass
class CallRecord:
    """Всё об одном вызове модели — для журнала, метрик и трейсов."""

    model: str
    messages: list[Message]
    tools: list[ToolSpec]
    latency_s: float
    attempts: int
    response: LLMResponse | None = None
    error: LLMError | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


Observer = Callable[[CallRecord], None]