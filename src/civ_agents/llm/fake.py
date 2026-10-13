"""Поддельный клиент модели для тестов и прогонов без видеокарты."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from civ_agents.llm.errors import LLMError, LLMNoToolCall
from civ_agents.llm.types import FinishReason, LLMResponse, Message, ToolCall, ToolSpec, Usage


@dataclass
class FakeCall:
    """Что получил FakeLLM в одном вызове — для проверок в тестах."""

    messages: list[Message]
    tools: list[ToolSpec]
    require_tool: bool


@dataclass
class FakeLLM:
    """Реализация LLMClient, которая возвращает заранее заданные ответы по очереди.

    Элемент очереди — готовый LLMResponse или исключение LLMError, которое нужно поднять.
    """

    script: list[LLMResponse | LLMError] = field(default_factory=list)
    model: str = "fake"
    calls: list[FakeCall] = field(default_factory=list)

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        require_tool: bool = False,
        deadline: float | None = None,
    ) -> LLMResponse:
        self.calls.append(FakeCall(list(messages), list(tools or []), require_tool))
        if not self.script:
            raise AssertionError(f"FakeLLM: очередь ответов пуста (вызов №{len(self.calls)})")
        item = self.script.pop(0)
        if isinstance(item, LLMError):
            raise item
        if require_tool and not item.tool_calls:
            raise LLMNoToolCall("Ожидался вызов инструмента, модель ответила текстом", response=item)
        return item

    # --- Удобные конструкторы ответов ------------------------------------

    @staticmethod
    def text(content: str) -> LLMResponse:
        return LLMResponse(text=content, finish_reason=FinishReason.STOP, usage=Usage(prompt_tokens=10, completion_tokens=5))

    @staticmethod
    def tool(name: str, call_id: str = "call_0", **arguments: object) -> LLMResponse:
        return LLMResponse(
            text="",
            tool_calls=[ToolCall(id=call_id, name=name, arguments=dict(arguments))],
            finish_reason=FinishReason.TOOL_CALLS,
            usage=Usage(prompt_tokens=10, completion_tokens=5),
        )