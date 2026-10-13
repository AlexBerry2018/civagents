"""Типы слоя модели. Остальная система работает только с ними, а не с типами SDK."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """Вызов инструмента моделью. Аргументы уже разобраны в словарь."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Одно сообщение диалога."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # только у assistant
    tool_call_id: str | None = None  # только у tool: на какой вызов это ответ

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> Message:
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


class ToolSpec(BaseModel):
    """Описание инструмента для модели: имя, описание, JSON-схема параметров."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class FinishReason(str, Enum):
    STOP = "stop"  # модель закончила текстом
    TOOL_CALLS = "tool_calls"  # модель вызвала инструмент(ы)
    LENGTH = "length"  # ответ обрезан лимитом max_tokens


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # часть промпта, взятая из кэша сервера


class LLMResponse(BaseModel):
    """Итог одного вызова модели."""

    text: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    model: str = ""
    attempts: int = 1

    def as_message(self) -> Message:
        """Ответ в виде сообщения для истории диалога."""
        return Message.assistant(self.text, self.tool_calls)